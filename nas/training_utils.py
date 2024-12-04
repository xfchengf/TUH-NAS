from torch import nn
from collections import defaultdict
from collections import deque
import torch
import math
from bisect import bisect_right
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import SGD
import logging
import os


def conv_bn(inp, oup, kernel, stride, padding, affine=True, activate_f='leaky'):
    layers = [
        nn.Conv3d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm3d(oup, affine=affine)
    ]
    if activate_f == 'leaky':
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    return nn.Sequential(*layers)


def sep_bn(inp, oup, rate=1):
    return nn.Sequential(
        nn.Conv3d(inp, inp, 3, stride=1,
                  padding=rate, dilation=rate, groups=inp,
                  bias=False),
        nn.BatchNorm3d(inp),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(inp, oup, 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.LeakyReLU(negative_slope=0.2, inplace=True))


class SmoothedValue(object):
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if v != v:
                continue
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return AttributeError("Attribute does not exist")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


class MetricLogger2(object):  # 多卡并行时使用
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                shape = v.shape
                v = torch.sum(v, dim=0) / shape[0]
                v = v.item()
            assert isinstance(v, (float, int))
            if v != v:
                continue
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return AttributeError("Attribute does not exist")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


class WarmupMultiStepLR(LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class PolynormialLR(LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iter) ** self.power
                for base_lr in self.base_lrs]


class PolyCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, max_iter, t_max, eta_min=0, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        self.t_max = t_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.t_max)) / 2
                * (1 - self.last_epoch / self.max_iter) ** self.power
                for base_lr in self.base_lrs]


class OptimizerDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def state_dict(self):
        return [optim.state_dict() for optim in self.values()]

    def load_state_dict(self, state_dicts):
        for state_dict, optim in zip(state_dicts, self.values()):
            optim.load_state_dict(state_dict)
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()


def make_optimizer(cfg, model):
    if cfg.SEARCH.SEARCH_ON:
        return make_search_optimizers(cfg, model)
    else:
        return make_normal_optimizer(cfg, model)


def make_normal_optimizer(cfg, model):
    params = []
    lr = 0
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.TRAIN.INIT_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})
    optimizer = SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_search_optimizers(cfg, model):
    optim_w = torch.optim.SGD(model.w_parameters(),
                              lr=cfg.SOLVER.SEARCH.LR_START,
                              momentum=cfg.SOLVER.SEARCH.MOMENTUM,
                              weight_decay=cfg.SOLVER.SEARCH.WEIGHT_DECAY)
    optim_a = torch.optim.Adam(model.a_parameters(),
                               lr=cfg.SOLVER.SEARCH.LR_A,
                               weight_decay=cfg.SOLVER.SEARCH.WD_A)
    return OptimizerDict(optim_w=optim_w, optim_a=optim_a)


def make_search_lr_scheduler(cfg, optimizer_dict):
    optimizer = optimizer_dict['optim_w']
    return PolyCosineAnnealingLR(
        optimizer,
        max_iter=cfg.SOLVER.MAX_EPOCH,
        t_max=cfg.SOLVER.SEARCH.T_MAX,
        eta_min=cfg.SOLVER.SEARCH.LR_END
    )


def make_lr_scheduler(cfg, optimizer):
    if cfg.SEARCH.SEARCH_ON:
        return make_search_lr_scheduler(cfg, optimizer)
    if cfg.SOLVER.SCHEDULER == 'poly':
        power = cfg.SOLVER.TRAIN.POWER
        max_iter = cfg.SOLVER.TRAIN.MAX_ITER
        return PolynormialLR(optimizer, max_iter, power)
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )


class Checkpointer(object):
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return
        if not self.save_to_disk:
            return
        data = {
            "model": self.model.state_dict()
        }
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)
        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            f = self.get_checkpoint_file()
        if not f:
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        checkpoint.pop('optimizer')
        checkpoint.pop('scheduler')
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    @staticmethod
    def _load_file(f):
        return torch.load(f, weights_only=True)

    def _load_model(self, checkpoint):
        model_state_dict = checkpoint.pop("model")
        try:
            self.model.load_state_dict(model_state_dict)
        except RuntimeError:
            self.model.module.load_state_dict(model_state_dict)

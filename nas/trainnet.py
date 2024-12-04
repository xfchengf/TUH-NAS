import os
import torch
from torch import nn
from torch.nn import functional as F
from .cell import FixCell
from .searchnet import HSIsearchnet
from .training_utils import conv_bn
from .decoders import build_decoder
from .HSIAM import HSIAM
from .Loss import CombinedLoss


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(x):
        return x


def get_genotype_from_search(cfg):
    search_cfg = cfg.clone()
    search_cfg.defrost()
    search_cfg.merge_from_list(['MODEL.META_ARCHITECTURE', 'trainnet',
                                'MODEL.AFFINE', True,
                                'SEARCH.SEARCH_ON', True])
    model = HSIsearchnet(search_cfg)
    search_result_dir = os.path.join(cfg.OUTPUT_DIR, '{}'.format(cfg.DATASET.DATA_SET), 'search/models/model_best.pth')
    ckpt = torch.load(search_result_dir, weights_only=True)
    restore = {k: v for k, v in ckpt['model'].items() if 'arch' in k}
    model.load_state_dict(restore, strict=False)
    return model.genotype()


class HSItrainnet(nn.Module):
    def __init__(self, cfg):
        super(HSItrainnet, self).__init__()
        geno_file = os.path.join(cfg.OUTPUT_DIR, '{}'.format(cfg.DATASET.DATA_SET),
                                 'search/models/model_best.geno')
        if os.path.exists(geno_file):
            print("Loading genotype from {}".format(geno_file))
            genotype = torch.load(geno_file, map_location=torch.device("cpu"), weights_only=True)
        else:
            genotype = get_genotype_from_search(cfg)
            print("Saving genotype to {}".format(geno_file))
            torch.save(genotype, geno_file)
        gene_cell = genotype
        self.genotype = genotype
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.stem0 = conv_bn(1, 32, (5, 3, 3), (1, 1, 1), (0, 1, 1))
        self.stem1 = conv_bn(32, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        skip_conv = [conv_bn(32, 48, (3, 1, 1), (2, 1, 1), (0, 0, 0), affine=False)]
        for i in range(1, self.num_layers):
            skip_conv.append(
                conv_bn(48, 48, (2, 1, 1), (2, 1, 1), (0, 0, 0),
                        affine=False))
        self.skip_conv = nn.Sequential(*skip_conv)
        self.cell = nn.ModuleList()
        self.cell_router = nn.ModuleList()
        self.scaler = nn.ModuleList()
        for layer, (gene) in enumerate(zip(gene_cell)):
            self.cell.append(FixCell(gene, 8))
            self.cell_router.append(conv_bn(32, 8, (2, 1, 1), (2, 1, 1), (0, 0, 0), activate_f='None'))
        self.decoder = build_decoder(cfg)
        self.criteria = CombinedLoss(d=0.2, e=0.4, f=0.4, ignore_lb=-1, focal_gamma=2.0)
        self.hsiam = HSIAM(a=0.4, b=0.3, c=0.3, in_channels=32)

    def genotype(self):
        return self.genotype

    def forward(self, images, targets=None, drop_prob=-1):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")
        h0 = self.stem0(images)
        h1 = self.stem1(F.leaky_relu(h0, negative_slope=0.2))
        endpoint = self.skip_conv(h1)
        for i, [cell, cell_router] in enumerate(zip(self.cell, self.cell_router)):
            h0 = h1
            h1 = cell(cell_router(h0), cell_router(h1), drop_prob)
            h1 = self.hsiam(h1)
        pred = self.decoder([endpoint, F.leaky_relu(h1, negative_slope=0.2)])
        if self.training:
            loss = self.criteria(pred, targets)
            return pred, loss
        else:
            return pred

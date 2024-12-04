import torch
import torch.nn.functional as F
from torch import nn
from .cell import CellSPEU, CellSPAU, CellFFU
from .genotypes import PRIMITIVES
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


class HSIsearchnet(nn.Module):   # 末尾加HSIAM
    def __init__(self, cfg):
        super(HSIsearchnet, self).__init__()
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.primitives_speu = PRIMITIVES[cfg.MODEL.PRIMITIVES_SPEU]
        self.primitives_spau = PRIMITIVES[cfg.MODEL.PRIMITIVES_SPAU]
        self.primitives_FFU = PRIMITIVES[cfg.MODEL.PRIMITIVES_FFU]
        self.activatioin_f = cfg.MODEL.ACTIVATION_F
        affine = cfg.MODEL.AFFINE
        self.stem0 = conv_bn(1, 32, (5, 3, 3), (1, 1, 1), (2, 1, 1), affine)
        self.stem1 = conv_bn(32, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1), affine)
        self.cells_speu = nn.ModuleList()
        self.cells_spau = nn.ModuleList()
        self.cells_FFU = nn.ModuleList()
        self.cell_router_speu = nn.ModuleList()
        self.cell_router_spau = nn.ModuleList()
        self.cell_configs = []

        for s in range(0, self.num_layers):
            self.cells_speu.append(CellSPEU(self.num_blocks, 8, self.primitives_speu, affine=affine))
            self.cells_spau.append(CellSPAU(self.num_blocks, 8, self.primitives_spau, affine=affine))
            self.cells_FFU.append(CellFFU(self.num_blocks, 8, self.primitives_FFU, affine=affine))
            self.cell_router_speu.append(
                conv_bn(32, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=affine, activate_f='None'))
            self.cell_router_spau.append(
                conv_bn(32, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=affine, activate_f='None'))

        self.decoder = build_decoder(cfg)

        k = sum(2 + i for i in range(self.num_blocks))
        num_ops = len(self.primitives_spau)
        self.arch_alphas = nn.Parameter(torch.ones(self.num_layers, k, num_ops))
        self.arch_betas = nn.Parameter(torch.ones(self.num_layers, k, num_ops))
        self.arch_deltas = nn.Parameter(torch.ones(self.num_layers, k, num_ops))
        self.arch_gammas = nn.Parameter(torch.cat((
            torch.tensor([[1.0, 1.0, 0]]),
            torch.tensor([[1.0, 1.0, 0]]),
            torch.tensor([[1.0, 1.0, 0]]),
            torch.tensor([[0, 0, 1.0]])), dim=0))
        self.criteria = CombinedLoss(d=0.2, e=0.4, f=0.4, ignore_lb=-1, focal_gamma=2.0)
        self.hsiam = HSIAM(a=0.4, b=0.3, c=0.3, in_channels=32)

    def w_parameters(self):
        return [param for name, param in self.named_parameters() if 'arch' not in name and param.requires_grad]

    def a_parameters(self):
        return [param for name, param in self.named_parameters() if 'arch' in name]

    def scores(self):
        alphas = F.softmax(self.arch_alphas, dim=-1)
        betas = F.softmax(self.arch_betas, dim=-1)
        deltas = F.softmax(self.arch_deltas, dim=-1)
        gammas = F.softmax(self.arch_gammas, dim=-1)
        return alphas, betas, deltas, gammas

    def forward(self, images, targets=None):
        alphas, betas, deltas, gammas = self.scores()
        input_0 = self.stem0(images)
        input_1 = self.stem1(F.leaky_relu(input_0, negative_slope=0.2))
        hidden_states = []
        for s in range(self.num_layers):
            cell_weights_speu = alphas[s]
            cell_weights_spau = betas[s]
            cell_weights_FFU = deltas[s]
            cell_weights_arch = gammas[s]
            input_0 = self.cell_router_speu[s](input_0)
            input_1 = self.cell_router_spau[s](input_1)
            out0 = self.cells_speu[s](input_0, input_1, cell_weights_speu) * cell_weights_arch[0]
            out0 = self.cell_router_spau[s](out0)
            out1 = self.cells_spau[s](out0, input_1, cell_weights_spau) * cell_weights_arch[1]
            out1 = self.cell_router_spau[s](out1)
            out2 = self.cells_FFU[s](out0, out1, cell_weights_FFU) * cell_weights_arch[2]
            out21 = self.hsiam(out2)
            hidden_states.append(out21)
            input_0 = out2
            input_1 = out21
        pred = self.decoder(hidden_states)
        if self.training:
            loss = self.criteria(pred, targets)
            return loss
        else:
            return pred

    def genotype(self):
        alphas, betas, deltas, gammas = self.scores()
        gene_cell = []
        for s in range(self.num_layers):
            if s == 3:
                gene_cell.append(self.cells_FFU[s].genotype(deltas[s]))
            elif s == 0 or s == 1 or s == 2:
                if gammas[s][0] >= gammas[s][1]:
                    gene_cell.append(self.cells_speu[s].genotype(alphas[s]))
                else:
                    gene_cell.append(self.cells_spau[s].genotype(betas[s]))
        return gene_cell

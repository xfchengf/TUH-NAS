import torch
import torch.nn as nn
from .operations import OPS, Identity


class MixedOp(nn.Module):
    def __init__(self, c, primitives, affine=True):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](c, affine)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class CellBase(nn.Module):
    def __init__(self, blocks, c, primitives, empty_a1=False, affine=True):
        super().__init__()
        self._steps = blocks
        self._multiplier = blocks
        self._ops = nn.ModuleList()
        self._empty_h1 = empty_a1
        self._primitives = primitives

        for i in range(self._steps):
            for j in range(2 + i):
                op = MixedOp(c, primitives, affine)
                self._ops.append(op)

        self.conv_end = nn.Sequential(
            nn.Conv3d(24, 32, (1, 1, 1), (1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, s0, s1, weights):
        states = [s1, s0]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states)
                    if not self._empty_h1 or j > 0)
            offset += len(states)
            states.append(s)
        out = torch.cat(states[-self._multiplier:], dim=1)
        out = self.conv_end(out)
        return out

    def genotype(self, weights):
        gene = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            w = weights[start:end].clone().detach()
            edges = sorted(range(i + 2),
                           key=lambda x: -max(w[x][s]
                                              for s in range(len(w[x]))
                                              if s != self._primitives.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(w[j])):
                    if k != self._primitives.index('none'):
                        if k_best is None or w[j][k] > w[j][k_best]:
                            k_best = k
                gene.append((self._primitives[k_best], j))
            start = end
            n += 1
        return gene


class CellSPEU(CellBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CellSPAU(CellBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CellFFU(CellBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def drop_path(x, drop_prob):
    if drop_prob > 0:
        keep_prob = 1 - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x = torch.where(mask == 1, x / keep_prob, x)
        x = torch.where(mask == 1, x, torch.zeros_like(x))
    return x


class FixCell(nn.Module):
    def __init__(self, genotype, c):
        super(FixCell, self).__init__()
        [op_names] = genotype
        self._steps = len(op_names) // 2
        self._multiplier = self._steps
        self._ops = nn.ModuleList()
        self._indices = []
        for name in op_names:
            op = OPS[name[0]](c, True)
            self._ops.append(op)
            self._indices.append(name[1])
        self._indices = tuple(self._indices)
        self.conv_end = nn.Sequential(
            nn.Conv3d(24, 32, (1, 1, 1), (1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, s0, s1, drop_prob):
        states = [s1, s0]
        for i in range(self._steps):
            s = 0
            for ind in [2 * i, 2 * i + 1]:
                op = self._ops[ind]
                h = op(states[self._indices[ind]])
                if self.training and drop_prob > 0:
                    if not isinstance(op, Identity):
                        h = drop_path(h, drop_prob)
                s = s + h
            states.append(s)
        out = torch.cat(states[-self._multiplier:], dim=1)
        out = self.conv_end(out)
        return out

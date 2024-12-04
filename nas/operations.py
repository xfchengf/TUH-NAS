import torch.nn as nn


OPS = {
    'none': lambda c, affine: Zero(),
    'skip_connect': lambda c, affine: Identity(),
    'acon_1-3': lambda c, affine: LeakyConvBN(c, c, 3, 1, affine=affine),
    'acon_1-5': lambda c, affine: LeakyConvBN(c, c, 5, 1, affine=affine),
    'acon_1-7': lambda c, affine: LeakyConvBN(c, c, 7, 1, affine=affine),
    'asep_1-3': lambda c, affine: LeakySepConv(c, c, 3, 1, affine=affine),
    'asep_1-5': lambda c, affine: LeakySepConv(c, c, 5, 1, affine=affine),
    'asep_1-7': lambda c, affine: LeakySepConv(c, c, 7, 1, affine=affine),
    'econ_3-1': lambda c, affine: LeakyConvBN(c, c, 1, 3, affine=affine),
    'econ_5-1': lambda c, affine: LeakyConvBN(c, c, 1, 5, affine=affine),
    'econ_7-1': lambda c, affine: LeakyConvBN(c, c, 1, 7, affine=affine),
    'esep_3-1': lambda c, affine: LeakySepConv(c, c, 1, 3, affine=affine),
    'esep_5-1': lambda c, affine: LeakySepConv(c, c, 1, 5, affine=affine),
    'esep_7-1': lambda c, affine: LeakySepConv(c, c, 1, 7, affine=affine),
    'con_3-3': lambda c, affine: LeakyConvBN(c, c, 3, 3, affine=affine),
    'con_5-3': lambda c, affine: LeakyConvBN(c, c, 3, 5, affine=affine),
    'con_3-5': lambda c, affine: LeakyConvBN(c, c, 5, 3, affine=affine),
    'con_5-5': lambda c, affine: LeakyConvBN(c, c, 5, 5, affine=affine),
    'con_5-7': lambda c, affine: LeakyConvBN(c, c, 7, 5, affine=affine),
    'con_7-5': lambda c, affine: LeakyConvBN(c, c, 5, 7, affine=affine),
    'con_7-7': lambda c, affine: LeakyConvBN(c, c, 7, 7, affine=affine),
    'dilated_3-1': lambda c, affine: DilatedConv1(c, c, 3, 2, affine=affine),
    'dilated_5-1': lambda c, affine: DilatedConv1(c, c, 5, 2, affine=affine),
    'dilated_7-1': lambda c, affine: DilatedConv1(c, c, 7, 2, affine=affine),
    'dilated_1-3': lambda c, affine: DilatedConv2(c, c, 3, 2, affine=affine),
    'dilated_1-5': lambda c, affine: DilatedConv2(c, c, 5, 2, affine=affine),
    'dilated_1-7': lambda c, affine: DilatedConv2(c, c, 7, 2, affine=affine),
    'dilated_3-3': lambda c, affine: DilatedConv3(c, c, 3, 2, affine=affine),
    'dilated_5-5': lambda c, affine: DilatedConv3(c, c, 5, 2, affine=affine),
    'dilated_7-7': lambda c, affine: DilatedConv3(c, c, 7, 2, affine=affine),
}


class LeakyConvBN(nn.Module):
    def __init__(self, c_in, c_out, spa_s, spe_s, affine=True):
        super(LeakyConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv3d(c_in, c_out, (1, spa_s, spa_s), padding=(0, spa_s // 2, spa_s // 2), bias=False),
            nn.Conv3d(c_out, c_out, (spe_s, 1, 1), padding=(spe_s // 2, 0, 0), bias=False),
            nn.BatchNorm3d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class LeakySepConv(nn.Module):
    def __init__(self, c_in, c_out, spa_s, spe_s, affine=True, repeats=2):
        super(LeakySepConv, self).__init__()

        def basic_op():
            return nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=False),
                nn.Conv3d(c_in, c_in, (1, spa_s, spa_s), padding=(0, spa_s // 2, spa_s // 2), groups=c_in, bias=False),
                nn.Conv3d(c_in, c_in, (spe_s, 1, 1), padding=(spe_s // 2, 0, 0), groups=c_in, bias=False),
                nn.Conv3d(c_in, c_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm3d(c_out, affine=affine),
            )

        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx),
                               basic_op())

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(x):
        return x


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    @staticmethod
    def forward(x):
        return x.mul(0.)


class DilatedConv1(nn.Module):  # 空洞卷积
    def __init__(self, c_in, c_out, spe_s, dilation, affine=True):
        super(DilatedConv1, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv3d(c_in, c_out, (spe_s, 1, 1), padding=(spe_s // 2 * dilation, 0, 0), dilation=(dilation, 1, 1),
                      bias=False),
            nn.BatchNorm3d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilatedConv2(nn.Module):  # 空洞卷积
    def __init__(self, c_in, c_out, spa_s, dilation, affine=True):
        super(DilatedConv2, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv3d(c_in, c_out, (1, spa_s, spa_s), padding=(0, spa_s // 2 * dilation, spa_s // 2 * dilation),
                      dilation=(1, dilation, dilation), bias=False),
            nn.BatchNorm3d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilatedConv3(nn.Module):  # 空洞卷积
    def __init__(self, c_in, c_out, kernel_size, dilation, affine=True):
        super(DilatedConv3, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv3d(c_in, c_out, kernel_size, padding=(kernel_size // 2) * dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

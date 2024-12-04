import torch
import numpy as np
from torch.nn import functional as F
from .training_utils import conv_bn, sep_bn
from einops import rearrange
from torch import nn, einsum


class ASPPModule(nn.Module):
    def __init__(self, inp, oup, rates, affine=True, use_gap=True, activate_f='ReLU'):
        super(ASPPModule, self).__init__()
        self.conv1 = conv_bn(inp, oup, 1, 1, 0, affine=affine, activate_f=activate_f)
        self.atrous = nn.ModuleList()
        self.use_gap = use_gap
        for rate in rates:
            self.atrous.append(sep_bn(inp, oup, rate))
        num_branches = 1 + len(rates)
        if use_gap:
            self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     conv_bn(inp, oup, 1, 1, 0, activate_f=activate_f))
            num_branches += 1
        self.conv_last = conv_bn(oup * num_branches, oup, 1, 1, 0, affine=affine, activate_f=activate_f)

    def forward(self, x):
        atrous_outs = [atrous(x) for atrous in self.atrous]
        atrous_outs.append(self.conv1(x))
        if self.use_gap:
            gap = self.gap(x)
            gap = F.interpolate(gap, size=x.size()[2:], mode='bilinear', align_corners=False)
            atrous_outs.append(gap)
        x = torch.cat(atrous_outs, dim=1)
        x = self.conv_last(x)
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class LinearBN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class Transformer_vit(nn.Module):
    def __init__(self, Dim_in, Crop_size, dim_head=32, heads=4, dropout=0.1, emb_dropout=0.1):
        super(Transformer_vit, self).__init__()
        tokens_num = Crop_size * Crop_size
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.pos_embedding = nn.Parameter(torch.randn(1, tokens_num, Dim_in))
        self.drop_out = nn.Dropout(emb_dropout)
        inner_dim = dim_head * heads
        self.attend = torch.nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(Dim_in, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, Dim_in),
            nn.Dropout(dropout)
        )
        self.ffnet = nn.Sequential(
            nn.Linear(Dim_in, dim_head * heads),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_head * heads, Dim_in),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(Dim_in)
        self.norm2 = nn.LayerNorm(Dim_in)

    def forward(self, x):
        b, c, h, w = x.shape
        sque_x = x.view(b, c, h * w).permute(0, 2, 1)
        x = sque_x + self.pos_embedding
        x = self.drop_out(x)
        qkv = self.to_qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        x = self.to_out(out) + x
        x = self.ffnet(self.norm2(x)) + x
        out = x.permute(0, 2, 1).view(b, c, h, w)
        return out


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.use_aspp = cfg.MODEL.USE_ASPP
        bxf = 48
        inp = 32
        rates = cfg.MODEL.ASPP_RATES
        self.pre_conv = ASPPModule(inp, 32, rates, use_gap=False, activate_f=self.activate_f)
        self.proj = conv_bn(bxf, 32, 1, 1, 0, activate_f=self.activate_f)
        self.transformer = Transformer_vit(Dim_in=64, Crop_size=cfg.DATASET.CROP_SIZE)
        self.pre_cls = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64))
        self.cls = nn.Conv2d(64, cfg.DATASET.CATEGORY_NUM, kernel_size=1, stride=1)

    def forward(self, x):
        x0, x1 = x
        x1 = self.pre_conv(x1)
        x0 = self.proj(x0)
        x = torch.cat((x0, x1), dim=1)
        x = x.mean(dim=2)
        x = self.transformer(x)
        x = self.pre_cls(x)
        pred = self.cls(x)
        return pred


class AutoDecoder(nn.Module):
    def __init__(self, cfg, out_strides):
        super(AutoDecoder, self).__init__()
        self.aspps = nn.ModuleList()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        bxf = 32
        affine = cfg.MODEL.AFFINE
        num_strides = len(out_strides)
        for i, out_stride in enumerate(out_strides):
            rate = out_stride
            inp = 32
            oup = bxf
            self.aspps.append(ASPPModule(inp, oup, [rate], affine=affine, use_gap=False, activate_f=self.activate_f))
            self.pre_cls = nn.Sequential(
                nn.Conv2d(bxf * num_strides, bxf * num_strides, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(bxf * num_strides),
                nn.Conv2d(bxf * num_strides, bxf * num_strides, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(bxf * num_strides),
            )
        self.cls = nn.Conv2d(bxf * num_strides, cfg.DATASET.CATEGORY_NUM, kernel_size=1, stride=1)

    def forward(self, x):
        x = [aspp(x_i) for aspp, x_i in zip(self.aspps, x)]
        x = torch.cat(x, dim=1).mean(dim=2)
        x = self.pre_cls(x)
        pred = self.cls(x)
        return pred


def build_decoder(cfg):
    if cfg.SEARCH.SEARCH_ON:
        out_strides = np.ones(cfg.MODEL.NUM_STRIDES, np.int16) * 2
        return AutoDecoder(cfg, out_strides)
    else:
        return Decoder(cfg)

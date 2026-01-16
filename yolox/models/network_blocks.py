#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from .attention import *


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )   
        return self.conv(x)


######################################## base module ########################################
# ------------------------------------------------------------
# same-padding helper（若已在文件里写过，可跳过）
def autopad(k, p=None, d=1):
    if p is not None:
        return p
    if isinstance(k, int):
        return d * (k - 1) // 2
    else:
        return tuple(d * (ki - 1) // 2 for ki in k)
# ------------------------------------------------------------
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False) #注意添加autopad，还没加
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

# class C2f(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

####改版c2f 
class C2f(nn.Module):
    """
    Faster Implementation of CSP Bottleneck with 2 convolutions.
    只保留 YOLO-X Bottleneck 支持的参数
    """
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, act="silu"):
        super().__init__()
        self.c  = int(c2 * e)                # hidden channels
        self.cv1 = Conv(c1,        2 * self.c, 1, 1, act=act)
        self.cv2 = Conv((2 + n)*self.c, c2,    1, 1, act=act)

        # Bottleneck in YOLO-X: (in, out, shortcut, expansion, depthwise, act)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, expansion=1.0, act=act)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
######################################## base module ########################################

######################################## C3 C2f DAttention end ########################################

# class Bottleneck_DAttention(Bottleneck):
#     """Standard bottleneck with DAttention."""

#     def __init__(self, c1, c2, fmapsize, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
#         super().__init__(c1, c2, shortcut, g, k, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.attention = DAttention(c2, fmapsize)
    
#     def forward(self, x):
#         return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))

# # class C3_DAttention(C3):
# #     def __init__(self, c1, c2, n=1, fmapsize=None, shortcut=False, g=1, e=0.5):
# #         super().__init__(c1, c2, n, shortcut, g, e)
# #         c_ = int(c2 * e)  # hidden channels
# #         self.m = nn.Sequential(*(Bottleneck_DAttention(c_, c_, fmapsize, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

# class C2f_DAttention(C2f):
#     def __init__(self, c1, c2, n=1, fmapsize=None, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(Bottleneck_DAttention(self.c, self.c, fmapsize, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

# ######################################## C3 C2f DAttention end ########################################

######################################## csp DAttention  ########################################

# class CSPBottleneckDAttention(nn.Module):
#     """CSP Bottleneck with DAttention."""
    
#     def __init__(self, c1, c2, fmapsize, shortcut=True, g=1, n=1, e=0.5):
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = BaseConv(c1, 2 * self.c, ksize=1, stride=1)  # 1x1 convolutions
#         self.cv2 = BaseConv(2 * self.c, c2, ksize=1, stride=1)  # Output layer
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))
#         self.attention = DAttention(c2, fmapsize)
    
#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, dim=1))  # 使用 list 转换，以便修改
#         for block in self.m:
#             y[1] = block(y[1])  # 处理第二部分
#         out = torch.cat(y, dim=1)  # 合并两个部分
#         return self.cv2(out) + self.attention(out)  # 返回结果


class CSPBottleneckDAttention(nn.Module):
    """
    CSP layer + Deformable Attention（可残差抑制）组合块，适用于 YOLOX Dark5 末端。
    默认 gamma=0，网络刚开始≈恒等映射，避免直接把预训练特征“打乱”。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fmapsize: tuple = (20, 20),      # Dark5 特征图尺寸（640 输入 → 20×20）
        n: int = 1,                      # 内部 bottleneck 次数；保持与原 base_depth 一致
        shortcut: bool = False,
        depthwise: bool = False,
        act: str = "silu",
        expansion: float = 0.50,
        n_heads: int = 8,
        n_groups: int = 4,
        offset_range_factor: int = 4,
    ):
        super().__init__()

        # 1) 常规 CSP
        self.csp = CSPLayer(
            in_channels,
            out_channels,
            n=n,
            shortcut=shortcut,
            depthwise=depthwise,
            act=act,
            expansion=expansion,
        )

        # 2) Deformable Attention
        self.attn = DAttention(
            channel=out_channels,
            q_size=fmapsize,                 # (H, W)
            n_heads=n_heads,
            n_groups=n_groups,
            stride=1,
            offset_range_factor=offset_range_factor,
            use_pe=True,
            dwc_pe=True,
            no_off=False,
        )

        # 3) 可学习缩放因子——恒等起步
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        """
        x : (B, C, H, W)  ->  same shape
        """
        out = self.csp(x)          # 原有特征变换
        out = self.attn(out)       # 加入注意力
        return x + self.gamma * out  # 残差融合；gamma=0 时保持恒等
    
######################################## csp DAttention end ########################################


class Bottleneck_DAttention(Bottleneck):
    """
    Standard Bottleneck + DAttention
    Compatible with YOLO-X Bottleneck API:
      Bottleneck(c1, c2, shortcut=True, expansion=0.5, depthwise=False, act="silu")
    """
    def __init__(self, c1, c2, fmapsize,
                 shortcut=True, e=0.5, act="silu"):
        # 调父类：expansion=e，别传多余参数
        super().__init__(c1, c2, shortcut, expansion=e, act=act)
        self.attention = DAttention(c2, fmapsize)  # 你的自定义注意力

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        y = self.attention(y)

        # 用父类的开关
        if self.use_add:          # ← 改这里
            y = x + y
        return y


class C2f_DAttention(C2f):
    """
    C2f variant that inserts DAttention inside each internal Bottleneck.
    """
    def __init__(self, c1, c2, n=1, fmapsize=None,
                 shortcut=False, e=0.5, act="silu"):
        super().__init__(c1, c2, n, shortcut, e, act)

        # 用 Bottleneck_DAttention 覆盖掉父类里那一串普通 Bottleneck
        self.m = nn.ModuleList(
            Bottleneck_DAttention(self.c, self.c, fmapsize,
                                  shortcut=shortcut, e=1.0, act=act)
            for _ in range(n)
        )
        

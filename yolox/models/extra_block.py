import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np  
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        
        self.local_pool = nn.AdaptiveAvgPool2d((patch_size, patch_size))  # 自适应池化 ，v2修改中
        
        self.mlp1 = nn.Linear(output_dim, output_dim // 2)  # 输入维度为通道数 C ，v2修改中
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        
        # self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        # self.norm = nn.LayerNorm(output_dim // 2)
        # self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1)
        # B, H, W, C = x.shape
        # P = self.patch_size

        # # Local branch
        # local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        # local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        # local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        # local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        # local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        # local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        # local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        # local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        # cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        # mask = cos_sim.clamp(0, 1)
        # local_out = local_out * mask
        # local_out = local_out @ self.top_down_transform

        # # Restore shapes
        # local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        # local_out = local_out.permute(0, 3, 1, 2)
        # local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        # output = self.conv(local_out)
        
        # v2，直接池化到固定尺寸
        B, C, H, W = x.shape
        x_pooled = self.local_pool(x)  # (B, C, patch_size, patch_size)
        x_pooled = x_pooled.view(B, C, -1)  # (B, C, patch_size*patch_size)
        x_pooled = x_pooled.permute(0, 2, 1)  # (B, patch_size*patch_size, C)

        # MLP 处理
        local_patches = self.mlp1(x_pooled)  # (B, patch_size*patch_size, output_dim // 2)
        local_patches = self.norm(local_patches)
        local_patches = self.mlp2(local_patches)  # (B, patch_size*patch_size, output_dim)

        # 注意力机制
        local_attention = F.softmax(local_patches, dim=-1)  # (B, patch_size*patch_size, output_dim)
        local_out = local_patches * local_attention

        # 提示机制
        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # (B, patch_size*patch_size, 1)
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform  # (B, patch_size*patch_size, output_dim)

        # 恢复形状
        local_out = local_out.permute(0, 2, 1)  # (B, output_dim, patch_size*patch_size)
        local_out = local_out.view(B, self.output_dim, self.patch_size, self.patch_size)  # (B, output_dim, P, P)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


######################################## Hierarchical Attention Fusion Block start ########################################

class HAFB(nn.Module):
    # Hierarchical Attention Fusion Block
    def __init__(self, inc, ouc, group=False):
        super(HAFB, self).__init__()
        ch_1, ch_2 = inc
        hidc = ouc // 2

        # self.lgb1_local = LocalGlobalAttention(hidc, 2)
        # self.lgb1_global = LocalGlobalAttention(hidc, 4)
        # self.lgb2_local = LocalGlobalAttention(hidc, 2)
        # self.lgb2_global = LocalGlobalAttention(hidc, 4)
        
        # v2，根据 patch_size 动态设置 LocalGlobalAttention 的 patch_size
        self.lgb1_local = LocalGlobalAttention(hidc, patch_size=2)  # P3 使用更小 patch
        self.lgb1_global = LocalGlobalAttention(hidc, patch_size=4)
        self.lgb2_local = LocalGlobalAttention(hidc, patch_size=2)
        self.lgb2_global = LocalGlobalAttention(hidc, patch_size=4)

        self.W_x1 = Conv(ch_1, hidc, 1, act=False)
        self.W_x2 = Conv(ch_2, hidc, 1, act=False)
        self.W = Conv(hidc, ouc, 3, g=4)

        self.conv_squeeze = Conv(ouc * 3, ouc, 1)
        self.rep_conv = RepConv(ouc, ouc, 3, g=(16 if group else 1))
        self.conv_final = Conv(ouc, ouc, 1)

    def forward(self, inputs):
        # x1, x2 = inputs

        # # 检查特征图尺寸是否满足 patch_size
        # P = 4  # LocalGlobalAttention 中的 patch_size
        # if x1.size(2) < P or x1.size(3) < P:
        #     x1 = F.interpolate(x1, size=(P, P), mode='bilinear', align_corners=False).contiguous()
        # if x2.size(2) < P or x2.size(3) < P:
        #     x2 = F.interpolate(x2, size=(P, P), mode='bilinear', align_corners=False).contiguous()

        # W_x1 = self.W_x1(x1)
        # W_x2 = self.W_x2(x2)
        # bp = self.W(W_x1 + W_x2)

        # x1 = torch.cat([self.lgb1_local(W_x1), self.lgb1_global(W_x1)], dim=1)
        # x2 = torch.cat([self.lgb2_local(W_x2), self.lgb2_global(W_x2)], dim=1)

        # return self.conv_final(self.rep_conv(self.conv_squeeze(torch.cat([x1, x2, bp], 1))))
        
        # v2,直接调用 LocalGlobalAttention（无需插值）
        x1, x2 = inputs
        W_x1 = self.W_x1(x1)
        W_x2 = self.W_x2(x2)
        bp = self.W(W_x1 + W_x2)

        # v2,直接调用 LocalGlobalAttention（无需插值）
        x1 = torch.cat([self.lgb1_local(W_x1), self.lgb1_global(W_x1)], dim=1)
        x2 = torch.cat([self.lgb2_local(W_x2), self.lgb2_global(W_x2)], dim=1)

        return self.conv_final(self.rep_conv(self.conv_squeeze(torch.cat([x1, x2, bp], 1))))

######################################## Hierarchical Attention Fusion Block end ########################################
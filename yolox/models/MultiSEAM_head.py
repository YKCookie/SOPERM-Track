# import math, copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def make_anchors(feats, strides, grid_cell_offset=0.5):
#     """Generate anchors from features."""
#     anchor_points, stride_tensor = [], []
#     assert feats is not None
#     dtype, device = feats[0].dtype, feats[0].device
#     for i, stride in enumerate(strides):
#         _, _, h, w = feats[i].shape
#         sx = torch.arange(end=w, device=device) + grid_cell_offset  # shift x
#         sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y

#         if "indexing" in torch.meshgrid.__code__.co_varnames:
#             sy, sx = torch.meshgrid(sy, sx, indexing="ij")
#         else:
#             sy, sx = torch.meshgrid(sy, sx)

#         anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
#         stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
#     return torch.cat(anchor_points), torch.cat(stride_tensor)

# def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
#     """Transform distance(ltrb) to box(xywh or xyxy)."""
#     lt, rb = distance.chunk(2, dim)
#     x1y1 = anchor_points - lt
#     x2y2 = anchor_points + rb
#     if xywh:
#         c_xy = (x1y1 + x2y2) / 2
#         wh = x2y2 - x1y1
#         return torch.cat((c_xy, wh), dim)  # xywh bbox
#     return torch.cat((x1y1, x2y2), dim)  # xyxy bbox
# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p

# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """Perform transposed convolution of 2D data."""
#         return self.act(self.conv(x))

# ####################################### SEAM start ########################################

# class Residual(nn.Module):
#     def __init__(self, fn):
#         super(Residual, self).__init__()
#         self.fn = fn

#     def forward(self, x):
#         return self.fn(x) + x

# class SEAM(nn.Module):
#     def __init__(self, c1, c2, n, reduction=16):
#         super(SEAM, self).__init__()
#         if c1 != c2:
#             c2 = c1
#         self.DCovN = nn.Sequential(
#             *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
#                     nn.GELU(),
#                     nn.BatchNorm2d(c2)
#                 )),
#                 nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(c2)
#             ) for i in range(n)]
#         )
#         self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(c2, c2 // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c2 // reduction, c2, bias=False),
#             nn.Sigmoid()
#         )

#         self._initialize_weights()
#         # self.initialize_layer(self.avg_pool)
#         self.initialize_layer(self.fc)


#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.DCovN(x)
#         y = self.avg_pool(y).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         y = torch.exp(y)
#         return x * y.expand_as(x)

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight, gain=1)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def initialize_layer(self, layer):
#         if isinstance(layer, (nn.Conv2d, nn.Linear)):
#             torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
#             if layer.bias is not None:
#                 torch.nn.init.constant_(layer.bias, 0)

# def DcovN(c1, c2, depth, kernel_size=3, patch_size=3):
#     dcovn = nn.Sequential(
#         nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
#         nn.SiLU(),
#         nn.BatchNorm2d(c2),
#         *[nn.Sequential(
#             Residual(nn.Sequential(
#                 nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2),
#                 nn.SiLU(),
#                 nn.BatchNorm2d(c2)
#             )),
#             nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
#             nn.SiLU(),
#             nn.BatchNorm2d(c2)
#         ) for i in range(depth)]
#     )
#     return dcovn

# class MultiSEAM(nn.Module):
#     def __init__(self, c1, c2, depth, kernel_size=3, patch_size=[3, 5, 7], reduction=16):
#         super(MultiSEAM, self).__init__()
#         if c1 != c2:
#             c2 = c1
#         self.DCovN0 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[0])
#         self.DCovN1 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[1])
#         self.DCovN2 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[2])
#         self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(c2, c2 // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c2 // reduction, c2, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y0 = self.DCovN0(x)
#         y1 = self.DCovN1(x)
#         y2 = self.DCovN2(x)
#         y0 = self.avg_pool(y0).view(b, c)
#         y1 = self.avg_pool(y1).view(b, c)
#         y2 = self.avg_pool(y2).view(b, c)
#         y4 = self.avg_pool(x).view(b, c)
#         y = (y0 + y1 + y2 + y4) / 4
#         y = self.fc(y).view(b, c, 1, 1)
#         y = torch.exp(y)
#         return x * y.expand_as(x)

# class DFL(nn.Module):
#     """
#     Integral module of Distribution Focal Loss (DFL).

#     Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
#     """

#     def __init__(self, c1=16):
#         """Initialize a convolutional layer with a given number of input channels."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
#         x = torch.arange(c1, dtype=torch.float)
#         self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
#         self.c1 = c1

#     def forward(self, x):
#         """Applies a transformer layer on input tensor 'x' and returns a tensor."""
#         b, _, a = x.shape  # batch, channels, anchors
#         return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
#         # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)   

# ####################################### SEAM end ########################################

# class MultiSEAM(nn.Module):
#     def __init__(self, c1, c2, depth, kernel_size=3, patch_size=[3, 5, 7], reduction=16):
#         super(MultiSEAM, self).__init__()
#         if c1 != c2:
#             c2 = c1
#         self.DCovN0 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[0])
#         self.DCovN1 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[1])
#         self.DCovN2 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[2])
#         self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(c2, c2 // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c2 // reduction, c2, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y0 = self.DCovN0(x)
#         y1 = self.DCovN1(x)
#         y2 = self.DCovN2(x)
#         y0 = self.avg_pool(y0).view(b, c)
#         y1 = self.avg_pool(y1).view(b, c)
#         y2 = self.avg_pool(y2).view(b, c)
#         y4 = self.avg_pool(x).view(b, c)
#         y = (y0 + y1 + y2 + y4) / 4
#         y = self.fc(y).view(b, c, 1, 1)
#         y = torch.exp(y)
#         return x * y.expand_as(x)

# class Detect_SEAM(nn.Module):
#     """YOLOv8 Detect head for detection models."""
#     dynamic = False  # force grid reconstruction
#     export = False  # export mode
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init

#     def __init__(self, nc=80, ch=()):
#         """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
#         self.no = nc + self.reg_max * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build
#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
#         self.cv2 = nn.ModuleList(
#             nn.Sequential(Conv(x, c2, 3), SEAM(c2, c2, 1, 16), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
#         self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), SEAM(c3, c3, 1, 16), nn.Conv2d(c3, self.nc, 1)) for x in ch)
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

#     def forward(self, x):
#         """Concatenates and returns predicted bounding boxes and class probabilities."""
#         shape = x[0].shape  # BCHW
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         if self.training:
#             return x
#         elif self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape

#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
#         if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
#             box = x_cat[:, :self.reg_max * 4]
#             cls = x_cat[:, self.reg_max * 4:]
#         else:
#             box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
#         dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

#         if self.export and self.format in ('tflite', 'edgetpu'):
#             # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
#             # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
#             # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
#             img_h = shape[2] * self.stride[0]
#             img_w = shape[3] * self.stride[0]
#             img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
#             dbox /= img_size

#         y = torch.cat((dbox, cls.sigmoid()), 1)
#         return y if self.export else (y, x)

#     def bias_init(self):
#         """Initialize Detect() biases, WARNING: requires stride availability."""
#         m = self  # self.model[-1]  # Detect() module
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
#         # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
#         for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
#             a[-1].bias.data[:] = 1.0  # box
#             b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

# class Detect_MultiSEAM(Detect_SEAM):
#     def __init__(self, nc=80, ch=()):
#         super().__init__(nc, ch)
#         self.nc = nc  # number of classes
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
#         self.no = nc + self.reg_max * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build
#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
#         self.cv2 = nn.ModuleList(
#             nn.Sequential(Conv(x, c2, 3), MultiSEAM(c2, c2, 1), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
#         self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), MultiSEAM(c3, c3, 1), nn.Conv2d(c3, self.nc, 1)) for x in ch)
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y

        if "indexing" in torch.meshgrid.__code__.co_varnames:
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        else:
            sy, sx = torch.meshgrid(sy, sx)

        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with activation."""

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

####################################### SEAM start ########################################

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class SEAM(nn.Module):
    def __init__(self, c1, c2, n, reduction=16):
        super(SEAM, self).__init__()
        if c1 != c2:
            c2 = c1
        self.DCovN = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                )),
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            ) for _ in range(n)]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

######################################## SEAM end ########################################

def DcovN(c1, c2, depth, kernel_size=3, patch_size=3):
    """DcovN module."""
    dcovn = nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
        nn.SiLU(),
        nn.BatchNorm2d(c2),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2),
                nn.SiLU(),
                nn.BatchNorm2d(c2)
            )),
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
            nn.SiLU(),
            nn.BatchNorm2d(c2)
        ) for _ in range(depth)]
    )
    return dcovn

class MultiSEAM(nn.Module):
    def __init__(self, c1, c2, depth, kernel_size=3, patch_size=[3, 5, 7], reduction=16):
        super(MultiSEAM, self).__init__()
        if c1 != c2:
            c2 = c1
        self.DCovN0 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[0])
        self.DCovN1 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[1])
        self.DCovN2 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y0 = self.DCovN0(x)
        y1 = self.DCovN1(x)
        y2 = self.DCovN2(x)
        y0 = self.avg_pool(y0).view(b, c)
        y1 = self.avg_pool(y1).view(b, c)
        y2 = self.avg_pool(y2).view(b, c)
        y4 = self.avg_pool(x).view(b, c)
        y = (y0 + y1 + y2 + y4) / 4
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class Detect_SEAM(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), SEAM(c2, c2, 1, 16), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), SEAM(c3, c3, 1, 16), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        outputs = []

        for i in range(self.nl):
            output_cv2 = self.cv2[i](x[i])
            output_cv3 = self.cv3[i](x[i])
            output = torch.cat((output_cv2, output_cv3), 1)
            outputs.append(output)

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in outputs], 2)

        # Compute box and class probabilities
        box = x_cat[:, :self.reg_max * 4]
        cls = x_cat[:, self.reg_max * 4:]
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)  # Correct output format


class Detect_MultiSEAM(Detect_SEAM):
    """Custom MultiSEAM detection head."""
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), MultiSEAM(c2, c2, 1), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), MultiSEAM(c3, c3, 1), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

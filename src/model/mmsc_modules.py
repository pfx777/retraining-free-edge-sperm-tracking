#!/usr/bin/env python3
"""Custom lightweight modules for YOLOv11n-MMSC-style fusion experiments.

These are clean-room approximations intended for an ablation study. They are not
the official implementations from the cited papers, which are not public.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import DFLoss, bbox2dist, bbox_iou


def _space_to_depth(x, scale):
    """Pixel-unshuffle based space-to-depth used by SPDConv."""
    return F.pixel_unshuffle(x, scale)


class DropPath(nn.Module):
    """Stochastic depth from timm, kept self-contained."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


class SPDConv(nn.Module):
    """Space-to-depth convolution. Keeps channel count unchanged in this config."""

    def __init__(self, c1, k=3, s=2):
        super().__init__()
        self.s = s
        self.cv = Conv(c1 * s * s, c1, k, 1)

    def forward(self, x):
        return self.cv(_space_to_depth(x, self.s))


class SPDLite(nn.Module):
    """Cost-controlled space-to-depth downsample: the lossless-detail tap.

    Motivation is measured, not assumed.  On this task the deployed detector
    finds the objects (recall at IoU 0.3 is 0.84-0.99) but cannot resolve how big
    they are: the matched det/GT side ratio has a per-video median of 0.737 /
    0.831 / 1.140 / 1.115, and the recall gap between IoU 0.3 and IoU 0.5 tracks
    |log ratio| monotonically.  The targets sit at 13-20 px in a 448 input, i.e.
    1.7-2.5 cells at stride 8, and stock YOLO11n's finest head input is stride 8:
    the stride-4 backbone feature is computed and then only ever seen through a
    single strided convolution, which keeps 1 of every 4 pixels.

    SPDConv fixes that but costs a 3x3 conv on the unshuffled 4*c1 channels --
    462 MMACs at c1=64/448, i.e. 16% of the deployed model, which is why the v8
    stack could not afford it.  Here the unshuffle is followed by a 1x1 (mix the
    four phases) and a depthwise 3x3 (give them a spatial window): 53 MMACs, 8.7x
    cheaper, and the operator set (reshape / 1x1 / depthwise) is what the NPU
    runs natively.

    Channel count in equals channel count out, which is what ``parse_model``
    assumes for a block it does not know.

    ``mode`` selects how the 2x2 blocks are reduced:

    ``"spd"``  space-to-depth via ``F.pixel_unshuffle``.  Phase-preserving, but
               ONNX lowers it to Reshape/Transpose, and on the RK3568 those are
               CPU ops: measured against the identical model without the tap,
               this form adds 1 Transpose (8.7 ms) and 2 Reshape (1.7 ms each)
               per frame, ~21% of the 57 ms detection budget.
    ``"slice"`` the same rearrangement written as four strided slices plus a
               Concat, which is bit-identical to ``"spd"`` but lowers to
               Slice/Concat -- neither is flagged by audit_onnx_ops.py.  Use this
               one on the board.  Because the two forms share every parameter,
               an ``"spd"`` checkpoint can be switched to ``"slice"`` after
               loading, with no retraining and no change in accuracy.
    ``"pool"`` a 3x3/stride-2 average pool first.  It is a low-pass instead of an
               exact rearrangement, so it does smear the finest phase a little,
               but it lowers to AveragePool, which every toolkit version runs on
               the NPU.

    Both stay at c1 channels in and out, which is what ``parse_model`` assumes.
    """

    def __init__(self, c1, s=2, mode="spd"):
        super().__init__()
        self.s = s
        self.mode = mode
        if mode in ("spd", "slice"):
            self.cv1 = Conv(c1 * s * s, c1, 1, 1)
        elif mode == "pool":
            self.down = nn.AvgPool2d(s, s)
            self.cv1 = Conv(c1, c1, 1, 1)
        else:
            raise ValueError(f"SPDLite mode must be 'spd', 'slice' or 'pool', got {mode!r}")
        self.cv2 = Conv(c1, c1, 3, 1, g=c1)

    def forward(self, x):
        if self.mode == "spd":
            y = _space_to_depth(x, self.s)
        elif self.mode == "slice":
            # Layout-identical to F.pixel_unshuffle: the phase index must be the
            # MINOR part of the output channel index (c*s*s + i*s + j).  Writing
            # it phase-major instead is a permutation of the following 1x1
            # conv's input channels, i.e. the same weights would mean something
            # else -- verified numerically before this was committed.
            n, c, h, w = x.shape
            y = torch.cat(
                [
                    x[..., i :: self.s, j :: self.s].unsqueeze(2)
                    for i in range(self.s)
                    for j in range(self.s)
                ],
                dim=2,
            )
            y = y.reshape(n, c * self.s * self.s, h // self.s, w // self.s)
        else:
            y = self.down(x)
        return self.cv2(self.cv1(y))


class CMUNeXtBlock(nn.Module):
    """Large-kernel depthwise bottleneck in the spirit of CMUNeXt."""

    def __init__(self, c1, k=7, e=0.5):
        super().__init__()
        c_ = max(8, int(c1 * e))
        self.cv1 = Conv(c1, c_, 1, 1)
        self.dw = Conv(c_, c_, k, 1, g=c_)
        self.cv2 = Conv(c_, c1, 1, 1, act=False)

    def forward(self, x):
        return x + self.cv2(self.dw(self.cv1(x)))


class StarBlock(nn.Module):
    """Element-wise multiplication of two convolution branches."""

    def __init__(self, c1, k=3, e=0.5):
        super().__init__()
        c_ = max(8, int(c1 * e))
        self.branch1 = Conv(c1, c_, k, 1)
        self.branch2 = Conv(c1, c_, k, 1)
        self.cv = Conv(c_, c1, 1, 1, act=False)

    def forward(self, x):
        return x + self.cv(self.branch1(x) * self.branch2(x))


class RCA(nn.Module):
    """Rectangular self-calibration attention from CGRSeg (ECCV 2024)."""

    def __init__(
        self,
        inp,
        kernel_size=1,
        ratio=1,
        band_kernel_size=11,
        dw_size=(1, 1),
        padding=(0, 0),
        stride=1,
        square_kernel_size=2,
        relu=True,
    ):
        super().__init__()
        self.dwconv_hw = nn.Conv2d(
            inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(
                inp,
                gc,
                kernel_size=(1, band_kernel_size),
                padding=(0, band_kernel_size // 2),
                groups=gc,
            ),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                gc,
                inp,
                kernel_size=(band_kernel_size, 1),
                padding=(band_kernel_size // 2, 0),
                groups=gc,
            ),
            nn.Sigmoid(),
        )

    def sge(self, x):
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w
        return self.excite(x_gather)

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        return att * loc


class RCM(nn.Module):
    """Rectangular self-calibration module from CGRSeg (ECCV 2024)."""

    def __init__(
        self,
        dim,
        token_mixer=RCA,
        norm_layer=nn.BatchNorm2d,
        mlp_layer=None,
        mlp_ratio=2,
        act_layer=nn.GELU,
        ls_init_value=1e-6,
        drop_path=0.0,
        dw_size=11,
        square_kernel_size=3,
        ratio=1,
    ):
        super().__init__()
        if mlp_layer is None:
            mlp_layer = ConvMlp
        self.token_mixer = token_mixer(
            dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size, ratio=ratio
        )
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return self.drop_path(x) + shortcut


class MSCAM(nn.Module):
    """Multi-scale channel attention block."""

    def __init__(self, c1, r=4):
        super().__init__()
        c_ = max(8, c1 // r)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = Conv(c1, c_, 1, 1, act=False)
        self.fc2 = Conv(c_, c1, 1, 1, act=False)
        self.spatial = Conv(c1, c1, 3, 1, g=c1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.fc1(self.avg_pool(x)))
        mx = self.fc2(self.fc1(self.max_pool(x)))
        return x * self.act(avg + mx) + self.spatial(x)


class GatedCtx(nn.Module):
    """NPU-native stand-in for the PSABlock inside C2PSA.

    PSABlock's spatial self-attention lowers to MatMul + Softmax + Transpose,
    which rknn-toolkit2 1.6.0 cannot schedule on the NPU.  The RK3568 board
    profile (runs/deploy/rk3568_deployment_results.md) attributes 50.7 ms to
    2x MatMul, 30.7 ms to 2x Softmax and 26.1 ms to 3x Transpose -- 19 CPU
    operator calls for 127.7 ms, 43% of the frame, against 125.3 ms for the
    entire 76-conv NPU stack.

    Dropping C2PSA outright is not free: the ablation ladder measured
    v8full 0.562 -> no_c2psa 0.444 -> no_c2psa+no_ema 0.469 dev MOTA, with the
    loss spread across all three dev videos and across both FP and FN, so the
    block is doing real work at the deepest scale.

    This block keeps that function -- modulate the feature map by a
    global-context gate -- using only ops that stay on the NPU: global average
    pooling, 1x1 convolutions, SiLU, Sigmoid and a depthwise 3x3 for the
    spatial term.  Channel count is preserved, so it drops straight into the
    slot C2PSA occupied.
    """

    def __init__(self, c1, k=3, r=8):
        super().__init__()
        c_ = max(8, c1 // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv(c1, c_, 1, 1)
        self.fc2 = Conv(c_, c1, 1, 1, act=False)
        self.spatial = Conv(c1, c1, k, 1, g=c1, act=False)
        self.act = nn.Sigmoid()
        self.out = Conv(c1, c1, 1, 1, act=False)

    def forward(self, x):
        ctx = self.fc2(self.fc1(self.pool(x)))
        return x + self.out(x * self.act(ctx + self.spatial(x)))


class MDFM(nn.Module):
    """Multi-scale difference fusion block."""

    def __init__(self, c1):
        super().__init__()
        c_ = max(8, c1 // 2)
        self.local = Conv(c1, c_, 3, 1)
        self.global_pool = nn.AvgPool2d(5, 1, 2)
        self.global_conv = Conv(c1, c_, 1, 1)
        self.fuse = Conv(2 * c_, c1, 1, 1, act=False)

    def forward(self, x):
        local = self.local(x)
        global_feat = self.global_conv(self.global_pool(x))
        return x + self.fuse(torch.cat([local, torch.abs(local - global_feat)], dim=1))


class NWDBboxLoss(nn.Module):
    """Compatibility class used when loading checkpoints trained with NWD loss."""

    def __init__(self, reg_max=16, c=12.8, alpha=0.5):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.c = c
        self.alpha = alpha

    @staticmethod
    def _nwd(pred, target, c):
        pcx = (pred[:, 0] + pred[:, 2]) / 2
        pcy = (pred[:, 1] + pred[:, 3]) / 2
        pw = (pred[:, 2] - pred[:, 0]).clamp_min(1e-6)
        ph = (pred[:, 3] - pred[:, 1]).clamp_min(1e-6)
        tcx = (target[:, 0] + target[:, 2]) / 2
        tcy = (target[:, 1] + target[:, 3]) / 2
        tw = (target[:, 2] - target[:, 0]).clamp_min(1e-6)
        th = (target[:, 3] - target[:, 1]).clamp_min(1e-6)
        dist2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        dist2 = dist2 + ((pw - tw) / 2) ** 2 + ((ph - th) / 2) ** 2
        return torch.exp(-torch.sqrt(dist2 + 1e-6) / c)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, *args, **kwargs):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred = pred_bboxes[fg_mask]
        target = target_bboxes[fg_mask]
        ciou = bbox_iou(pred, target, xywh=False, CIoU=True)
        nwd = self._nwd(pred, target, self.c).unsqueeze(-1)
        combined = self.alpha * (1.0 - ciou) + (1.0 - self.alpha) * (1.0 - nwd)
        loss_iou = (combined * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl


class PixelNWDBboxLoss(nn.Module):
    """Pixel-space CIoU + NWD hybrid box loss (stride-rescaled for tiny boxes)."""

    def __init__(self, reg_max=16, c=12.8, alpha=0.5):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.c = c
        self.alpha = alpha

    @staticmethod
    def _nwd(pred_xyxy, target_xyxy, c):
        pcx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
        pcy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
        pw = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp_min(1.0)
        ph = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp_min(1.0)
        tcx = (target_xyxy[:, 0] + target_xyxy[:, 2]) / 2
        tcy = (target_xyxy[:, 1] + target_xyxy[:, 3]) / 2
        tw = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp_min(1.0)
        th = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp_min(1.0)
        dist2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        dist2 = dist2 + ((pw - tw) / 2) ** 2 + ((ph - th) / 2) ** 2
        return torch.exp(-torch.sqrt(dist2 + 1e-6) / c)

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred = pred_bboxes[fg_mask]
        target = target_bboxes[fg_mask]
        stride_fg = stride.view(1, -1, 1).expand_as(pred_bboxes)[fg_mask]
        pred_px = pred * stride_fg
        target_px = target * stride_fg
        ciou = bbox_iou(pred, target, xywh=False, CIoU=True)
        nwd = self._nwd(pred_px, target_px, self.c).unsqueeze(-1)
        combined = self.alpha * (1.0 - ciou) + (1.0 - self.alpha) * (1.0 - nwd)
        loss_iou = (combined * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl


class InnerMPDIoUNWDBboxLoss(nn.Module):
    """Small-object box loss: Inner-MPDIoU + NWD (SpermYOLOv8-E recipe).

    The Inner-MPDIoU term replaces the usual CIoU term, while NWD replaces DFL.
    ``ratio`` controls the auxiliary inner-box scale and ``c`` is the NWD
    normalization constant. Inputs are expected in normalized xyxy coordinates,
    matching Ultralytics' ``v8DetectionLoss`` call site.
    """

    def __init__(self, reg_max=16, ratio=0.7, c=12.8):
        super().__init__()
        self.reg_max = reg_max
        self.ratio = ratio
        self.c = c

    @staticmethod
    def _inner(box, ratio):
        cx = (box[:, 0] + box[:, 2]) / 2
        cy = (box[:, 1] + box[:, 3]) / 2
        w = (box[:, 2] - box[:, 0]).clamp_min(1e-6)
        h = (box[:, 3] - box[:, 1]).clamp_min(1e-6)
        return torch.stack(
            [
                cx - w * ratio / 2,
                cy - h * ratio / 2,
                cx + w * ratio / 2,
                cy + h * ratio / 2,
            ],
            dim=1,
        )

    @classmethod
    def _inner_mpdiou(cls, pred, target, ratio, eps=1e-7):
        p = cls._inner(pred, ratio)
        t = cls._inner(target, ratio)
        inter = (
            (torch.minimum(p[:, 2], t[:, 2]) - torch.maximum(p[:, 0], t[:, 0])).clamp_min(0)
            * (torch.minimum(p[:, 3], t[:, 3]) - torch.maximum(p[:, 1], t[:, 1])).clamp_min(0)
        )
        p_area = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
        t_area = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
        union = p_area + t_area - inter + eps
        iou = inter / union

        cw = torch.maximum(pred[:, 2], target[:, 2]) - torch.minimum(pred[:, 0], target[:, 0])
        ch = torch.maximum(pred[:, 3], target[:, 3]) - torch.minimum(pred[:, 1], target[:, 1])
        denom = (cw.pow(2) + ch.pow(2)).clamp_min(eps)
        d1 = (p[:, 0] - t[:, 0]).pow(2) + (p[:, 1] - t[:, 1]).pow(2)
        d2 = (p[:, 2] - t[:, 2]).pow(2) + (p[:, 3] - t[:, 3]).pow(2)
        return iou - d1 / denom - d2 / denom

    @staticmethod
    def _nwd(pred, target, c):
        pcx = (pred[:, 0] + pred[:, 2]) / 2
        pcy = (pred[:, 1] + pred[:, 3]) / 2
        pw = (pred[:, 2] - pred[:, 0]).clamp_min(1e-6)
        ph = (pred[:, 3] - pred[:, 1]).clamp_min(1e-6)
        tcx = (target[:, 0] + target[:, 2]) / 2
        tcy = (target[:, 1] + target[:, 3]) / 2
        tw = (target[:, 2] - target[:, 0]).clamp_min(1e-6)
        th = (target[:, 3] - target[:, 1]).clamp_min(1e-6)
        dist2 = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)
        dist2 = dist2 + ((pw - tw) / 2).pow(2) + ((ph - th) / 2).pow(2)
        return torch.exp(-torch.sqrt(dist2 + 1e-6) / c)

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        *args,
        **kwargs,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred = pred_bboxes[fg_mask]
        target = target_bboxes[fg_mask]

        mpdiou = self._inner_mpdiou(pred, target, self.ratio)
        loss_iou = ((1.0 - mpdiou).unsqueeze(-1) * weight).sum() / target_scores_sum

        nwd = self._nwd(pred, target, self.c).unsqueeze(-1)
        loss_dfl = ((1.0 - nwd) * weight).sum() / target_scores_sum
        return loss_iou, loss_dfl


class WIoUv1BboxLoss(nn.Module):
    """Distance-attention IoU loss, simplified from the WIoU family."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        *args,
        **kwargs,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred = pred_bboxes[fg_mask]
        target = target_bboxes[fg_mask]

        iou = bbox_iou(pred, target, xywh=False, CIoU=False)
        pcx = (pred[:, 0] + pred[:, 2]) / 2
        pcy = (pred[:, 1] + pred[:, 3]) / 2
        tcx = (target[:, 0] + target[:, 2]) / 2
        tcy = (target[:, 1] + target[:, 3]) / 2
        cw = torch.maximum(pred[:, 2], target[:, 2]) - torch.minimum(pred[:, 0], target[:, 0])
        ch = torch.maximum(pred[:, 3], target[:, 3]) - torch.minimum(pred[:, 1], target[:, 1])
        dist2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        r = torch.exp(dist2 / (cw.pow(2) + ch.pow(2) + 1e-6)).unsqueeze(1)

        loss_iou = (r * (1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class WIoUv3BboxLoss(nn.Module):
    """WIoU v3 with dynamic non-monotonic gradient gain."""

    def __init__(self, reg_max=16, alpha=1.9, delta=3.0):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.alpha = alpha
        self.delta = delta

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        *args,
        **kwargs,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred = pred_bboxes[fg_mask]
        target = target_bboxes[fg_mask]

        iou = bbox_iou(pred, target, xywh=False, CIoU=False)
        pcx = (pred[:, 0] + pred[:, 2]) / 2
        pcy = (pred[:, 1] + pred[:, 3]) / 2
        tcx = (target[:, 0] + target[:, 2]) / 2
        tcy = (target[:, 1] + target[:, 3]) / 2
        cw = torch.maximum(pred[:, 2], target[:, 2]) - torch.minimum(pred[:, 0], target[:, 0])
        ch = torch.maximum(pred[:, 3], target[:, 3]) - torch.minimum(pred[:, 1], target[:, 1])
        dist2 = ((pcx - tcx) ** 2 + (pcy - tcy) ** 2).unsqueeze(1)
        r_w = torch.exp(dist2 / ((cw.unsqueeze(1)) ** 2 + (ch.unsqueeze(1)) ** 2 + 1e-6))

        loss_iou_raw = 1.0 - iou
        with torch.no_grad():
            beta = loss_iou_raw / (loss_iou_raw.mean() + 1e-6)
            r = beta / (self.delta * torch.pow(self.alpha, beta - self.delta))

        loss_iou = (r * r_w * loss_iou_raw * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class MSCAMV2(nn.Module):
    """Stronger multi-scale channel attention with 1x1/3x3/5x5 branches."""

    def __init__(self, c1, r=4):
        super().__init__()
        c_ = max(8, c1 // r)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = Conv(c1, c_, 1, 1, act=False)
        self.fc2 = Conv(c_, c1, 1, 1, act=False)

        self.branch1 = Conv(c1, c_, 1, 1)
        self.branch3 = Conv(c1, c_, 3, 1, g=1)
        self.branch5 = Conv(c1, c_, 5, 1, g=1)
        self.fuse = Conv(3 * c_, c1, 1, 1, act=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.fc1(self.avg_pool(x)))
        mx = self.fc2(self.fc1(self.max_pool(x)))
        att = self.act(avg + mx)

        y1 = self.branch1(x)
        y3 = self.branch3(x)
        y5 = self.branch5(x)
        multi = self.fuse(torch.cat([y1, y3, y5], dim=1))
        return x * att + multi


class SegNextAttention(nn.Module):
    """Lightweight SegNeXt-style multi-scale convolutional attention.

    This is closer to the SpermYOLOv8-E MSCA block than the earlier MSCAM/MSCAMV2
    approximations: a 5x5 depthwise local aggregation, a dilated 7x7 depthwise
    context aggregation, and a 1x1 channel mixer. It preserves channel count.
    """

    def __init__(self, c1):
        super().__init__()
        self.dw1 = Conv(c1, c1, 5, 1, g=c1)
        self.dw2 = Conv(c1, c1, 7, 1, g=c1, d=3)
        self.pw = Conv(c1, c1, 1, 1, act=False)

    def forward(self, x):
        att = self.dw2(self.dw1(x))
        return x * self.pw(att)


class EMA(nn.Module):
    """Efficient Multi-Scale Attention for small-object feature enhancement."""

    def __init__(self, channels, factor=32):
        super().__init__()
        self.groups = factor
        assert channels % self.groups == 0
        group_channels = channels // self.groups

        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = Conv(group_channels, group_channels, 1, 1, act=False)
        self.conv3x3 = Conv(group_channels, group_channels, 3, 1, g=group_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        group_x = x.reshape(b * self.groups, -1, h, w)

        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        gc = c // self.groups
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, gc, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, gc, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class C3k2_EMA(nn.Module):
    """Channel-preserving C3k2 variant with EMA attention blocks."""

    def __init__(self, c1, n=1, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        assert c_ % 32 == 0
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.m = nn.ModuleList(EMA(c_) for _ in range(n))
        self.cv2 = Conv((2 + n) * c_, c1, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SE(nn.Module):
    """Squeeze-and-Excitation channel attention used by DP-YOLOv8n.

    The channel count is inferred lazily from the first forward pass so the
    module can be placed in a YAML with `[-1, 1, SE, []]` without hard-coding
    the width-scaled channel count.
    """

    def __init__(self, r=16):
        super().__init__()
        self.r = r
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = None

    def _build(self, c, device, dtype):
        c_ = max(8, c // self.r)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c_, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c, 1, bias=False),
            nn.Sigmoid(),
        )
        self.fc.to(device=device, dtype=dtype)

    def forward(self, x):
        if self.fc is None:
            self._build(x.shape[1], x.device, x.dtype)
        return x * self.fc(self.pool(x))


class GSConv(nn.Module):
    """Channel-preserving GSConv for the DP-YOLOv8n-style Slim-Neck.

    This is the GSConv grouped-shuffle block from ``slim-neck-by-gsconv``,
    adapted to be YAML-friendly by inferring ``c1`` lazily. It keeps the
    channel count unchanged, so it can be inserted directly after the
    channel-mixing C3k2 blocks in the neck without changing downstream layers.
    """

    def __init__(self, k=3, s=1):
        super().__init__()
        self.k = k
        self.s = s
        self.cv1 = None
        self.cv2 = None

    def _build(self, c, device, dtype):
        assert c % 2 == 0, f"GSConv requires an even channel count, got {c}"
        c_ = c // 2
        self.cv1 = Conv(c, c_, self.k, self.s)
        self.cv2 = Conv(c_, c_, 5, 1, g=c_)
        self.cv1.to(device=device, dtype=dtype)
        self.cv2.to(device=device, dtype=dtype)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        return x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous().view(b, c, h, w)

    def forward(self, x):
        if self.cv1 is None:
            self._build(x.shape[1], x.device, x.dtype)
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        return self.channel_shuffle(torch.cat((x1, x2), 1), 2)


class GSBottleneck(nn.Module):
    """GS bottleneck used inside the VoVGSCSP Slim-Neck block."""

    def __init__(self, c, e=0.5):
        super().__init__()
        c_ = max(8, int(c * e))
        self.cv1 = Conv(c, c_, 1, 1)
        self.gs = GSConv(3, 1)
        self.cv2 = Conv(c_, c, 1, 1, act=False)

    def forward(self, x):
        return x + self.cv2(self.gs(self.cv1(x)))


class VoVGSCSP(nn.Module):
    """Channel-preserving VoV-GSCSP for the DP-YOLOv8n Slim-Neck.

    The input channel count is inferred lazily from the first forward pass so
    the block can be used in a YAML as ``[-1, 1, VoVGSCSP, []]`` without
    hard-coding the width-scaled channel count.
    """

    def __init__(self, n=3, e=0.5):
        super().__init__()
        self.n = n
        self.e = e
        self.cv1 = None
        self.cv2 = None
        self.m = None
        self.cv3 = None

    def _build(self, c, device, dtype):
        c_ = max(8, int(c * self.e))
        assert c_ % 2 == 0, f"VoVGSCSP requires an even hidden channel count, got {c_}"
        self.cv1 = Conv(c, c_, 1, 1)
        self.cv2 = Conv(c, c_, 1, 1)
        self.m = nn.Sequential(*(GSBottleneck(c_, e=1.0) for _ in range(self.n)))
        self.cv3 = Conv(2 * c_, c, 1, 1, act=False)
        self.to(device=device, dtype=dtype)

    def forward(self, x):
        if self.cv1 is None:
            self._build(x.shape[1], x.device, x.dtype)
        y = torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)
        return self.cv3(y)


class SAFM(nn.Module):
    """Spatial Adaptive Feature Modulation."""

    def __init__(self, channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        att = self.att(torch.cat([avg, mx], dim=1))
        return x + x * att


class SubPixelUpsample(nn.Module):
    """Learned sub-pixel (pixel-shuffle) upsampling, replacing nearest-neighbor."""

    def __init__(self, c, scale=2):
        super().__init__()
        self.conv = Conv(c, c * scale * scale, 1, 1, act=False)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.shuffle(self.conv(x))


class CARAFE(nn.Module):
    """Content-aware reassembly of features (CARAFE) upsampling."""

    def __init__(self, c, scale=2, k_up=3, group=16):
        super().__init__()
        self.scale = scale
        self.k_up = k_up
        self.encoder = nn.Sequential(
            nn.Conv2d(c, group, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(group, k_up * k_up * scale * scale, 1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        kernels = F.softmax(self.encoder(x), dim=1)  # (b, k_up^2, sH, sW)
        x_un = F.unfold(x, kernel_size=self.k_up, padding=self.k_up // 2)  # (b, c*k_up^2, h*w)
        x_un = x_un.view(b, c, self.k_up * self.k_up, h, w)
        x_un = x_un.repeat_interleave(self.scale, dim=3).repeat_interleave(self.scale, dim=4)
        return torch.einsum("bkhw,bckhw->bchw", kernels, x_un)


class DySample(nn.Module):
    """Ultra-lightweight dynamic upsampler from "Learning to Upsample by Learning to Sample".

    The first positional argument is the input channel count, matching the
    convention used by the other YOLO-custom modules in this file. ``lp``
    (linear + pixel-shuffle) is used by default because it is the simplest
    variant that already outperforms nearest-neighbour upsampling for small
    objects.
    """

    def __init__(self, c1, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in {"lp", "pl"}
        assert c1 % groups == 0
        if style == "pl":
            assert c1 % scale**2 == 0

        in_channels = c1 if style == "lp" else c1 // scale**2
        out_channels = 2 * groups * scale**2
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)

        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h]))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def _sample(self, x, offset):
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h]))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
        )
        normalizer = torch.tensor([w, h], device=x.device, dtype=x.dtype).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(b, -1, h, w), self.scale).view(
            b, 2, -1, self.scale * h, self.scale * w
        ).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(b, -1, self.scale * h, self.scale * w)

    def forward(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        if self.style == "pl":
            x = F.pixel_shuffle(x, self.scale)
            offset = F.pixel_unshuffle(offset, self.scale)
        return self._sample(x, offset)


class InceptionDWConv2d(nn.Module):
    """Multi-scale depthwise token mixer from InceptionNeXt/MetaNeXt.

    Splits the channel dimension into an identity branch and three depthwise
    branches: square 3x3, horizontal 1xk, and vertical kx1. This is the exact
    "multi-branch depthwise separable convolution" used by DRM-YOLO's MetaDWBlock.
    """

    def __init__(self, in_chs, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = max(1, int(in_chs * branch_ratio))
        square_padding = square_kernel_size // 2
        band_padding = band_kernel_size // 2
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_padding, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, (1, band_kernel_size), padding=(0, band_padding), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, (band_kernel_size, 1), padding=(band_padding, 0), groups=gc)
        self.split_indexes = (in_chs - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


class ConvMlp(nn.Module):
    """1x1-conv MLP used inside MetaNeXtBlock, preserving spatial dimensions."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.norm = nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)

    def forward(self, x):
        return self.fc2(self.act(self.norm(self.fc1(x))))


class MetaDWBlock(nn.Module):
    """MetaDWBlock: InceptionDWConv2d token mixer + 1x1-conv MLP + layer scale."""

    def __init__(self, dim, mlp_ratio=4, ls_init_value=1.0):
        super().__init__()
        self.token_mixer = InceptionDWConv2d(dim)
        self.norm = nn.BatchNorm2d(dim)
        self.mlp = ConvMlp(dim, int(mlp_ratio * dim))
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return x + shortcut


class MetaDWStage(nn.Module):
    """Hierarchical MetaDWStage for YOLO backbones.

    DRM-YOLO places a strided Conv before each stage, so this stage keeps the
    spatial resolution and stacks ``n`` MetaDWBlock layers channel-to-channel.
    """

    def __init__(self, c1, s=1, n=3):
        super().__init__()
        self.blocks = nn.Sequential(*(MetaDWBlock(c1) for _ in range(n)))

    def forward(self, x):
        return self.blocks(x)


MetaNeXtStage = MetaDWStage


class FRFCFuse(nn.Module):
    """Scale-feature relationship cross-fusion (FRFDet SFRCF).

    This is a clean-room port of SFRCFMul/SFRCFAdd from FRFDet that works with
    vanilla Ultralytics YAML parsing.  Instead of taking a ``from`` list of two
    tensors, it follows a ``Concat`` layer, splits the concatenated tensor back
    into the two scales, and returns a tensor with the same number of channels.

    Args:
        c1: channels of the first scale (deep/upsampled feature).
        c2: channels of the second scale (shallow/skip feature).
        kernel_pair: depthwise kernel sizes for the two relationship branches.
        mode: ``"mul"`` for compact models, ``"add"`` for larger models.
    """

    def __init__(self, c1, c2, kernel_pair=(3, 5), mode="mul"):
        super().__init__()
        assert mode in {"mul", "add"}
        self.c1, self.c2 = c1, c2
        self.mode = mode
        self.pre = [c1 // 2, c2 // 2]
        self.post = [c1 // 2, c2 // 2]
        k0, k1 = kernel_pair

        self.group_conv1 = nn.ModuleList(
            [Conv(self.pre[0], self.pre[0], k0, 1, g=self.pre[0]),
             Conv(self.pre[1], self.pre[1], k0, 1, g=self.pre[1])]
        )
        self.group_conv2 = nn.ModuleList(
            [Conv(self.pre[0], self.pre[0], k1, 1, g=self.pre[0]),
             Conv(self.pre[1], self.pre[1], k1, 1, g=self.pre[1])]
        )
        self.point_conv1 = nn.ModuleList(
            [nn.Conv2d(self.pre[0], self.post[0], 1, bias=True),
             nn.Conv2d(self.pre[1], self.post[1], 1, bias=True)]
        )
        self.point_conv2 = nn.ModuleList(
            [nn.Conv2d(self.pre[0], self.post[1], 1, bias=True),
             nn.Conv2d(self.pre[1], self.post[0], 1, bias=True)]
        )
        self.fuse_conv = nn.ModuleList(
            [Conv(self.pre[0], c1, 1, 1, act=False),
             Conv(self.pre[1], c2, 1, 1, act=False)]
        )
        self.act = nn.SiLU(inplace=False)

    def forward(self, x):
        x1, x2 = x.split([self.c1, self.c2], dim=1)
        x1_0, x1_1 = torch.chunk(x1, 2, dim=1)
        x2_0, x2_1 = torch.chunk(x2, 2, dim=1)

        y1_0 = self.point_conv1[0](self.group_conv1[0](x1_0))
        y1_1 = self.point_conv2[0](self.group_conv2[0](x1_1))
        y2_0 = self.point_conv1[1](self.group_conv1[1](x2_0))
        y2_1 = self.point_conv2[1](self.group_conv2[1](x2_1))

        if self.mode == "mul":
            y1 = y1_0 * self.act(y2_1)
            y2 = y2_0 * self.act(y1_1)
        else:
            y1 = y1_0 + self.act(y2_1)
            y2 = y2_0 + self.act(y1_1)

        y1 = x1 + self.fuse_conv[0](y1)
        y2 = x2 + self.fuse_conv[1](y2)
        return torch.cat((y1, y2), dim=1)


class IBS_D(nn.Module):
    """Inverse bidirectional sampling, downsampling variant from FRFDet.

    Replaces a strided convolution while preserving the channel count.  The
    original ``einops`` rearrangement is equivalent to ``pixel_shuffle`` for
    stride two, which keeps this implementation dependency-free.
    """

    def __init__(self, c1, stride=2, k=3, hidden_ratio=2):
        super().__init__()
        assert c1 % (stride ** 2) == 0
        hidden = c1 * hidden_ratio
        self.stride = stride
        self.conv = nn.Sequential(
            Conv(c1, hidden, 1, 1),
            Conv(hidden, hidden, k, 1, g=hidden),
            Conv(hidden, c1 // (stride ** 2), 1, 1),
        )

    def forward(self, x):
        return F.pixel_unshuffle(self.conv(x), self.stride)


class IBS_U(nn.Module):
    """Inverse bidirectional sampling, upsampling variant from FRFDet.

    Replaces nearest-neighbour upsampling while preserving the channel count.
    Uses ``pixel_unshuffle`` instead of the original ``einops`` rearrange.
    """

    def __init__(self, c1, scale=2, k=3, hidden_ratio=2):
        super().__init__()
        assert c1 % (scale ** 2) == 0
        hidden = c1 * hidden_ratio
        self.scale = scale
        self.conv = nn.Sequential(
            Conv(c1 // (scale ** 2), hidden, 1, 1),
            Conv(hidden, hidden, k, 1, g=hidden),
            Conv(hidden, c1, 1, 1),
        )

    def forward(self, x):
        return self.conv(F.pixel_shuffle(x, self.scale))


class LCEB(nn.Module):
    """Local Context Enhancement Block."""

    def __init__(self, channels):
        super().__init__()
        self.local3 = Conv(channels, channels, 3, 1, g=channels)
        self.local5 = Conv(channels, channels, 5, 1, g=channels)
        self.fuse = Conv(2 * channels, channels, 1, 1, act=False)

    def forward(self, x):
        return x + self.fuse(torch.cat([self.local3(x), self.local5(x)], dim=1))


class SpermDetLCEB(nn.Module):
    """Faithful local-context enhancement block from SpermDet."""

    def __init__(self, c1, c2=None, k=(3, 5)):
        super().__init__()
        c2 = c1 if c2 is None else c2
        c_ = c1 // 2
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3_1 = Conv(c_, c_, 3, 1)
        self.cv3_2 = Conv(c_, c_, 3, 1)
        self.merge = Conv(2 * c_, c2, 3, 1)
        self.att = nn.Sequential(
            nn.Conv2d(c_, c_ // 2, 1, 1),
            nn.BatchNorm2d(c_ // 2),
            nn.SiLU(inplace=False),
            nn.Conv2d(c_ // 2, c_, 5, 1, 2),
            nn.Sigmoid(),
        )
        self.affine = Conv(c_, c_, 3, 1)
        self.res = None if c1 == c2 else Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        residual = x if self.res is None else self.res(x)
        b = self.cv2(x)
        b_1 = self.cv3_1(self.att(b) * self.affine(b))
        b_2 = self.cv3_2(b)
        return residual + self.merge(torch.cat([b_1, b_2], dim=1))


class SDF(nn.Module):
    """Semantic-aware Dual-path Fusion."""

    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c1 if c2 is None else c2
        c_ = max(8, c2 // 2)
        self.sem = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c1, c_, 1, 1, act=False),
            nn.SiLU(inplace=False),
            Conv(c_, c2, 1, 1, act=False),
            nn.Sigmoid(),
        )
        self.detail = Conv(c1, c2, 3, 1)
        self.fuse = Conv(2 * c2, c2, 1, 1, act=False)
        self.res = c1 == c2

    def forward(self, x):
        detail = self.detail(x)
        sem = self.sem(x)
        y = self.fuse(torch.cat([detail, detail * sem], dim=1))
        return y + x if self.res else y


class SpermDetSDF(nn.Module):
    """Faithful semantic-aware dual-path fusion from SpermDet.

    Input is [x_h, x_l]: deep/high-level feature and shallow/low-level feature.
    """

    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c_h = c1
        self.c_l = c2
        self.proj_l = None if c2 == c1 else Conv(c2, c1, 1, 1)
        self.sa = nn.Sequential(
            Conv(c1, c1 // 2, 1, 1),
            nn.Conv2d(c1 // 2, c1, 1, 1),
            nn.Sigmoid(),
        )
        self.refine1 = Bottleneck(c1, c1)
        self.refine2 = Bottleneck(c1, c1)
        self.merge = Conv(2 * c1, c1, 3, 1)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x_h, x_l = x[0], x[1]
        else:
            x_h, x_l = x.split([self.c_h, self.c_l], dim=1)
        x_l_orig = x_l
        x_l_proj = x_l if self.proj_l is None else self.proj_l(x_l)
        sa = self.sa(x_h)
        x_l_ref = torch.cat([self.refine1(x_l_proj * sa), self.refine2(x_l_proj * (1 - sa))], dim=1)
        fused = self.merge(x_l_ref) + x_h
        return torch.cat([fused, x_l_orig], dim=1)


class MDFMV2(nn.Module):
    """Stronger multi-scale difference fusion with local/dilated/global branches."""

    def __init__(self, c1):
        super().__init__()
        c_ = max(8, c1 // 2)
        self.local = Conv(c1, c_, 3, 1)
        self.dilated = Conv(c1, c_, 3, 1, d=3)
        self.global_pool = nn.AvgPool2d(5, 1, 2)
        self.global_conv = Conv(c1, c_, 1, 1)
        self.fuse = Conv(4 * c_, c1, 1, 1, act=False)

    def forward(self, x):
        local = self.local(x)
        dilated = self.dilated(x)
        global_feat = self.global_conv(self.global_pool(x))
        diff = torch.abs(local - global_feat)
        return x + self.fuse(torch.cat([local, dilated, global_feat, diff], dim=1))


class DyHeadBlock(nn.Module):
    """Small task-aware and scale-aware attention block used by Detect_DyHead."""

    def __init__(self, c):
        super().__init__()
        self.task = nn.Sequential(Conv(c, c, 1, 1, act=False), nn.Sigmoid())
        self.scale = nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv(c, c, 1, 1, act=False), nn.Sigmoid())

    def forward(self, x):
        return x + x * self.task(x) * self.scale(x)


class Detect_DyHead(Detect):
    """YOLO Detect head with a lightweight Dynamic-Head-style preprocessing block."""

    def __init__(self, nc=1, ch=()):
        super().__init__(nc=nc, ch=tuple(ch))
        self.dy_blocks = nn.ModuleList(DyHeadBlock(c) for c in ch)

    def forward(self, x):
        x = [self.dy_blocks[i](xi) for i, xi in enumerate(x)]
        return super().forward(x)


class Detect_DyHeadReg(Detect):
    """Gate head with extra capacity on the REGRESSION branch only.

    Why this exists
    ---------------
    The measured diagnosis for this project is that MOTA is limited by *box-size
    regression* on 13-29 px objects: recall at IoU 0.3 is 0.84-0.99, the centre
    offset is a stable 0.083-0.106 of a GT side, and the gap between IoU-0.3 and
    IoU-0.5 recall tracks the size ratio monotonically.  Box regression is `cv2`;
    classification is `cv3`.

    In the v8 stack MSCAM and MDFM were applied after *every* C3k2 in the neck
    and head -- expensive (MDFM x6 is 12.7% of the model) and aimed at both
    branches and at every level.  This head applies the same blocks to the input
    of `cv2` at the finest detection level only and leaves `cv3` untouched:
    capacity where the measured error is, and none where it is not.

    The blocks are spliced INTO ``cv2`` itself rather than into a rewritten
    forward pass.  A first version transcribed ``Detect.forward`` from the
    repo-bundled ultralytics, but the installed version returns a dict in
    training mode (``preds["boxes"]`` for the loss) while the bundled one returns
    a list, so the transcription broke training on the server with
    ``TypeError: list indices must be integers or slices, not str``.  Prepending
    to ``cv2`` leaves ``forward`` entirely to the installed ultralytics, so the
    layer is version-agnostic, and ``cv3`` is untouched by construction rather
    than by careful transcription.
    """

    def __init__(self, nc=1, ch=(), reg_levels=(0,), blocks=("MSCAM", "MDFM")):
        super().__init__(nc=nc, ch=tuple(ch))
        self.dy_blocks = nn.ModuleList(DyHeadBlock(c) for c in ch)
        self.reg_levels = tuple(int(i) for i in reg_levels)
        kinds = {"MSCAM": MSCAM, "MDFM": MDFM}
        for i in self.reg_levels:
            c = int(self.cv2[i][0].conv.in_channels)
            pre = nn.Sequential(*[kinds[b](c) for b in blocks])
            self.cv2[i] = nn.Sequential(pre, *list(self.cv2[i].children()))

    def forward(self, x):
        x = [self.dy_blocks[i](xi) for i, xi in enumerate(x)]
        return super().forward(x)


class DyHeadBlockV2(nn.Module):
    """Task, scale, and spatial attention with residual connection."""

    def __init__(self, c):
        super().__init__()
        c_ = max(8, c // 4)
        self.task = nn.Sequential(Conv(c, c, 1, 1, act=False), nn.Sigmoid())
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c, c_, 1, 1, act=False),
            nn.SiLU(inplace=False),
            Conv(c_, c, 1, 1, act=False),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(Conv(c, 1, 3, 1, act=False), nn.Sigmoid())

    def forward(self, x):
        return x + x * self.task(x) * self.scale(x) * self.spatial(x)


class Detect_DyHeadV2(Detect):
    """Stronger Detect head with V2 DyHead blocks."""

    def __init__(self, nc=1, ch=()):
        super().__init__(nc=nc, ch=tuple(ch))
        self.dy_blocks = nn.ModuleList(DyHeadBlockV2(c) for c in ch)

    def forward(self, x):
        x = [self.dy_blocks[i](xi) for i, xi in enumerate(x)]
        return super().forward(x)


class CrossScaleDyHeadBlock(nn.Module):
    """Task + scale + spatial attention with cross-scale context."""

    def __init__(self, c, ctx_c=None):
        super().__init__()
        ctx_c = c if ctx_c is None else ctx_c
        c_ = max(8, c // 4)
        self.task = nn.Sequential(Conv(c, c, 1, 1, act=False), nn.Sigmoid())
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c, c_, 1, 1, act=False),
            nn.SiLU(inplace=False),
            Conv(c_, c, 1, 1, act=False),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(Conv(c, 1, 3, 1, act=False), nn.Sigmoid())
        self.ctx_proj = None if ctx_c == c else Conv(ctx_c, c, 1, 1, act=False)

    def forward(self, x, ctx):
        if self.ctx_proj is not None:
            ctx = self.ctx_proj(ctx)
        return x + x * self.task(x) * self.scale(x) * self.spatial(x + ctx)


class Detect_DyHeadV3(Detect):
    """Detect head with cross-scale dynamic attention blocks."""

    def __init__(self, nc=1, ch=()):
        super().__init__(nc=nc, ch=tuple(ch))
        self.c_common = min(ch)
        self.scale_proj = nn.ModuleList(
            Conv(c, self.c_common, 1, 1, act=False) if c != self.c_common else nn.Identity()
            for c in ch
        )
        self.blocks = nn.ModuleList(CrossScaleDyHeadBlock(c, self.c_common) for c in ch)

    def forward(self, x):
        ref_h = min(xi.shape[-2] for xi in x)
        ref_w = min(xi.shape[-1] for xi in x)
        pooled = [
            self.scale_proj[i](F.adaptive_avg_pool2d(xi, (ref_h, ref_w)))
            for i, xi in enumerate(x)
        ]
        ctx = torch.stack(pooled, dim=0).mean(0)
        out = []
        for i, xi in enumerate(x):
            ctx_i = F.interpolate(ctx, size=xi.shape[-2:], mode="nearest")
            out.append(self.blocks[i](xi, ctx_i))
        return super().forward(out)


class CueFiLM(nn.Module):
    """Image-level texture cue -> per-level (gain, bias) for the detect head.

    Why this exists
    ---------------
    Measured: every deployment session needs a multiplicative correction of the
    detector's box size (0.90 / 1.40 / 1.50 / 0.95).  That single per-session
    factor is worth up to +0.26 MOTA out of distribution and is the largest
    content defect we have measured.  Two alternatives were tested and both
    failed: rescaling the *input* to restore the apparent object size (mean
    MOTA 0.369 vs 0.664 for box scaling) and multi-scale fusion (0.6208 /
    0.5571 vs 0.6639 for the single 448 pass).  So the defect lives in the box
    *output*, and the fix has to be in the head.

    The label-free calibration rule already proves the information is there: a
    single constant times a texture cue predicts the per-session factor with
    13.6% error, and the constant is stable across models (3.07 / 3.16 / 3.18).
    This module computes that cue *inside the graph* (from the finest feature
    map, so no extra input and no extra deployment work) and turns it into a
    per-image, per-level affine modulation of the head's inputs -- i.e. a
    self-calibrating head.

    The last layer is zero-initialised, so at step 0 the modulation is the
    identity and the arm starts exactly at the delivered head (single variable).
    """

    def __init__(self, nl: int, hidden: int = 8):
        super().__init__()
        self.nl = nl
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 2 * nl))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer(
            "lap",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3),
            persistent=False,
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        g = x0.mean(1, keepdim=True)                       # (B,1,H,W) scalar field
        lap = F.conv2d(g, self.lap.to(g.dtype), padding=1)  # high-pass
        hi = lap.pow(2).mean((2, 3))                       # high-frequency energy
        var = g.var((2, 3), unbiased=False)                # contrast
        amp = g.abs().mean((2, 3))                         # level
        cue = torch.stack([torch.log1p(hi), torch.log1p(var), torch.log1p(amp)], -1)
        return self.net(cue).view(x0.shape[0], self.nl, 2)


class Detect_SelfCal(Detect):
    """Detect head modulated per image by an in-graph magnification/texture cue.

    Single-variable arm: the head is the stock `Detect` plus `CueFiLM` (58 extra
    parameters) applied to the per-level inputs before the stock box/class
    branches.  Because the modulation acts on the tensors that *both* the loss
    and the inference path decode, no change to the loss or to the decode is
    needed -- training and deployment stay consistent.
    """

    def __init__(self, nc: int = 1, ch: tuple = ()):
        super().__init__(nc=nc, ch=tuple(ch))
        self.film = CueFiLM(self.nl)

    def forward(self, x):
        ab = self.film(x[0])
        out = []
        for i, xi in enumerate(x):
            a = ab[:, i, 0].view(-1, 1, 1, 1)
            b = ab[:, i, 1].view(-1, 1, 1, 1)
            out.append(xi * (1.0 + a) + b)
        return super().forward(out)


class CueGain(nn.Module):
    """E12b: a *constrained*, cue-driven multiplicative box-size gain.

    What v1 got wrong (measured, 04:00): the cue->gain path was a free MLP
    supervised only by the detection loss, so it collapsed to a *constant*
    (gain 0.248 for every session) -- i.e. it used the knob as a global feature
    scaler and never calibrated anything (S4 rho=+0.18, S2/S3 collapsed).

    Three hard constraints here:

    1. **multiplicative only** -- `g = exp(s * z + b)`, no additive bias, so the
       knob can only scale box sizes;
    2. **cue-driven with the correct sign** -- `z` is a dimensionless texture
       statistic of the finest feature map, `s` starts at +0.5 (the direction the
       label-free kappa rule implies: higher texture energy -> larger boxes), so
       a session-dependent signal exists from step 0 instead of being free;
    3. **applied to the decoded box distances**, i.e. after the box branch's
       BatchNorm, where a gain actually changes box size (a gain on the *input*
       of a BN stack is largely cancelled by the BN).

    Only two scalars are learnable, and the gain is computed once per image.
    """

    def __init__(self, s_init: float = 0.5, b_init: float = 0.0):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(float(s_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))
        self.register_buffer(
            "lap",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3),
            persistent=False,
        )

    def cue(self, x0: torch.Tensor) -> torch.Tensor:
        g = x0.mean(1, keepdim=True)
        lap = F.conv2d(g, self.lap.to(g.dtype), padding=1)
        # Same form as the label-free rule s = kappa * sqrt(f_mean * lapvar):
        #     log s = log kappa + 0.5 * (log f_mean + log lapvar)
        # NOTE: it must be ABSOLUTE, not contrast-normalised.  The first version
        # used log(hi/base), which is invariant to an overall amplitude scaling
        # -- that erased exactly the information the rule is built on (a 4x
        # contrast change moved the gain by only 0.003).
        sharp = lap.abs().mean((2, 3)).clamp(min=1e-12)        # "f_mean" analogue
        lapvar = lap.pow(2).mean((2, 3)).clamp(min=1e-12)      # "lapvar"
        return 0.5 * (torch.log(sharp) + torch.log(lapvar))    # (B,1)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        z = self.cue(x0).squeeze(-1)               # (B,)
        return torch.exp(self.s * z + self.b)      # (B,)


class GainDFL(nn.Module):
    """Wraps a head's DFL so the decoded distances are scaled by the last gain."""

    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self._head = [head]                        # plain list: no module cycle

    def forward(self, x):
        out = self.base(x)
        head = self._head[0]
        g = getattr(head, "_last_gain", None)
        if g is not None and out.dim() == 3:
            out = out * g.view(-1, 1, 1)
        return out


def selfcal_input_stash(head, module, args):
    """Stash the network input so the gain can be computed *before* BatchNorm.

    Why this matters (measured 05:0x): computing the cue from the finest
    *feature* map made the gain identical for two images whose contrast differed
    4x -- the backbone's BN normalises exactly the global contrast/energy
    statistic the label-free rule is built on.  So v1's inability to calibrate
    was not only "the knob was free", the knob had (almost) no session signal to
    begin with.  Register with
        head_layer.register_forward_pre_hook(functools.partial(selfcal_input_stash, head))
    """
    if args and torch.is_tensor(args[0]) and args[0].dim() == 4:
        head._last_input = args[0]
    return None


def selfcal_v2_pre_hook(module, args):
    """Compute the per-image gain from the finest feature map and stash it.

    Registered on the *head* (not on the box branch) so it always sees x[0];
    `GainDFL` and the patched loss then read `head._last_gain`, which keeps
    training and inference consistent -- the loss in this ultralytics version
    decodes with its **own** DFL, so a gain applied only inside the head would
    never be trained.
    """
    gain_mod = getattr(module, "gain_mod", None)
    if gain_mod is None or not args:
        return None
    src = getattr(module, "_last_input", None)
    x = args[0]
    if src is None:
        if not isinstance(x, (list, tuple)) or not x or not torch.is_tensor(x[0]):
            return None
        src = x[0]
    module._last_gain = gain_mod(src)              # (B,) tensor
    return None                                    # features untouched


# NOTE (found 2026-09-23): this patch used to reach the head through
# `self.model`, but this ultralytics version's v8DetectionLoss does NOT store the
# model (`__init__` keeps a local `m = model.model[-1]`), so `head` was always
# None and the multiplication never happened.  Consequence: the E12b/E12c/E31
# "learned" gain scalars never received any gradient and stayed exactly at their
# initialisation (verified: s=0.50000, b=2.44531 for E12c; s=0.25000, b=1.22266
# for E31).  Their numbers are still valid -- they are the label-free cue rule
# applied inside the decoder -- but they are NOT a learned module.  A
# module-level registry (below) makes the loss path actually work for new arms.
_GAIN_LOSS_HEAD: dict = {"head": None}


def install_gain_aware_loss(head=None):
    """Make v8DetectionLoss's own decode see the head's gain."""
    import ultralytics.utils.loss as ul
    if head is not None:
        _GAIN_LOSS_HEAD["head"] = head
    if getattr(ul.v8DetectionLoss, "_gain_aware", False):
        return
    _orig = ul.v8DetectionLoss.bbox_decode

    def bbox_decode(self, anchor_points, pred_dist):
        d = _orig(self, anchor_points, pred_dist)
        head = _GAIN_LOSS_HEAD.get("head")
        g = getattr(head, "_last_gain", None) if head is not None else None
        if g is not None and d.dim() == 3 and d.shape[0] == g.shape[0]:
            d = d * g.view(-1, 1, 1)
        return d

    ul.v8DetectionLoss.bbox_decode = bbox_decode
    ul.v8DetectionLoss._gain_aware = True


class LogSizeBboxLoss(nn.Module):
    """BboxLoss + an explicit **multiplicative box-size** penalty.

    Mechanism (all measured on this project):

    * the dominant detection-content defect is a *per-session multiplicative*
      box-size bias: the deployment sessions need 0.90 / 1.40 / 1.50 / 0.95,
      i.e. the predicted boxes are up to 1.5x too small, while the centres are
      right.  Rescaling boxes fixes it and is worth **+0.26 MOTA** out of
      distribution; rescaling the *input* does not (0.369 vs 0.664, E14);
    * MOT's IoU-0.5 rule turns a 1.4-1.5x size error into a *complete* miss
      (IoU ~0.44-0.51 against a perfectly centred box), while CIoU still sees a
      comfortable overlap (>=0.7) and therefore provides only a weak gradient
      against exactly the error that destroys MOTA.

    Fix: add `|log(w_pred/w_gt)| + |log(h_pred/h_gt)|` on the foreground boxes,
    weighted by the same target-score weight and normaliser the other terms use.
    The log makes the penalty scale-invariant, which matches the measured
    failure mode (a constant factor per session, not an additive offset).

    Single-variable arm: only the bbox loss changes; architecture, data, split,
    augmentation and the ratio between IoU and DFL stay as delivered.
    """

    def __init__(self, reg_max: int = 16, size_weight: float = 0.25, eps: float = 1e-6):
        super().__init__()
        from ultralytics.utils.loss import BboxLoss
        self.base = BboxLoss(reg_max)
        self.size_weight = float(size_weight)
        self.eps = float(eps)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        loss_iou, loss_dfl = self.base(pred_dist, pred_bboxes, anchor_points,
                                       target_bboxes, target_scores,
                                       target_scores_sum, fg_mask, imgsz, stride)
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if weight.numel() == 0:
            return loss_iou, loss_dfl
        p = pred_bboxes[fg_mask]
        t = target_bboxes[fg_mask]
        pw = (p[:, 2] - p[:, 0]).clamp(min=self.eps)
        ph = (p[:, 3] - p[:, 1]).clamp(min=self.eps)
        tw = (t[:, 2] - t[:, 0]).clamp(min=self.eps)
        th = (t[:, 3] - t[:, 1]).clamp(min=self.eps)
        log_size = (torch.log(pw / tw).abs() + torch.log(ph / th).abs())
        loss_size = (log_size.unsqueeze(-1) * weight).sum() / target_scores_sum
        return loss_iou + self.size_weight * loss_size, loss_dfl


def install_logsize_loss(size_weight: float = 0.25):
    """Swap the bbox loss used by every v8DetectionLoss created afterwards."""
    import ultralytics.utils.loss as ul
    _orig_init = ul.v8DetectionLoss.__init__

    def patched_init(self, model, tal_topk=10):
        _orig_init(self, model, tal_topk=tal_topk)
        device = next(model.parameters()).device
        self.bbox_loss = LogSizeBboxLoss(reg_max=self.reg_max,
                                         size_weight=size_weight).to(device)

    ul.v8DetectionLoss.__init__ = patched_init
    return LogSizeBboxLoss


def selfcal_head_pre_hook(module, args):
    """Forward pre-hook form of the self-calibrating modulation.

    Why a hook instead of `head.forward_head = closure`: a closure defined inside
    `AblationTrainer.get_model` cannot be pickled, so `torch.save` of the first
    checkpoint died with
        AttributeError: Can't pickle local object
        'AblationTrainer.get_model.<locals>.calibrated_head_head'
    (02:16 crash).  A module-level function is picklable, and the modulation
    weights live on the head as a real submodule (`film_mod`), so the checkpoint
    round-trips.

    It only rewrites the head's input list; every return value of the head is
    left exactly as stock (which is what the loss, the validator and NMS read).
    """
    film = getattr(module, "film_mod", None)
    if film is None or not args:
        return None
    x = args[0]
    if not isinstance(x, (list, tuple)) or not x or not torch.is_tensor(x[0]):
        return None
    ab = film(x[0])
    out = []
    for i, xi in enumerate(x):
        ga = ab[:, i, 0].view(-1, 1, 1, 1)
        gb = ab[:, i, 1].view(-1, 1, 1, 1)
        out.append(xi * (1.0 + ga) + gb)
    return (out,) + tuple(args[1:])


def register():
    import ultralytics.nn.modules as ulm
    import ultralytics.nn.tasks as tasks

    classes = (
        CueFiLM, Detect_SelfCal, LogSizeBboxLoss, CueGain, GainDFL,
        SPDConv, SPDLite, CMUNeXtBlock, StarBlock, RCA, RCM,
        MSCAM, MDFM, GatedCtx, Detect_DyHead, Detect_DyHeadReg,
        MSCAMV2, MDFMV2, Detect_DyHeadV2, Detect_DyHeadV3, EMA, C3k2_EMA, SE, GSConv, GSBottleneck, VoVGSCSP, SAFM,
        SegNextAttention,
        SubPixelUpsample, CARAFE, DySample, LCEB, SDF, SpermDetLCEB, SpermDetSDF,
        InceptionDWConv2d, ConvMlp, MetaDWBlock, MetaDWStage, MetaNeXtStage,
        FRFCFuse, IBS_D, IBS_U,
        NWDBboxLoss, PixelNWDBboxLoss, InnerMPDIoUNWDBboxLoss, WIoUv1BboxLoss, WIoUv3BboxLoss,
    )
    for cls in classes:
        setattr(tasks, cls.__name__, cls)
        setattr(ulm, cls.__name__, cls)
    import sys as _sys
    _main = _sys.modules.get("__main__")
    if _main is not None:
        _main.NWDBboxLoss = NWDBboxLoss
        _main.PixelNWDBboxLoss = PixelNWDBboxLoss
        _main.InnerMPDIoUNWDBboxLoss = InnerMPDIoUNWDBboxLoss
    print("[MMSC] registered:", ", ".join(dict.fromkeys(cls.__name__ for cls in classes)))


if __name__ == "__main__":
    register()

#!/usr/bin/env python3
"""Generic single-variable ablation trainer for the v8 detector family."""

import os

import mmsc_modules
from mmsc_modules import Detect_DyHead
from mmsc_modules import Detect_DyHeadReg
from mmsc_modules import Detect_SelfCal
from mmsc_modules import CueFiLM
from mmsc_modules import selfcal_head_pre_hook

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


mmsc_modules.register()

CFG = os.environ.get("CFG", "/root/code/yolo11n_mmsc_fusion_v8.yaml")
NAME = os.environ.get("NAME", "ablation")
BASE = os.environ.get("BASE", "/root/code/yolo11n.pt")
DATA = os.environ.get("DATA", "/root/code/datasets/visem_official/data.yaml")
EPOCHS = int(os.environ.get("EPOCHS", "100"))
USE_DYHEAD = int(os.environ.get("DYHEAD", "1"))
# Regression-branch ablation: with REG=1 the head keeps its gates but gains
# MSCAM/MDFM on the *box* branch (cv2) of the levels named in REG_LEVELS, while
# cv3 (classification) is left exactly as it was.  This is the "relocate the
# blocks to where the measured error is" arm: the diagnosis says MOTA is limited
# by box-size regression on 13-29 px objects, and level 0 is the finest level.
USE_REG = int(os.environ.get("REG", "0"))
# Self-calibrating head: modulate the detect head per image with an in-graph
# texture cue (the label-free calibration rule's own statistic), so the
# per-session box-size prior is learned instead of patched afterwards.
USE_SELFCAL = int(os.environ.get("SELFCAL", "0"))
# Two-frame composite input: R = current frame, G = reference frame k* back,
# B = |difference|.  Only the image pipeline changes (single-variable arm); the
# reference gap is sampled uniformly from 1..MOTION_GAP so that one model serves
# the whole deployment stride range.
USE_MOTION = int(os.environ.get("MOTION", "0"))
MOTION_GAP = int(os.environ.get("MOTION_GAP", "4"))
# Loss-side arm: add an explicit multiplicative box-size term to the bbox loss,
# because the measured dominant defect is a per-session *factor* on the box size
# that CIoU barely penalises but MOT's IoU-0.5 turns into a total miss.
USE_LOGSIZE = int(os.environ.get("LOGSIZE", "0"))
LOGSIZE_W = float(os.environ.get("LOGSIZE_W", "0.25"))
# E12b: the *constrained* self-calibration -- a cue-driven multiplicative gain on
# the decoded box distances, two learnable scalars, correct sign at init.  v1
# (free FiLM on the head inputs) collapsed to a constant gain, see
# VISEM_TASK.md 04:00.
USE_GAIN = int(os.environ.get("GAIN", "0"))
GAIN_S0 = float(os.environ.get("GAIN_S0", "0.5"))
GAIN_B0 = float(os.environ.get("GAIN_B0", "0.0"))
# E18: extra capacity on the finest-level BOX branch only.  Note that
# DYHEAD=1 REG=1 would NOT be single-variable: Detect_DyHeadReg bundles its own
# DyHead blocks.  This path splices MSCAM+MDFM into the *pretrained* head's
# cv2[0] instead, so every existing weight is kept and only the new blocks are
# fresh.
USE_REGBOX = int(os.environ.get("REGBOX", "0"))
REGBOX_BLOCKS = os.environ.get("REGBOX_BLOCKS", "MSCAM,MDFM")
REG_LEVELS = os.environ.get("REG_LEVELS", "0")
REG_BLOCKS = os.environ.get("REG_BLOCKS", "MSCAM,MDFM")
# E32: the SAME NPU-native gate that wins on the P3 neck output (E26), but
# spliced into the finest-level BOX branch only (cv2[0]) -- i.e. moved from
# "shared features" to "where the measured defect is".  Motivation is E17+E26:
#   * after E26 the remaining stride-1 FN are dominated by boxes that are close
#     but not accurate enough (47% at IoU 0.3-0.5);
#   * E18 showed the splice keeps every pretrained weight (only the new block is
#     fresh), and GatedCtx is the audited-clean block.
# This is NOT a repeat of E18 (which spliced MSCAM+MDFM at the same place): the
# block differs, and the comparison is against E26 rather than the stock model.
USE_GCTXBOX = int(os.environ.get("GCTXBOX", "0"))
# E37: per-anchor multiplicative box-size correction (see mmsc_anchor_gain.py).
# Motivation is the measured oracle ceiling: fixing every detection's box size to
# its ground truth raises MOTA from 0.5704 to 0.7270 -- no per-session constant
# can capture that, it has to be per object.
USE_ANCHORGAIN = int(os.environ.get("ANCHORGAIN", "0"))
SEED = int(os.environ.get("SEED", "42"))

# Install the two-frame composite image pipeline here, i.e. AFTER every env var
# is defined: an earlier version had this block right after mmsc_modules.register()
# and the run died with NameError: USE_MOTION (the self-cal chain crashed at
# 01:47 because of it).
if USE_MOTION:
    from motion_dataset import install_motion_dataset
    install_motion_dataset(gap_max=MOTION_GAP)
    print(f"[ablation] two-frame composite input installed "
          f"(reference gap uniform in 1..{MOTION_GAP})")

if USE_LOGSIZE:
    from mmsc_modules import install_logsize_loss
    install_logsize_loss(size_weight=LOGSIZE_W)
    print(f"[ablation] log-size bbox loss installed (weight {LOGSIZE_W})")

if USE_GAIN:
    from mmsc_modules import install_gain_aware_loss
    print("[ablation] gain-aware loss decode installed")
# save_period=1 writes an epoch*.pt per epoch; on this dataset that is ~25 GB per
# 100-epoch run and it filled the 50 GB data disk once already.  Only best.pt and
# last.pt are needed downstream, so default to no per-epoch snapshots.
SAVE_PERIOD = int(os.environ.get("SAVE_PERIOD", "-1"))
# Training resolution.  Every ablation so far was trained at 640 and then run at
# 448 on the board, which is a train/test resolution mismatch -- and the
# per-video box-scale calibration exists precisely because the box-size prior is
# learned at the training resolution.  Training at the deployment resolution
# addresses both at once.
IMGSZ = int(os.environ.get("IMGSZ", "640"))
# Augmentation.  Every arm so far ran with mosaic switched OFF (mosaic=0.0,
# close_mosaic=0) and no photometric jitter at all -- values inherited from the
# first recipe and never questioned.  The measured failure mode is low confidence
# on dim, crowded frames, which is precisely what mosaic (more objects per image)
# and brightness jitter are for, so they are exposed here rather than edited in.
MOSAIC = float(os.environ.get("MOSAIC", "0.0"))
CLOSE_MOSAIC = int(os.environ.get("CLOSE_MOSAIC", "0"))
MIXUP = float(os.environ.get("MIXUP", "0.0"))
HSV_V = float(os.environ.get("HSV_V", "0.0"))
SCALE_AUG = float(os.environ.get("SCALE_AUG", "0.5"))


class AblationTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose).to(self.device)
        if USE_DYHEAD:
            old_head = model.model[-1]
            channels = [seq[0].conv.in_channels for seq in old_head.cv2]
            if USE_REG:
                new_head = Detect_DyHeadReg(
                    nc=model.yaml["nc"],
                    ch=channels,
                    reg_levels=tuple(int(x) for x in REG_LEVELS.split(",") if x.strip()),
                    blocks=tuple(b.strip() for b in REG_BLOCKS.split(",") if b.strip()),
                )
            else:
                new_head = Detect_DyHead(nc=model.yaml["nc"], ch=channels)
            new_head.stride = old_head.stride
            new_head.inplace = getattr(old_head, "inplace", True)
            new_head.i = old_head.i
            new_head.f = old_head.f
            new_head.type = "Detect_DyHeadReg" if USE_REG else "Detect_DyHead"
            new_head.np = sum(p.numel() for p in new_head.parameters())
            model.model[-1] = new_head
            model = model.to(self.device)
            new_head.bias_init()
            print(f"[ablation] {NAME}: {new_head.type} installed channels={channels}"
                  + (f" reg_levels={REG_LEVELS} blocks={REG_BLOCKS}" if USE_REG else ""))
        if USE_SELFCAL:
            # Wrap whatever head is already installed (stock Detect, or the
            # DyHead variant) with an in-graph cue modulation.  Wrapping instead
            # of replacing keeps this a genuine single-variable change: the
            # backbone, neck and head weights/architecture are untouched, and at
            # step 0 the modulation is the identity (zero-initialised last
            # layer), so training starts exactly at the in-place model.
            head = model.model[-1]
            # NOTE: the modulation is CueFiLM alone (58 params).  Wrapping with
            # Detect_SelfCal would smuggle a whole second Detect head (751k
            # params) into the model, which would stop this from being a
            # single-variable change.
            film = CueFiLM(head.nl)
            film.to(self.device)
            head.film_mod = film
            # Use a module-level forward PRE-hook, not a closure: closures cannot
            # be pickled, so the first checkpoint save died
            # (AttributeError: Can't pickle local object ...calibrated_head_head,
            # 02:16 crash).  Hooks must also not wrap `Detect.forward`, because
            # that method builds the (y, preds) tuple the validator/NMS read.
            head.register_forward_pre_hook(selfcal_head_pre_hook)
            head.type = f"{getattr(head, 'type', 'Detect')}+SelfCal"
            print(f"[ablation] {NAME}: self-calibrating modulation installed on "
                  f"{head.type} (extra params "
                  f"{sum(p.numel() for p in film.parameters())})")
        if USE_GAIN:
            # E12b: the *constrained* version.  v1's free FiLM collapsed to a
            # constant gain (VISEM_TASK.md 04:00), so here the calibration is
            # (i) multiplicative only, (ii) driven by an in-graph texture cue
            # with the sign the label-free rule implies, and (iii) applied to the
            # decoded box distances -- after the box branch's BatchNorm, where a
            # gain can actually change box size.
            from mmsc_modules import CueGain, GainDFL, selfcal_v2_pre_hook
            from mmsc_modules import selfcal_input_stash
            import functools
            head = model.model[-1]
            gain_mod = CueGain(s_init=GAIN_S0, b_init=GAIN_B0).to(self.device)
            head.gain_mod = gain_mod
            head.dfl = GainDFL(head.dfl, head)
            # the cue must come from the INPUT image: the backbone's BatchNorm
            # erases the very contrast/energy statistic the rule is built on
            model.model[0].register_forward_pre_hook(
                functools.partial(selfcal_input_stash, head))
            head.register_forward_pre_hook(selfcal_v2_pre_hook)
            head.type = f"{getattr(head, 'type', 'Detect')}+CueGain"
            # the loss must see the same gain (module-level registry: this
            # ultralytics version's loss does not keep the model)
            install_gain_aware_loss(head)
            print(f"[ablation] {NAME}: constrained cue gain installed "
                  f"(s0={GAIN_S0}, b0={GAIN_B0}; only 2 learnable scalars)")
        if USE_REGBOX:
            # Single-variable box-branch capacity: prepend blocks to cv2[0] of
            # the LOADED head (all pretrained weights preserved).  Motivation:
            # the FN attribution (E17) shows 80% of the stride-1 FN come with a
            # detection box that is close but not accurate enough, i.e. the
            # bottleneck is box quality at the finest level.
            import torch.nn as _nn
            from mmsc_modules import MSCAM, MDFM
            kinds = {"MSCAM": MSCAM, "MDFM": MDFM}
            blocks = [b.strip() for b in REGBOX_BLOCKS.split(",") if b.strip()]
            head = model.model[-1]
            c = int(head.cv2[0][0].conv.in_channels)
            pre = _nn.Sequential(*[kinds[b](c) for b in blocks])
            head.cv2[0] = _nn.Sequential(pre, *list(head.cv2[0].children()))
            added = sum(p.numel() for b in blocks for p in kinds[b](c).parameters())
            head.type = f"{getattr(head, 'type', 'Detect')}+RegBox0"
            print(f"[ablation] {NAME}: box-branch blocks {blocks} spliced into "
                  f"cv2[0] (channels {c}, ~{added:,} new params), cv3 untouched")
        if USE_GCTXBOX:
            # E32: GatedCtx on the finest-level BOX branch only.  Same splice
            # mechanics as E18 (all pretrained weights kept, cv3 untouched), but
            # with the block that already won on the P3 neck output in E26.
            import torch.nn as _nn
            from mmsc_modules import GatedCtx
            head = model.model[-1]
            c = int(head.cv2[0][0].conv.in_channels)
            gate = GatedCtx(c)
            head.cv2[0] = _nn.Sequential(gate, *list(head.cv2[0].children()))
            head.type = f"{getattr(head, 'type', 'Detect')}+GatedCtxBox0"
            print(f"[ablation] {NAME}: GatedCtx spliced into the finest-level BOX "
                  f"branch cv2[0] (channels {c}, +{sum(p.numel() for p in gate.parameters()):,} "
                  f"params), cv3 untouched -- this is E32 (compare against E26)")
        if USE_ANCHORGAIN:
            # E37: per-anchor size gain, identity at init (zero-init 1x1 convs).
            from mmsc_anchor_gain import install_anchor_gain, install_anchor_gain_loss
            head = model.model[-1]
            ch = [int(seq[0].conv.in_channels) for seq in head.cv2]
            mod = install_anchor_gain(head, ch)
            install_anchor_gain_loss(head)
            print(f"[ablation] {NAME}: per-anchor size gain installed on "
                  f"{head.type} (+{sum(p.numel() for p in mod.parameters()):,} params, "
                  f"levels {ch}); gain == 1.0 at init")
        return model


model = YOLO(CFG).load(BASE)

model.train(
    trainer=AblationTrainer,
    data=DATA,
    epochs=EPOCHS,
    patience=0,
    imgsz=IMGSZ,
    batch=16,
    device=0,
    workers=8,
    amp=False,
    optimizer="SGD",
    lr0=0.001,
    lrf=0.0001,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=10,
    warmup_momentum=0.8,
    cos_lr=True,
    close_mosaic=CLOSE_MOSAIC,
    mosaic=MOSAIC,
    mixup=MIXUP,
    hsv_v=HSV_V,
    scale=SCALE_AUG,
    multi_scale=False,
    label_smoothing=0.1,
    project="runs/visem_official",
    name=NAME,
    exist_ok=True,
    seed=SEED,
    rect=True,
    iou=0.5,
    max_det=500,
    save_period=SAVE_PERIOD,
)

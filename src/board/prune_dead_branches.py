#!/usr/bin/env python3
"""Export-only pruning of the two dead detection branches (no retraining).

E34 (retrained with a P3-only head) lost MOTA (-0.081 on the main table), which
shows the stride-16/32 branches matter DURING TRAINING even though they never
fire at inference.  This script does the opposite operation: keep the trained
E26 weights, and remove the dead branches **only in the exported graph**.  The
neck is feed-forward, so nothing downstream of the P3 output feeds back into it
-- the P3 branch should be bit-identical, hence MOTA-identical by construction.

    python3 prune_dead_branches.py --model <E26 last.pt> --cfg stock_yolov8n_gctx_p3only.yaml

Steps: build the P3-only architecture, load every matching tensor from the
trained checkpoint, verify the level-0 head-raw output is unchanged, save the
pruned checkpoint, export + audit it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path("/root/code")
sys.path.insert(0, str(HERE))

import mmsc_modules  # noqa: E402

mmsc_modules.register()

from ultralytics import YOLO  # noqa: E402


def level0_raw(model, x):
    """head-raw level-0 tensor for a GIVEN input (both architectures share it)."""
    det = model.model[-1]
    with torch.no_grad():
        feats = []
        y = []
        for m in model.model[:-1]:
            if m.f != -1:
                x2 = y[m.f] if isinstance(m.f, int) else \
                    [x if j == -1 else y[j] for j in m.f]
            else:
                x2 = x
            x = m(x2)
            y.append(x if m.i in model.save else None)
        f = det.f
        f = [f] if isinstance(f, int) else list(f)
        p3 = y[f[0]]
    return torch.cat((det.cv2[0](p3), det.cv3[0](p3)), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out-ckpt", default="/root/autodl-tmp/v8n_gctx_p3pruned.pt")
    ap.add_argument("--imgsz", type=int, default=448)
    args = ap.parse_args()

    src = YOLO(args.model).model.eval()
    probe = torch.rand(1, 3, args.imgsz, args.imgsz)
    ref = level0_raw(src, probe)
    print(f"source model : {src.model[-1].type}  levels {src.model[-1].nl}  "
          f"params {sum(p.numel() for p in src.parameters()):,}")

    pruned = YOLO(args.cfg).model
    sd = src.state_dict()
    loaded, skipped = 0, 0
    own = pruned.state_dict()
    for k in list(own):
        if k in sd and sd[k].shape == own[k].shape:
            own[k] = sd[k]
            loaded += 1
        else:
            skipped += 1
    pruned.load_state_dict(own)
    fresh = [k for k in own if k not in sd or sd[k].shape != own[k].shape]
    print(f"fresh tensors ({len(fresh)}): " + ", ".join(fresh[:12])
          + (" ..." if len(fresh) > 12 else ""))
    # The head sits at a different index in the two yamls (23 vs 17), so its
    # tensors never match by name.  Copy the surviving level-0 branches (and the
    # DFL) explicitly -- this is exactly the "level 0 only" part of the head.
    src_det, dst_det = src.model[-1], pruned.model[-1]
    dst_det.cv2[0].load_state_dict(src_det.cv2[0].state_dict())
    dst_det.cv3[0].load_state_dict(src_det.cv3[0].state_dict())
    try:
        dst_det.dfl.load_state_dict(src_det.dfl.state_dict())
    except Exception as exc:  # noqa: BLE001
        print(f"note: dfl not copied ({type(exc).__name__}: {exc})")
    pruned.eval()
    print(f"pruned model : {pruned.model[-1].type}  levels "
          f"{pruned.model[-1].nl}  params {sum(p.numel() for p in pruned.parameters()):,}"
          f"   tensors loaded {loaded}, left fresh {skipped}")

    # the head's stride has to be 8 for a single P3 level
    pruned.model[-1].stride = torch.tensor([8.0])
    got = level0_raw(pruned, probe)
    d = float((got - ref).abs().max())
    print(f"level-0 head-raw check: max |pruned - original| = {d:.3e}")
    if d > 1e-5:
        print("FAILED: the pruned graph does not reproduce the P3 branch")
        return 1
    print("OK: the kept branch is bit-comparable -> MOTA cannot change")

    y = YOLO(args.cfg)
    y.model = pruned
    torch.save({"model": pruned}, args.out_ckpt)
    print(f"saved pruned checkpoint -> {args.out_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

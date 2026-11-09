#!/usr/bin/env python3
"""Export a VISEM checkpoint to ONNX in the format rknn_convert.py expects.

Differences from onnx_export_norm.py (which this replaces for MMSC models):
  * calls mmsc_modules.register() first -- without it a checkpoint containing
    SPDConv / CMUNeXtBlock / StarBlock / MSCAM / MDFM / C3k2_EMA / Detect_DyHead
    cannot be unpickled at all;
  * the ghost_conv import is no longer hard-required (it is imported only if
    it is present, for the older ghost checkpoints);
  * --check-only loads the model and runs one forward pass without exporting,
    so a registration or head problem shows up in seconds instead of after a
    full export;
  * prints the output shape and the score range, so a silently-broken export
    (all-zero output, wrong layout) is caught here rather than on the board.

Output contract, matching rknn_convert.py and run_tracking_deepocsort.py:
  input  "images"   float32 NCHW, RGB, 0-1   (RKNN applies mean=0 std=255)
  output "output0"  (1, 4 + nc, 8400) with the four box coordinates divided by
                    imgsz, i.e. normalised to [0, 1]

    # preflight only (seconds)
    python export_onnx_deploy.py --model <best.pt> --imgsz 640 --check-only

    # real export
    python export_onnx_deploy.py --model <best.pt> --imgsz 640 \
        --out runs/deploy/lkctx_640.onnx

    # audit the exported graph for ops RKNN cannot schedule
    python audit_onnx_ops.py --onnx runs/deploy/lkctx_640.onnx
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import types
from pathlib import Path

ROOT = Path("/root/code")
_HERE = Path(__file__).resolve().parent
# Local smoke-test convenience: prefer a bundled ultralytics source tree if the
# package is not installed.  No effect on the server, where it is installed.
for _extra in (_HERE / "ultralytics-8.3.163", ROOT, _HERE):
    if _extra.is_dir() and str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import torch  # noqa: E402

import mmsc_modules  # noqa: E402

mmsc_modules.register()

with contextlib.suppress(Exception):  # older ghost checkpoints only
    import ghost_conv  # noqa: F401

from ultralytics import YOLO  # noqa: E402


def load_model(model_path: str, device: str, fuse: bool):
    m = YOLO(model_path)
    if not fuse:
        # Matches the previous working exports: skip Conv+BN fusion so the graph
        # stays as trained.  Pass --fuse to merge BN before exporting.
        m.model.fuse = lambda verbose=False: m.model
    model = m.model.to(device).eval()
    det = model.model[-1]
    det.export = True
    return model, det


def patch_normalised_boxes(det, imgsz: int, clamp: bool = True):
    """Divide the decoded boxes by imgsz so the ONNX output is in [0, 1].

    ``clamp`` also clips to [0, 1].  Background anchors have unconstrained box
    regression, so a 4-scale model can decode boxes far outside the image:
    ablation_lkctx30 spans [-148, 866] px raw ([-0.23, 1.35] normalised) while
    the 3-scale models stay inside [0, 640].  That matters for INT8: an output
    range of 1.58 normalised over 255 levels is a 0.0062 step, i.e. 4 px at 640,
    against 19-29 px targets.  Clipping makes the output range exactly [0, 1],
    and the boxes being clipped are outside the frame anyway.
    """
    # Keep the untouched method so repeated patching does not wrap itself and
    # divide twice.
    orig = getattr(det, "_inference_orig", None) or det._inference
    det._inference_orig = orig
    size = float(imgsz)

    def _inference_norm(self, x):
        y = orig(x).clone()
        y[:, :4] = y[:, :4] / size
        if clamp:
            y[:, :4] = y[:, :4].clamp_(0.0, 1.0)
        return y

    det._inference = types.MethodType(_inference_norm, det)


def load_letterboxed(path: str, imgsz: int) -> torch.Tensor:
    """Letterbox to imgsz with the same 114 padding value the pipeline uses."""
    import cv2
    import numpy as np

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    r = imgsz / max(h, w)
    nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    dy, dx = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = img
    return torch.from_numpy(canvas).permute(2, 0, 1).float().div_(255).unsqueeze(0)


def patch_raw_head(det):
    """Return the per-scale cat(cv2, cv3) tensors instead of the decoded output.

    This is the "no-DFL" / head-raw export.  Detect.forward has already computed
    x[i] = cat(cv2[i](xi), cv3[i](xi)) by the time _inference is called, so
    returning that list drops the DFL softmax, the anchor decode and the
    normalisation/clamp in one move.

    Two independent reasons to want it:

    * speed.  On RK3568 the DFL/decode tail contributes Softmax + 3x Transpose +
      13x Reshape + ScatterND, all on the A55 CPU.  The measured CPU segment is
      127.7 ms of a 294.6 ms frame, and INT8 does not speed any of it up: measured
      board numbers for the same model are 318 ms FP16 vs 192 ms INT8, i.e. only
      1.66x overall even though the NPU part is 3.0x faster.

    * INT8 correctness.  asymmetric_quantized-8 converts this graph without a
      single 'unsupported' warning and then produces an ALL-ZERO output, on the
      simulator and on the board alike.  asymmetric_quantized-16 fails outright
      with "unsupported tensor dtype in Sub" at /model.23/Constant_9, i.e. inside
      the decode.  The project already has working INT8 models built the other
      way (talp_nodfl_r1_full_416_int8.rknn), so the decode tail is the suspect.

    Output contract matches export_head_raw_onnx.py, so the existing
    rknn_headless_detect_raw.py / rknn_headless_nodfl_pipeline.py consumers apply.
    """
    orig = getattr(det, "_inference_orig", None) or det._inference
    det._inference_orig = orig

    def _inference_raw(self, x):
        return tuple(x)

    det._inference = types.MethodType(_inference_raw, det)


def unwrap_outputs(out) -> list:
    """Normalise whatever the head returns into a flat list of tensors."""
    if torch.is_tensor(out):
        return [out]
    if isinstance(out, (list, tuple)):
        if out and all(torch.is_tensor(o) for o in out):
            return list(out)
        # Detect.forward returns (y, x) when self.export is False
        for o in out:
            if isinstance(o, (list, tuple)) and o and all(torch.is_tensor(t) for t in o):
                return list(o)
    return [out]


def self_test() -> int:
    """Check the two things that silently break this export.

    1. the normalisation patch divides the four box columns and leaves the class
       scores alone (getting this wrong produced the all-zero INT8 outputs);
    2. the checkpoint classes are registered, so a MMSC .pt can be unpickled.
    """

    class FakeDet(torch.nn.Module):
        def _inference(self, x):
            # real Detect._inference returns (B, 4 + nc, N)
            return torch.tensor([[[2.0, 3.0, 9.0],
                                  [4.0, 5.0, 9.0],
                                  [6.0, 7.0, 9.0],
                                  [8.0, 9.0, 9.0],
                                  [0.5, 0.4, 0.3],
                                  [0.25, 0.2, 0.1]]])

    det = FakeDet()
    patch_normalised_boxes(det, 640, clamp=False)
    y = det._inference(None)
    assert tuple(y.shape) == (1, 6, 3), y.shape
    got = [[round(v, 8) for v in y[0, c, :].tolist()] for c in range(4)]
    want = [[round(v / 640, 8) for v in (2.0, 3.0, 9.0)],
            [round(v / 640, 8) for v in (4.0, 5.0, 9.0)],
            [round(v / 640, 8) for v in (6.0, 7.0, 9.0)],
            [round(v / 640, 8) for v in (8.0, 9.0, 9.0)]]
    assert got == want, f"box normalisation wrong:\n  got  {got}\n  want {want}"
    assert [y[0, c, 0].item() for c in (4, 5)] == [0.5, 0.25], "class columns were modified"

    class BigDet(torch.nn.Module):
        def _inference(self, x):
            # one box above the frame, one to the left of it
            return torch.tensor([[[800.0], [-100.0], [320.0], [320.0], [0.9], [0.1]]])

    big = BigDet()
    patch_normalised_boxes(big, 640, clamp=True)
    yb = big._inference(None)
    assert yb[0, 0, 0].item() == 1.0 and yb[0, 1, 0].item() == 0.0, yb.tolist()

    big_raw = BigDet()
    patch_normalised_boxes(big_raw, 640, clamp=False)
    yr = big_raw._inference(None)
    assert abs(yr[0, 0, 0].item() - 800 / 640) < 1e-6 and abs(yr[0, 1, 0].item() + 100 / 640) < 1e-6

    # unwrap_outputs must tell a head-raw tuple of tensors apart from the
    # (y, x) pair Detect.forward returns when export is False.
    a, b = torch.zeros(1, 5, 10), torch.ones(1, 5, 10)
    assert len(unwrap_outputs(a)) == 1
    assert unwrap_outputs((a, b))[1] is b
    assert len(unwrap_outputs(([a, b, a], [a, b, a]))) == 3

    # patch_raw_head must be reversible: feeding its outputs back through the
    # saved original must reproduce the decode, which is what the export check
    # relies on.
    class FakeHead(torch.nn.Module):
        def _inference(self, x):
            return torch.cat([t.flatten(2) for t in x], 2)

    fh = FakeHead()
    xs = [torch.rand(1, 3, 4, 4), torch.rand(1, 3, 2, 2)]
    before = fh._inference(xs)
    patch_raw_head(fh)
    raw = fh._inference(xs)
    assert len(raw) == 2 and raw[0] is xs[0] and raw[1] is xs[1]
    after = fh._inference_orig(list(raw))
    assert torch.allclose(before, after), "head-raw round-trip changed the decode"

    import ultralytics.nn.tasks as tasks

    for name in ("GatedCtx", "CMUNeXtBlock", "Detect_DyHead", "MDFM", "MSCAM", "SPDConv"):
        assert hasattr(tasks, name), f"{name} missing from ultralytics.nn.tasks after register()"

    print("self-test OK: box normalisation divides only the 4 box columns; "
          "MMSC classes are registered in ultralytics.nn.tasks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="check the normalisation patch and class registration, then exit")
    ap.add_argument("--model", default=None, help="checkpoint (.pt); not needed with --self-test")
    ap.add_argument("--out", default=None, help="output .onnx path (required unless --check-only)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--opset", type=int, default=19)
    ap.add_argument("--fuse", action="store_true", help="merge Conv+BN before export")
    ap.add_argument("--simplify", action="store_true", help="run onnxsim if it is installed")
    ap.add_argument("--no-clamp", action="store_true",
                    help="do not clip the exported boxes to [0,1].  Clipping is ON by default "
                         "because a 4-scale model decodes background anchors far outside the "
                         "frame, which widens the output range and coarsens INT8 output quantisation")
    ap.add_argument("--head-raw", action="store_true",
                    help="export the raw head branches (no DFL, no decode, no sigmoid).  This is "
                         "the no-DFL export: it removes the Softmax/Transpose/Reshape tail that "
                         "costs ~46 ms of CPU per frame, and that INT8 turns into an all-zero "
                         "output.  Output layout matches export_head_raw_onnx.py")
    ap.add_argument("--check-only", action="store_true",
                    help="load the model, run one forward pass, print the output shape, do not export")
    ap.add_argument("--image", default=None,
                    help="use a real image (letterboxed to imgsz) instead of a random dummy, so the "
                         "score and box statistics below describe real detections")
    ap.add_argument("--spd-slice", action="store_true",
                    help="rewrite every SPDLite tap from the pixel_unshuffle form to the equivalent "
                         "strided-slice form before exporting.  The two are bit-identical and share "
                         "all parameters, so the checkpoint is unchanged and no retraining is "
                         "needed -- but pixel_unshuffle lowers to Reshape/Transpose (CPU ops on the "
                         "RK3568) while the slice form lowers to Slice/Concat, which the NPU runs.")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.model:
        ap.error("--model is required")
    if not args.check_only and not args.out:
        ap.error("--out is required unless --check-only")

    print(f"model : {args.model}")
    print(f"imgsz : {args.imgsz}   device: {args.device}   fuse: {args.fuse}")

    model, det = load_model(args.model, args.device, args.fuse)
    if args.spd_slice:
        import torch as _torch

        import mmsc_modules as _mmsc

        taps = [m for m in model.model.modules() if isinstance(m, _mmsc.SPDLite)]
        before = {id(m): m.mode for m in taps}
        x_probe = _torch.rand(1, 3, args.imgsz, args.imgsz, device=args.device)
        with _torch.no_grad():
            y_before = model(x_probe)
        for m in taps:
            m.mode = "slice"
        with _torch.no_grad():
            y_after = model(x_probe)

        def _flat(v):
            if isinstance(v, (list, tuple)):
                for item in v:
                    yield from _flat(item)
            else:
                yield v

        delta = max(
            float((a.detach() - b.detach()).abs().max()) for a, b in zip(_flat(y_before), _flat(y_after))
        )
        print(f"spd-slice: {len(taps)} SPDLite tap(s) {[before[id(m)] for m in taps]} -> 'slice'")
        print(f"spd-slice: max |slice_form - unshuffle_form| = {delta:.3e}")
        if delta > 1e-5:
            raise SystemExit(
                "the slice form is not equivalent to the unshuffle form on this checkpoint; "
                "refusing to export a graph that would change accuracy"
            )
        del x_probe
    n_layers = len(model.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"head  : {det.type}   layers: {n_layers}   params: {n_params:,}")
    print(f"nc    : {getattr(det, 'nc', '?')}   reg_max: {getattr(det, 'reg_max', '?')}   "
          f"in_ch: {[int(s[0].conv.in_channels) for s in det.cv2]}")
    try:
        st = det.stride
        st_txt = st.tolist() if hasattr(st, "tolist") else st
    except Exception:  # noqa: BLE001
        st_txt = "?"
    print(f"stride: {st_txt}   (the head multiplies decoded boxes by this)")

    # A deterministic [0,1] dummy: the model expects RGB in 0-1, and randn gives
    # values in [-4,4] which are far out of distribution, making the score
    # statistics below meaningless and non-reproducible between runs.  --image
    # replaces it with a real frame, which is the only way to get meaningful
    # "how many anchors fire" numbers.
    if args.image:
        dummy = load_letterboxed(args.image, args.imgsz).to(args.device)
        print(f"input  : real image {args.image} (letterboxed to {args.imgsz}, pad 114)")
    else:
        gen = torch.Generator(device="cpu").manual_seed(0)
        dummy = torch.rand(1, 3, args.imgsz, args.imgsz, generator=gen).to(args.device)
        print("input  : deterministic random [0,1] dummy (pass --image for real detections)")

    def forward():
        with torch.no_grad():
            return model(dummy)

    n_out = 1
    if args.head_raw:
        # Capture the real decoded output first, so the raw branches can be
        # checked against it: feeding them back through the original _inference
        # must reproduce the decode exactly.  Without this, a raw export whose
        # tensor order or channel split is wrong would still "look fine" here
        # and only fail in the board-side post-processing.
        decoded = unwrap_outputs(forward())[0]
        patch_raw_head(det)
        outs = unwrap_outputs(forward())
        n_out = len(outs)
        print(f"export : head-raw (no DFL, no decode, no sigmoid) -> {n_out} outputs")
        ok = True
        ref = det._inference_orig(list(outs))
        ref = unwrap_outputs(ref)[0]
        diff = (ref - decoded).abs().max().item()
        print(f"check  : max |decode(head_raw) - decode(original)| = {diff:.3e}")
        if diff > 1e-3:
            print("ERROR: the raw branches do not reproduce the original decode;")
            print("       tensor order or channel split is wrong")
            ok = False
        else:
            print("         -> raw branches verified: they reconstruct the original decode exactly")
        for i, o in enumerate(outs):
            if o.ndim != 4:
                print(f"  out{i}: expected (N, C, H, W), got {tuple(o.shape)}")
                ok = False
                continue
            box, cls = o[:, :-1], o[:, -1]
            print(f"  out{i}: {tuple(o.shape)}  box-logit {box.min().item():+.3f}..{box.max().item():+.3f}"
                  f"  cls-logit {cls.min().item():+.3f}..{cls.max().item():+.3f}")
            if not bool(torch.isfinite(o).all()):
                print(f"  out{i}: NOT FINITE")
                ok = False
            if float(o.abs().max()) == 0.0:
                print(f"  out{i}: ALL ZERO")
                ok = False
        print("  (the board-side pipeline applies the DFL softmax, anchor decode and sigmoid;")
        print("   outputs match export_head_raw_onnx.py so rknn_headless_detect_raw.py applies)")
        if args.check_only:
            print("\ncheck-only: head-raw forward " + ("OK" if ok else "FAILED"))
            return 0 if ok else 1
        if not ok:
            return 1
    else:
        # Pass 1: raw decode.
        raw = unwrap_outputs(forward())[0]
        raw_rng = (raw[:, :4].min().item(), raw[:, :4].max().item())
        print(f"raw    : shape={tuple(raw.shape)}  box min/max = {raw_rng[0]:.4f} / {raw_rng[1]:.4f}")
        print("         (un-normalised decode; pixels when the head multiplies by strides)")

        # Pass 2: normalise but do NOT clamp, so the patch can be verified exactly.
        # Comparing min/max spans cannot do this -- clamping changes the span.
        patch_normalised_boxes(det, args.imgsz, clamp=False)
        noclamp = unwrap_outputs(forward())[0]
        err = (noclamp[:, :4] * args.imgsz - raw[:, :4]).abs().max().item()
        print(f"check  : max |patched*imgsz - raw| over the box columns = {err:.3e}")
        if err > 1e-3:
            print("ERROR: the normalisation patch did not intercept _inference")
            return 1
        print("         -> patch verified (boxes are exactly raw / imgsz)")

        # Pass 3: the graph that will actually be exported.
        clamp = not args.no_clamp
        patch_normalised_boxes(det, args.imgsz, clamp=clamp)
        y = unwrap_outputs(forward())[0]
        y_rng = (y[:, :4].min().item(), y[:, :4].max().item())
        print(f"export : shape={tuple(y.shape)} dtype={y.dtype}")
        print(f"         box min/max = {y_rng[0]:.4f} / {y_rng[1]:.4f}"
              + ("  (clamped)" if clamp else f"  (NOT clamped; raw/{(1 / args.imgsz):.6f})"))

        # The range that actually matters: background anchors have unconstrained
        # box regression, so only the anchors the post-processing selects count.
        scores = y[:, 4:, :].amax(dim=1)[0]
        for thr in (0.25, 0.05):
            sel = scores > thr
            if bool(sel.any()):
                b = y[0, :4, :][:, sel]
                print(f"         over anchors with cls>{thr}: n={int(sel.sum()):>6}  "
                      f"box {b.min().item():.4f} .. {b.max().item():.4f}")
            else:
                print(f"         over anchors with cls>{thr}: none")
        step = (y_rng[1] - y_rng[0]) / 255
        print(f"         output range {y_rng[0]:.3f}..{y_rng[1]:.3f} over 255 INT8 levels "
              f"= {step:.5f} normalised = {step * args.imgsz:.2f} px at {args.imgsz}")

        print(f"         cls min/max = {y[:, 4:].min().item():.4f} / {y[:, 4:].max().item():.4f}  "
              f"(sigmoid scores; a random-noise input scores low, that is expected)")
        if y.shape[0] != 1 or y.ndim != 3 or y.shape[1] < 5:
            print("ERROR: unexpected output shape; rknn_convert.py expects (1, 4+nc, N)")
            return 1
        if y_rng[1] > 1.01 or y_rng[0] < -0.01:
            print(f"WARNING: boxes reach outside [0,1] ({y_rng[0]:.3f}..{y_rng[1]:.3f}) because "
                  f"--no-clamp was given.")
            print("         The RKNN post-processing must clamp, and the output quantisation step")
            print("         widens accordingly -- see the px figure above.")
            return 1

        if args.check_only:
            print("\ncheck-only: model loads, forwards and produces normalised boxes. OK")
            return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["images"],
        output_names=[f"output{i}" for i in range(n_out)],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"\nexported: {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify

            model_onnx = onnx.load(str(out))
            simplified, ok = simplify(model_onnx)
            if ok:
                onnx.save(simplified, str(out))
                print(f"simplified: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
            else:
                print("simplify: onnxsim reported failure, keeping the un-simplified graph")
        except Exception as exc:  # noqa: BLE001
            print(f"simplify: skipped ({type(exc).__name__}: {exc})")

    print("\nnext: python audit_onnx_ops.py --onnx " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""List the op types in an ONNX graph and flag the ones RKNN runs on the CPU.

Why
---
The RK3568 board profile (runs/deploy/rk3568_deployment_results.md) showed that
the *runtime cost* of this model family is set by how many ops rknn-toolkit2
cannot schedule on the NPU, not by FLOPs: 19 CPU operator calls cost 127.7 ms of
a 294.6 ms frame (43%) while the whole 76-conv NPU stack cost 125.3 ms.  MatMul
averaged 25.4 ms per call and Softmax 15.4 ms per call, against 1.65 ms for a
convolution.

So the useful pre-conversion question is "which unsupported ops does this graph
contain", and this script answers it from the ONNX alone -- no board, no
rknn-toolkit needed.

    python audit_onnx_ops.py --onnx runs/deploy/lkctx_640.onnx

Caveat: this is a static heuristic for RKNPU2 (rknn-toolkit2 1.6.x).  Support
does shift between toolkit versions, and a flagged op may still be handled, so
treat the flags as "verify this in the toolkit log" rather than as a verdict.
The authoritative line is the converter's own `unsupported` warning.
"""

from __future__ import annotations

import argparse
from collections import Counter

# op -> (severity, why)
HOSTILE = {
    # ---- the family that dominated the measured RK3568 profile ----
    "MatMul": (3, "measured 25.4 ms/call on the A55; the single biggest cost"),
    "Gemm": (3, "same layout overhead as MatMul when it cannot be folded"),
    "Softmax": (3, "measured 15.4 ms/call; PSABlock and the DFL head both emit one"),
    "Transpose": (3, "measured 8.7 ms/call; also fails INT8 in the earlier attempt"),
    "Einsum": (3, "lowers to MatMul/Transpose; used by DySample"),
    # ---- normalisation / attention ----
    "LayerNormalization": (3, "not scheduled on NPU by 1.6.x"),
    "GroupNormalization": (3, "used by EMA; not scheduled on NPU by 1.6.x"),
    "InstanceNormalization": (3, "not scheduled on NPU by 1.6.x"),
    "MultiHeadAttention": (3, "not supported"),
    "Attention": (3, "not supported"),
    # ---- other unsupported / fallback ops ----
    "GridSample": (3, "used by DySample/CARAFE; not supported"),
    "NonMaxSuppression": (3, "must be done on CPU anyway"),
    "RoiAlign": (3, "not supported"),
    "TopK": (3, "not supported"),
    "ScatterND": (2, "appeared in the board profile as a CPU op"),
    "GatherND": (2, "usually CPU"),
    "LSTM": (3, "not supported"),
    "GRU": (3, "not supported"),
    "Mod": (3, "not supported"),
    "Trilu": (3, "not supported"),
    # ---- layout / cheap-but-CPU ----
    "Reshape": (1, "measured 1.7 ms/call, but there were 11 of them (19.2 ms)"),
    "Shape": (1, "usually constant-folded; check if it survived"),
    "SpaceToDepth": (2, "SPDConv lowers to this; support varies by toolkit version"),
    "DepthToSpace": (2, "support varies by toolkit version"),
    "Abs": (1, "used by MDFM; check whether it stays on the NPU"),
}

SEV = {3: "HIGH", 2: "MED ", 1: "low "}


def collect(node_types) -> Counter:
    return Counter(node_types)


def flagged_ops(ops: Counter) -> dict[str, int]:
    return {op: n for op, n in ops.items() if op in HOSTILE}


def high_risk(ops: Counter) -> list[str]:
    return sorted(op for op in ops if op in HOSTILE and HOSTILE[op][0] == 3)


def self_test() -> int:
    """The verdict logic is the part that decides whether INT8 is worth trying,
    so check it against graphs built from known op lists."""
    attention = collect(["Conv"] * 40 + ["Reshape"] * 11 + ["MatMul"] * 2
                        + ["Softmax"] * 2 + ["Transpose"] * 3)
    assert high_risk(attention) == ["MatMul", "Softmax", "Transpose"], high_risk(attention)
    assert flagged_ops(attention)["Reshape"] == 11

    # the lkctx-style graph: no attention family, only the cheap layout ops
    lkctx = collect(["Conv"] * 60 + ["Concat"] * 21 + ["Reshape"] * 11
                    + ["Softmax"] * 1 + ["Sigmoid"] * 6 + ["Abs"] * 7)
    assert high_risk(lkctx) == ["Softmax"], high_risk(lkctx)
    assert "Abs" in flagged_ops(lkctx)

    clean = collect(["Conv"] * 80 + ["Concat"] * 10 + ["Sigmoid"] * 5 + ["MaxPool"] * 3)
    assert high_risk(clean) == [], high_risk(clean)
    assert flagged_ops(clean) == {}

    print("self-test OK: verdict flags the attention family, tolerates lkctx's "
          "single DFL softmax, and passes a clean conv-only graph")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="check the verdict logic, then exit")
    ap.add_argument("--onnx", default=None, help="path to .onnx; not needed with --self-test")
    ap.add_argument("--top", type=int, default=25, help="how many op types to print")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.onnx:
        ap.error("--onnx is required")

    import onnx

    model = onnx.load(args.onnx)
    ops = collect(n.op_type for n in model.graph.node)
    total = sum(ops.values())

    print(f"graph     : {args.onnx}")
    print(f"nodes     : {total} across {len(ops)} op types")
    try:
        d = [x.dim_value for x in model.graph.input[0].type.tensor_type.shape.dim]
        o = [x.dim_value for x in model.graph.output[0].type.tensor_type.shape.dim]
        print(f"input     : {d}")
        print(f"output    : {o}")
    except Exception:  # noqa: BLE001
        pass

    flagged = flagged_ops(ops)
    flagged_nodes = sum(flagged.values())
    print(f"flagged   : {flagged_nodes}/{total} nodes ({100 * flagged_nodes / max(1, total):.1f}%) "
          f"in {len(flagged)} op types")

    print(f"\n{'sev':<5}{'op':<22}{'count':>6}  why")
    print("-" * 78)
    for op, n in sorted(flagged.items(), key=lambda kv: (-HOSTILE[kv[0]][0], -kv[1])):
        print(f"{SEV[HOSTILE[op][0]]:<5}{op:<22}{n:>6}  {HOSTILE[op][1]}")

    print(f"\nall op types ({args.top} most frequent):")
    print(f"  {'op':<24}{'count':>6}")
    for op, n in ops.most_common(args.top):
        mark = "  <-- flagged" if op in HOSTILE else ""
        print(f"  {op:<24}{n:>6}{mark}")

    high = high_risk(ops)
    print()
    if high:
        print(f"VERDICT: {len(high)} high-risk op type(s) present: {', '.join(sorted(high))}")
        print("         rknn-toolkit2 will most likely run these on the A55.  Check the")
        print("         converter log for 'unsupported', then profile on the board with")
        print("         rknn_verify_npu.py before trusting any latency estimate.")
    else:
        print("VERDICT: no high-risk op type present.  Nothing in this graph matches the")
        print("         MatMul/Softmax/Transpose family that cost 43% of the frame on the")
        print("         RK3568, so INT8 has a real chance of passing where it failed before.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

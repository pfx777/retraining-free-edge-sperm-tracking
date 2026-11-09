#!/usr/bin/env python3
"""
export_head_raw_onnx.py

导出 Detect head 的“原始输出” ONNX：
不做 DFL、decode、sigmoid，只输出三个尺度的 box+cls 张量。
这样 RKNN 只需要跑卷积，DFL/decode 放到 CPU numpy。
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/root/code/runs/detect/runs/visem_tracking/talp_r45n10/weights/epoch24.pt")
parser.add_argument("--out", default="/root/code/runs/deploy/talp_head_raw.onnx")
parser.add_argument("--opset", type=int, default=12)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


class HeadRaw(nn.Module):
    def __init__(self, detect):
        super().__init__()
        self.detect = detect

    def forward(self, p3, p4, p5):
        x = [p3, p4, p5]
        outs = []
        for i in range(self.detect.nl):
            box = self.detect.cv2[i](x[i])
            cls = self.detect.cv3[i](x[i])
            outs.append(torch.cat((box, cls), 1))
        return outs


def main():
    yolo = YOLO(args.model)
    model = yolo.model.to(args.device).eval()
    detect = model.model[-1]
    wrapper = HeadRaw(detect).to(args.device).eval()

    p3 = torch.randn(1, 58, 80, 80).to(args.device)
    p4 = torch.randn(1, 116, 40, 40).to(args.device)
    p5 = torch.randn(1, 256, 20, 20).to(args.device)

    with torch.no_grad():
        outs = wrapper(p3, p4, p5)
    print("raw head output shapes:")
    for o in outs:
        print(" ", tuple(o.shape))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (p3, p4, p5),
        str(out_path),
        input_names=["p3", "p4", "p5"],
        output_names=["out0", "out1", "out2"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()

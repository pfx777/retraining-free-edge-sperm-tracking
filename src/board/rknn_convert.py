#!/usr/bin/env python3
"""Convert TALP ONNX to RK3568 RKNN with RKNN-Toolkit2.

Default: INT8 asymmetric u8 quantization (0-255 uint8 input).
Use --fp16 to build a non-quantized model for isolation testing.
"""
import argparse

from rknn.api import RKNN


def parse_shape(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument("--out", required=True, help="output .rknn path")
    parser.add_argument("--dataset", required=True, help="calibration dataset.txt")
    parser.add_argument("--target", default="rk3568")
    parser.add_argument("--fp16", action="store_true",
                        help="build without quantization (FP16/FP32 isolation test)")
    parser.add_argument("--quant-dtype", default=None,
                        choices=[
                            "asymmetric_quantized-8",
                            "asymmetric_quantized-16",
                            "dynamic_fix_point_quantized-16",
                        ],
                        help="override RKNN quantized_dtype for INT8/INT16 experiments")
    parser.add_argument("--input-size", default=None,
                        help="optional fixed input shape for dynamic-batch ONNX, e.g. 1,3,640,640")
    args = parser.parse_args()

    rknn = RKNN(verbose=True)

    # YOLO ONNX expects RGB 0-1 internally. RKNN receives uint8 0-255,
    # so mean=0, std=255 reproduces the same preprocessing.
    config_kwargs = {
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "target_platform": args.target,
    }
    if args.quant_dtype:
        config_kwargs["quantized_dtype"] = args.quant_dtype
    print("config:", rknn.config(**config_kwargs))

    load_kwargs = {"model": args.onnx}
    if args.input_size:
        load_kwargs["input_size_list"] = [parse_shape(args.input_size)]
    print("load_onnx:", rknn.load_onnx(**load_kwargs))

    if args.fp16:
        print("build (fp16, no quantization):",
              rknn.build(do_quantization=False, dataset=None))
    else:
        print("build (int8):",
              rknn.build(do_quantization=True, dataset=args.dataset))

    print("export_rknn:", rknn.export_rknn(args.out))
    print("release:", rknn.release())
    print(f"DONE: {args.out}")


if __name__ == "__main__":
    main()

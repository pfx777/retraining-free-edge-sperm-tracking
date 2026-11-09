# Training and Detection Dumps

This directory contains the scripts used to train the detector family and to
produce the frame-level detections consumed by the host-side scoring and scale
adaptation experiments.

## Files

- `train_ablation.py`: single-variable trainer for the YOLOv8n/GatedCtx family.
- `run_arch_arm2.sh`: example 30-epoch training and evaluation chain, including
  the ONNX operator audit performed before training.
- `queue_seed2024.sh`: example cluster queue file for the fourth random seed.
- `dump_dets_generic.py`: dump Ultralytics detections in MOT-normalised format.
- `infer_dump.py`: dump detections from the official YOLOv5 implementation.
- `infer_dump_spermdet.py`: dump detections from the third-party SpermDet model.
- `dump_family_mot.py`: dump detections for stock YOLO-family checkpoints.
- `res_pareto.py`: resolution and box-scale sweep over cached detections.
- `calibrated_table.py`: combine detections and tracking with the paper's
  scale-adaptation settings.
- `fasttrack.py`: lightweight tracking utilities used by the sweep scripts.

## Environment

Install the dependencies in the repository root and set `PYTHONPATH` as
described in the top-level README. The original runs used Python 3.11,
PyTorch with CUDA, Ultralytics 8.3.163, and the public VISEM-Tracking dataset.

The shell scripts preserve the original cluster layout to document how the
experiments were launched. On another machine, either place the checkout at
`/root/code` or adjust the path variables at the top of `run_arch_arm2.sh`.

## Example training command

```bash
cd /root/code
bash run_arch_arm2.sh v8n_gctx448 stock_yolov8n_gctx.yaml E26 42
```

Required outputs include:

```text
datasets/visem_official/data.yaml
yolov8n.pt
runs/detect/runs/visem_official/<run-name>/weights/last.pt
```

The official and third-party comparisons use the same class-0 annotations,
image size, detector output budget and tracking protocol. Raw data are not
redistributed with this repository.

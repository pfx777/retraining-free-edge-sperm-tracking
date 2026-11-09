# Model Artifacts

- `stock_v8n_gctx_pruned_448_int8.rknn`: final P3-only pruned INT8 board model.
- `stock_v8n_gctx_448_int8.rknn`: unpruned INT8 comparison model.
- `stock_yolov8n_448_gctx_pruned_headraw.onnx`: pruned ONNX export before RKNN conversion.
- `stock_yolov8n_448_gctx_headraw.onnx`: unpruned ONNX export before RKNN conversion.

The corresponding model definitions are in `../src/model/`. SHA-256 hashes are stored in `SHA256SUMS.txt`.

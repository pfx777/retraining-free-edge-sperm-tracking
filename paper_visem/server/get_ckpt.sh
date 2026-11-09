#!/usr/bin/env bash
# List the official checkpoint tree (names only, no blobs) and download the .pt
# files we need through the hf-mirror resolve endpoint (git-lfs is unavailable).
set -uo pipefail
WORK=/root/autodl-tmp/visem_official_job
META="$WORK/hf-meta"
CKPT="$WORK/ckpt"
mkdir -p "$CKPT"

echo "=== tree listing (names only)"
git -C "$META" ls-tree -r --name-only HEAD 2>/dev/null | grep -E "best_checkpoints/.*\.(pt|yaml)$" | head -40

echo "=== download the .pt files we care about"
for arch in yolov5n yolov5x; do
  path=$(git -C "$META" ls-tree -r --name-only HEAD 2>/dev/null | grep -E "best_checkpoints/$arch/.*\.pt$" | head -1)
  if [ -z "$path" ]; then echo "  no .pt found for $arch"; continue; fi
  url="https://hf-mirror.com/datasets/SimulaMet-HOST/VISEM-Tracking/resolve/main/$path"
  out="$CKPT/$(basename "$path" | sed "s/^/$arch\_/")"
  echo "  $path -> $(basename "$out")"
  if [ ! -s "$out" ] || [ "$(stat -c%s "$out")" -lt 1000000 ]; then
    curl -sL --max-time 900 -o "$out" "$url" && echo "    downloaded $(du -h "$out" | cut -f1)"
  else
    echo "    already have $(du -h "$out" | cut -f1)"
  fi
  file "$out" | head -1
done

echo "=== checkpoints on disk"
ls -la "$CKPT"
echo "=== disk"
df -h /root/autodl-tmp | tail -1

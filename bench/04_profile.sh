#!/usr/bin/env bash
# Nsight Systems capture per arm at past=2048 fp16, for per-kernel attribution
# (issue #28352 gap items 3-4: launch counts, concat vs prep vs flash time).
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
BENCH=../onnxruntime/test/python/transformers
RESULTS="$(pwd)/results"
mkdir -p "$RESULTS"

command -v nsys >/dev/null || { echo "nsys not found; install Nsight Systems CLI"; exit 1; }

export ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1
for arm in gqa_xqa gqa_flash gqa_cudnn attn_past attn_scatter; do
  out="$RESULTS/nsys_${arm}_fp16_p2048"
  (cd "$BENCH" && nsys profile -o "$out" --export=sqlite --force-overwrite true \
      python benchmark_onnx_attention_vs_gqa.py \
      --profile --arms "$arm" --past-seq-len 2048 --dtype float16 --profile-iters 100) \
    2>&1 | tee "$RESULTS/profile_${arm}.log"
  # parse_kernels.py (this dir) instead of the in-tree parse_nsys.py: the
  # latter's NVTX filter only sees registered strings and misses
  # torch.cuda.nvtx ranges; parse_kernels.py aggregates the whole capture and
  # divides out the warmup instead.
  python parse_kernels.py "${out}.sqlite" \
    | tee "$RESULTS/kernels_${arm}_fp16_p2048.txt"
done

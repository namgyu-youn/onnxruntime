#!/usr/bin/env bash
# Nsight Systems capture per arm at past=2048 fp16, for per-kernel attribution
# (PR #29715 A/B: cuDNN fused-attention kernel vs flash_fwd_splitkv, launch counts).
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
BENCH=../onnxruntime/test/python/transformers
RESULTS="$(pwd)/results"
mkdir -p "$RESULTS"

command -v nsys >/dev/null || { echo "nsys not found; install Nsight Systems CLI"; exit 1; }

export ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1
for arm in gqa_xqa gqa_flash gqa_cudnn attn_past attn_scatter attn_scatter_cudnn; do
  out="$RESULTS/nsys_${arm}_fp16_p2048"
  (cd "$BENCH" && nsys profile -o "$out" --export=sqlite --force-overwrite true \
      python benchmark_onnx_attention_vs_gqa.py \
      --profile --arms "$arm" --past-seq-len 2048 --dtype float16 --profile-iters 100) \
    2>&1 | tee "$RESULTS/profile_${arm}.log"
  # parse_kernels.py, not the in-tree parse_nsys.py: its NVTX filter cannot
  # see torch.cuda.nvtx ranges.
  python parse_kernels.py "${out}.sqlite" \
    | tee "$RESULTS/kernels_${arm}_fp16_p2048.txt"
done

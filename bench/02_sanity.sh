#!/usr/bin/env bash
# Cross-arm output parity + GQA backend confirmation. Run before sweeping.
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
BENCH=../onnxruntime/test/python/transformers
mkdir -p results

export ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1
for dtype in float16 bfloat16; do
  (cd "$BENCH" && python benchmark_onnx_attention_vs_gqa.py --sanity --dtype "$dtype") \
    2>&1 | tee "results/sanity_${dtype}.log"
done

echo
echo "Check both logs: all parity lines PASS; 'use_xqa' printed for gqa_xqa only."

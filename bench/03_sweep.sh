#!/usr/bin/env bash
# Decode-latency sweep, all arms, fp16 + bf16, plus scatter-cost attribution run.
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
BENCH=../onnxruntime/test/python/transformers
RESULTS="$(pwd)/results"
mkdir -p "$RESULTS"

for dtype in float16 bfloat16; do
  (cd "$BENCH" && python benchmark_onnx_attention_vs_gqa.py \
      --dtype "$dtype" --csv "$RESULTS/sweep.csv") \
    2>&1 | tee "$RESULTS/sweep_${dtype}.log"
done

# Attribution: external-cache arm without the TensorScatter nodes; the delta
# vs the attn_scatter rows in sweep.csv is the per-step scatter/append cost.
for dtype in float16 bfloat16; do
  (cd "$BENCH" && python benchmark_onnx_attention_vs_gqa.py \
      --dtype "$dtype" --arms attn_scatter --attention-only \
      --csv "$RESULTS/sweep_attn_only.csv") \
    2>&1 | tee "$RESULTS/sweep_attn_only_${dtype}.log"
done

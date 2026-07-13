#!/usr/bin/env bash
# Environment setup for the #28352 attention benchmark (run on the CUDA device).
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Torch must be the cu130 line: nightly onnxruntime-gpu links against CUDA 13
# (libcudart.so.13), provided in-process by the cu130 wheel's nvidia-*-cu13
# deps. With cu128 torch, onnxruntime fails to import.
pip install torch triton --index-url https://download.pytorch.org/whl/cu130
# onnx pinned: gqa_test_helper.py stamps models with onnx's default opset, and
# onnx 1.22 defaults to opset 27 which ORT (<=26) rejects at session load.
pip install 'onnx==1.21.*' numpy packaging

# Nightly onnxruntime-gpu (records its commit in every benchmark log).
# If you have a proven local wheel for this GPU, `pip install /path/to/wheel` instead.
pip install --pre onnxruntime-gpu \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ \
    --extra-index-url https://pypi.org/simple

mkdir -p results
{
  date --iso-8601=seconds
  nvidia-smi
  python -c "import torch, onnxruntime as ort; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('ort', ort.__version__); print(ort.get_build_info())"
} | tee results/env.log

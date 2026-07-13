#!/usr/bin/env bash
# Environment setup for the #28352 attention benchmark (run on the CUDA device).
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Torch with CUDA (cu128 covers Ada/Hopper/Blackwell; adjust if your driver needs another line)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install onnx numpy packaging
pip install triton || echo "triton unavailable; benchmark falls back to CUDA-event timing (fine)"

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

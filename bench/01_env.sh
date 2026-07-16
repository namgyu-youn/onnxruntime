#!/usr/bin/env bash
# Environment setup for the #29715 A/B (run on the CUDA device).
# Builds an onnxruntime-gpu wheel from this branch — the PR is in no nightly.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# torch provides tensors/streams only; match the index to the box's CUDA major.
pip install torch triton --index-url https://download.pytorch.org/whl/cu128
# onnx pinned: gqa_test_helper.py stamps models with onnx's default opset, and
# onnx 1.22 defaults to opset 27 which ORT (<=26) rejects at session load.
pip install 'onnx==1.21.*' numpy packaging wheel setuptools psutil cmake ninja

# Wheel from this branch; =native restricts the build to this GPU (~30-60 min).
# Override CUDA_HOME / CUDNN_HOME if the box's install is elsewhere.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDNN_HOME="${CUDNN_HOME:-/usr}"
ROOT_FLAG=$([ "$(id -u)" = 0 ] && echo --allow_running_as_root || true)
(cd .. && ./build.sh --config Release --build_dir build/cuda_29715 --parallel $ROOT_FLAG \
    --use_cuda --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
    --build_wheel --skip_tests --compile_no_warning_as_error \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=native onnxruntime_BUILD_UNIT_TESTS=OFF)
pip install --force-reinstall ../build/cuda_29715/Release/dist/onnxruntime_gpu-*.whl

mkdir -p results
{
  date --iso-8601=seconds
  nvidia-smi
  ldconfig -p | grep -i cudnn || echo "WARNING: libcudnn not in ldconfig"
  python -c "import torch, onnxruntime as ort; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('ort', ort.__version__); print(ort.get_build_info())"
} | tee results/env.log

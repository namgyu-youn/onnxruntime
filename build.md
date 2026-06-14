# benchmark_gqa.py — local setup

## Option A: Pre-built wheel (no local C++ changes)

```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install matplotlib pandas
uv pip install onnxruntime-gpu onnx numpy
```

```bash
PYTHONPATH=onnxruntime/test/python/transformers \
  python onnxruntime/test/python/transformers/benchmark_gqa.py
```

> Does NOT reflect local `.cc`/`.cuh` changes.

---

## Option B: Full build (required to verify XQA changes)

### 1. Setup (one-time, ~1 min)

```bash
uv venv .venv && source .venv/bin/activate
uv pip install numpy packaging wheel setuptools
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install matplotlib pandas onnx
```

### 2. Install CUDA toolkit (one-time)

```bash
# Add NVIDIA repo and install CUDA 12.8 nvcc + headers
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb && apt-get update -q
apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-crt-12-8 cuda-cudart-dev-12-8
# cuda_home will be /usr/local/cuda-12.8
```

> **Note:** Do NOT use `cuda_home=/usr` — that has CUDA 12.0 (too old; abseil-cpp
> lts_20250814 requires CUDA ≥ 12.1 for `cudaLibraryLoadData` at runtime).

> GPU is A100 (SM 80). Use `CMAKE_CUDA_ARCHITECTURES=80`.
> cuDNN is provided via the Python venv (`nvidia-cudnn-cu12`).

### 3. Initial C++ build (one-time, ~25-30 min)

```bash
python tools/ci_build/build.py \
  --allow_running_as_root \
  --build_dir build \
  --config Release \
  --use_cuda \
  --cuda_home /usr/local/cuda-12.8 \
  --cudnn_home /workspace/onnxruntime/.venv/lib/python3.12/site-packages/nvidia/cudnn \
  --skip_tests \
  --skip_submodule_sync \
  --parallel 16 \
  --compile_no_warning_as_error \
  --targets onnxruntime_pybind11_state \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=80 \
    onnxruntime_BUILD_UNIT_TESTS=OFF \
    onnxruntime_ENABLE_PYTHON=ON \
    CMAKE_CXX_STANDARD=20 \
    CMAKE_CUDA_STANDARD=20
```

### 4. Run benchmark

The system's default `libcudart.so.12` may point to a stale CUDA 12.0 installation.
Use `LD_PRELOAD` to force the 12.8 runtime:

```bash
LD_PRELOAD=/usr/local/cuda-12.8/lib64/libcudart.so.12 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/workspace/onnxruntime/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib \
PYTHONPATH=build/Release:onnxruntime/test/python/transformers \
  python onnxruntime/test/python/transformers/benchmark_gqa.py
```

To verify XQA is active, compare against the baseline with `ORT_ENABLE_XQA=0`:

```bash
LD_PRELOAD=/usr/local/cuda-12.8/lib64/libcudart.so.12 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/workspace/onnxruntime/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib \
PYTHONPATH=build/Release:onnxruntime/test/python/transformers \
  ORT_ENABLE_XQA=0 \
  python onnxruntime/test/python/transformers/benchmark_gqa.py
```

With XQA enabled, token-generation latency (`past_sequence_length` table) should be
~30–57% faster at context lengths ≥ 256.

### 5. Incremental rebuild after C++ changes (~1-3 min)

```bash
cmake --build build/Release --target onnxruntime_pybind11_state -- -j16
```

Then re-run step 4.

---

## abseil-cpp patch note

`cmake/patches/abseil/absl_cuda_warnings.patch` has been extended to fix a C++23
template alias (`IfRRef::AddPtr`) in `absl/container/internal/common.h` that nvcc
rejects in C++20 mode. The fix wraps the alias in `std::enable_if_t<true, ...>` to
make it a dependent type lookup. This is required for abseil-cpp lts_20250814.

# Benchmark runbook — issue #28352: ONNX Attention (CUDA) vs contrib GQA decode

Reproducible benchmark recipe for the `~34–111 µs/token` decode-latency claim in
[microsoft/onnxruntime#28352](https://github.com/microsoft/onnxruntime/issues/28352).
Run everything in this directory **on the CUDA device** (SM80+ required; the
XQA and Flash decode paths do not exist below Ampere).

The benchmark itself lives in-tree (where the issue asks for it to be committed):
`onnxruntime/test/python/transformers/benchmark_onnx_attention_vs_gqa.py`.
See its module docstring for the arms and the measurement rules.

## 0. Get the branch and environment

```bash
git fetch <this-remote> explore/28352-bench && git checkout explore/28352-bench
cd bench
./01_env.sh          # creates ./venv, installs torch/onnx/nightly onnxruntime-gpu
source venv/bin/activate
```

A local onnxruntime-gpu wheel can replace the nightly — every log records
`onnxruntime.get_build_info()`, so the wheel commit is captured either way.
**Do not mix wheels between steps.**

Optional: `pip install triton` — the script then times with
`triton.testing.do_bench` (same timer as the in-tree `benchmark_gqa.py`);
otherwise it falls back to a CUDA-event loop. Either is fine; the timer used is
recorded in the provenance header and must be identical across arms (it is,
per run).

## 1. Sanity — before trusting any number

```bash
./02_sanity.sh
```

This runs one decode step per arm on **identical inputs** and checks the
outputs agree, with `ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1` so the GQA op
prints its resolved backend. Check in `results/sanity_*.log`:

- `Operator=GroupQueryAttention ... use_xqa` appears for the `gqa_xqa` arm and
  **not** for `gqa_flash` (which must show flash attention instead).
- All parity lines say `PASS`. A FAIL means the arms are not computing the same
  attention — stop and investigate before sweeping.

ONNX Attention has no debug print; its backend is confirmed in step 3 by
kernel names (expect `flash_fwd_splitkv_kernel` / `flash_fwd_kernel`, and no
`fmha_cutlassF` = MEA, no `UnfusedAttention` GEMM+softmax kernels). The sweep
runs ONNX arms with `ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION=1` so a silent
Flash→MEA flip cannot happen; ineligibility would surface as the (very slow,
obvious) unfused kernel.

## 2. Sweep

```bash
./03_sweep.sh
```

Sweeps past KV length 128→4096 (Llama-3-8B decode shape: B=1, H=32, H_kv=8,
head=128, KV buffer 8192) for fp16 and bf16, all arms, plus an
`--attention-only` attribution run of the external-cache arm (TensorScatter
nodes removed, so scatter cost = difference between the two runs). Results:
`results/sweep.csv`, `results/sweep_attn_only.csv`, and markdown tables in the
logs. First iterations (allocation + Flash autotune) are excluded by the
timer's warmup.

## 3. Profile (per-kernel attribution)

```bash
./04_profile.sh
```

One `nsys` capture per arm at past=2048 fp16, exported to sqlite, then a
per-kernel table via `parse_kernels.py` (in this directory — the in-tree
`parse_nsys.py` NVTX filter cannot see torch.cuda.nvtx ranges, which are
unregistered strings):

```bash
python parse_kernels.py results/nsys_<arm>_fp16_p2048.sqlite
```

It reports launches/step, avg µs, and µs/step per kernel (issue gap items 3–4).

## 4. Package results

```bash
./05_collect.sh   # tars results/ -> attn_vs_gqa_results_<host>_<date>.tar.gz
```

Copy the tarball back to the dev machine into `.claude_workdir/results/`.

## Knobs

- Bigger sweep (memory permitting): edit `03_sweep.sh`
  (`--sweep ... 8192 --max-seq-len 16384`).
- Different model shape: `--q-heads/--kv-heads/--head-size`.
- bf16 profile captures: add a loop over `--dtype` in `04_profile.sh`.

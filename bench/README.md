# Benchmark runbook — PR #29715: cuDNN SDPA decode tier for ONNX Attention (CUDA)

A/B of the Phase-1 cuDNN SDPA decode tier
([#29715](https://github.com/microsoft/onnxruntime/pull/29715), design
[#29714](https://github.com/microsoft/onnxruntime/issues/29714)) against the
Flash tier and contrib GQA, using our #28352/#29684 benchmark
(`onnxruntime/test/python/transformers/benchmark_onnx_attention_vs_gqa.py`;
arms in its docstring). Run on the CUDA device: SM80+, cuDNN ≥ 9.3 and not
9.10.0/9.10.1 (else the tier silently falls back and `attn_scatter_cudnn`
aborts on its routing assertion).

`attn_scatter` = tier off (Flash), `attn_scatter_cudnn` = tier on.
`ORT_ENABLE_CUDNN_FLASH_ATTENTION=0` also disables the SM90+ auto-preference,
so **one wheel from this branch serves both sides** — no separate main build.

## 0. Build

```bash
git fetch <this-remote> explore/29715-bench && git checkout explore/29715-bench
cd bench
./01_env.sh          # venv + deps + wheel built from this branch (~30-60 min)
source venv/bin/activate
```

Check the cuDNN line in `results/env.log`. Do not mix wheels between steps.

## 1. Sanity

```bash
./02_sanity.sh
```

Parity vs an fp32 torch reference, fp16 + bf16, 3-D and 4-D (`--cache-4d`)
cache layouts. The 4-D runs cover the mixed-layout path the PR review flagged
as untested — keep those logs. Every parity line must PASS;
`attn_scatter_cudnn` aborts itself on wrong routing.

## 2. Sweep

```bash
./03_sweep.sh
```

Past 128→4096 (B=1, H=32/8, D=128, buffer 8192), fp16 + bf16, all arms, plus
`--attention-only` attribution and the 4-D sweep. Headline read:
`attn_scatter_cudnn` vs `attn_scatter` (the tier's win) vs
`gqa_cudnn`/`gqa_xqa` (remaining GQA gap).

## 3. Profile (optional)

```bash
./04_profile.sh
```

nsys per arm at past=2048 fp16 → per-kernel table via `parse_kernels.py`.
`attn_scatter_cudnn` should show a cuDNN/fmha kernel instead of
`flash_fwd_splitkv_kernel`.

## 4. Package

```bash
./05_collect.sh   # tars results/ — copy to .claude_workdir/results/
```

Knobs: `--sweep`/`--max-seq-len`, `--q-heads/--kv-heads/--head-size`, dtype
loop in `04_profile.sh`. If `gqa_xqa` aborts on SM120, that's #29706 — drop
the arm and note it.

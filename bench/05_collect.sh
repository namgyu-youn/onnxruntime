#!/usr/bin/env bash
# Package results for transfer back to the dev machine (.claude_workdir/results/).
set -euo pipefail
cd "$(dirname "$0")"

nvidia-smi > results/nvidia-smi.final.log || true
tarball="attn_cudnn_29715_results_$(hostname)_$(date +%Y%m%d_%H%M).tar.gz"
tar czf "$tarball" results/
echo "Created $tarball — copy it to .claude_workdir/results/ on the dev machine."

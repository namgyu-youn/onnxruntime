#!/usr/bin/env python3
"""Per-kernel attribution for the #28352 profile captures.

parse_nsys.py's NVTX-range filter does not see torch.cuda.nvtx ranges (they are
unregistered strings, and async submission lets GPU timestamps drift past the
host-side range windows), so this aggregates over the whole capture instead:
the profiled config repeats warmup+profile_iters times, so after dropping each
kernel's first `--skip-first` calls, launches-per-step = count / profile_iters.

Usage: python parse_kernels.py capture.sqlite [--skip-first 20] [--iters 100]
"""

import argparse
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("db")
parser.add_argument("--skip-first", type=int, default=20, help="warmup calls to drop per kernel name")
parser.add_argument("--iters", type=int, default=100, help="profiled iterations (for per-step math)")
args = parser.parse_args()

conn = sqlite3.connect(args.db)
rows = conn.execute(
    """
    WITH numbered AS (
        SELECT s.value AS name, k.end - k.start AS dur,
               ROW_NUMBER() OVER (PARTITION BY s.value ORDER BY k.start) AS n
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
    )
    SELECT name, COUNT(*) AS calls, AVG(dur) AS avg_ns, SUM(dur) AS total_ns
    FROM numbered WHERE n > ? GROUP BY name ORDER BY total_ns DESC
    """,
    (args.skip_first,),
).fetchall()

print(f"| kernel | launches/step | avg us | us/step |  ({args.db}, skip_first={args.skip_first})")
print("|---|---|---|---|")
total_per_step = 0.0
for name, calls, avg_ns, total_ns in rows:
    per_step = calls / args.iters
    if per_step < 0.5:  # warmup-only kernels (cuBLAS init, autotune candidates)
        continue
    short = name.split("(")[0][:100]
    us_step = total_ns / args.iters / 1000.0
    total_per_step += us_step
    print(f"| `{short}` | {per_step:.0f} | {avg_ns / 1000.0:.2f} | {us_step:.2f} |")
print(f"\nGPU time per decode step (sum of above): {total_per_step:.2f} us")

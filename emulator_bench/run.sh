#!/usr/bin/env bash
# Reproduces the CataPro paper runs: feature cache, training (kcat, km; 3 seeds; fp32), result table.
# Usage: emulator_bench/run.sh [--dry-run] [cache|train|aggregate|all]   (train also evaluates)
# Env: PY GPUS RUNS_PER_GPU SEEDS DATA_ROOT OUT_ROOT EMULATOR_ROOT DRY_RUN
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STEP=all
for a in "$@"; do
  case $a in --dry-run) DRY_RUN=1 ;; cache | train | aggregate | all) STEP=$a ;; *) echo "bad argument: $a" >&2; exit 2 ;; esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

PY=${PY:-python}
GPUS=${GPUS:-0 1 2 3}
RUNS_PER_GPU=${RUNS_PER_GPU:-2}
SEEDS=${SEEDS:-666 777 888}
EMULATOR_ROOT=${EMULATOR_ROOT:-$(realpath ../..)}
DATA_ROOT=${DATA_ROOT:-$EMULATOR_ROOT/data/processed/datasets}
OUT_ROOT=${OUT_ROOT:-$EMULATOR_ROOT/outputs/baselines/CataPro}
B=emulator_bench
# The launcher builds the feature cache first, then trains; finished runs are skipped. fp32 (no --amp).
# shellcheck disable=SC2206
LAUNCH=($PY $B/launch_parallel_bench.py --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" --cache_dir "$OUT_ROOT/embeddings" --tasks kcat km --seeds $SEEDS --gpus $GPUS --runs_per_gpu "$RUNS_PER_GPU")

want cache && run "${LAUNCH[@]}" --features_only
want train && run "${LAUNCH[@]}"
want aggregate && run $PY $B/aggregate_catapro_results.py --out_root "$OUT_ROOT"
exit 0

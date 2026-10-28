#!/usr/bin/env bash
# Checks that the CataPro wrapper matches the original code. Usage: verify_equivalence.sh [--gpu]
# Expected env: mldb (torch, transformers, sentencepiece, rdkit, pytest). The script does not create envs.
# Tests run on CPU with tiny random T5 encoders. To use cached real weights instead:
#   CUDA_VISIBLE_DEVICES= python -m pytest tests -m real_weights
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected conda env: mldb"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
if [[ ${1:-} == --gpu ]]; then
    echo "No gpu tests: production runs fp32, the path the CPU tests compare."
fi
out=$(CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_original_equivalence.py -q -rs -p no:cacheprovider --color=no -o addopts="" -m "not real_weights" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi

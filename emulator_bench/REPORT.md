# CataPro emulator_bench report

## Method summary

CataPro predicts kcat or Km from an enzyme sequence and a substrate SMILES.
Enzyme: mean-pooled ProtT5-XL-UniRef50 embedding (1024). Substrate: MolT5 embedding (768) plus MACCS keys (167).
The 1959-d vector goes through BatchNorm on the ProtT5 and MolT5 parts, one hidden layer (256, BatchNorm, ReLU), and one output.
The loss is RMSE. The paper uses 10-fold cross-validation on 0.4-similarity clusters. kcat/Km uses a second model that is not used here.
The paper evaluates kcat and Km only. The wrapper does not run Ki.

## Original default settings

The paper (Methods, "Construction of CataPro models") only states one hidden layer and RMSE loss.
All other values come from the released code. Paper and code do not disagree.

| Setting | Value | Source |
|---|---|---|
| Optimiser | Adam, betas (0.9, 0.999), weight decay 0.01 | `training/kcat/training.py:96` (Km: `training/Km/training.py:91`) |
| Learning rate | 1e-3 | `training/kcat/training.py:66` |
| Batch size | 64 | `training/kcat/training.py:62` |
| Epochs | 150 | `training/kcat/training.py:70` |
| Dropout | 0.0 | `training/kcat/training.py:68` |
| Early stopping | patience 20, min delta 0.001, on val loss | `training/kcat/training.py:85-86` |
| Checkpoint | best val loss | `training/kcat/training.py:111-115` |
| Loss | RMSE (sqrt(MSE + 1e-6)) | `training/kcat/utils.py` |
| Architecture | 1959 -> 256 -> 1, BatchNorm | `inference/model.py`, paper Fig. 1b |
| Protein length | over 1000 residues: first 500 + last 500 | `inference/utils.py` (Seq_to_vec) |
| Singleton train batch | skipped (Km only) | `training/Km/utils.py:run_a_training_epoch` |

## What the wrapper does now

For one task (kcat or km) it finds every default split folder on disk:
`random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`,
`enzyme_sequence_splits/threshold_*`, `enzyme_structure_splits/threshold_*`, `substrate_splits/threshold_*`, `conformer_cosine_splits/threshold_*`.
It builds features (cached), trains with the original settings for seeds 666, 777, 888, and writes val and test metrics.
Thresholds are discovered from disk. Output goes to `<EMULaToR>/outputs/baselines/CataPro` (`--out_root`). The dataset tree is read only.
`aggregate_catapro_results.py` gives mean and std over seeds.

## Files

Removed: `tune_optuna.py`, `default_hparams.json` (held tuned values, Optuna-style keys, and a stale split list), `predict_single_target.py` (not part of evaluation).
Changed: `run_split_benchmarks.py`, `launch_parallel_bench.py`, `aggregate_catapro_results.py`, `train_single_target_tvt.py`, `utils.py`, `build_tvt_features.py`, `feature_utils.py`, `README.md`, `commands.txt`.
Unchanged: `model.py`.
No original (non-emulator_bench) code was edited.

## Bugs fixed

- Default batch size was 2048 in the trainer and runner. It is now the original 64.
- `no_early_stopping: true` in the default json disabled early stopping. Early stopping is now always on, as in the original.
- Default paths pointed to `data/processed/baselines/CataPro`, which does not exist. Splits are now read from `data/processed/datasets/kinetic_dataset/<task>`.
- Features and results were written inside the split folders (under `data`). They now go to `--out_root`.
- Default split groups left out `uniprot_time_splits` in the runner. The list now matches the default splits.
- Seed default was 42. It is now 666 777 888.
- `batch_encode_plus` no longer exists in transformers 5.x (mldb has 5.19). The tokenizer is now called directly. The result is the same.
- `time_running.dat` wrote a literal `\n`. Fixed.
- Relative `emulator_bench/...` paths for sub-scripts and the cache broke when the working directory was not the repo root. Paths are now based on the script folder.
- `aggregate_catapro_results.py` used old `CataPro-Kcat/CataPro-Km` folders. It now reads `--out_root`.
- `launch_parallel_bench.py` duplicated the discovery and summary code of the runner. It now imports from `run_split_benchmarks.py`.

## Deviations from the original protocol

- The singleton-batch skip (from the original Km script) is always on. BatchNorm cannot train on one row, so the original kcat script would crash in that case. Otherwise it changes nothing.
- Train, val, and test come from explicit files, not 10-fold CV. The best epoch is chosen on the val file, not on the test fold.
- Features are computed in batches with padding and an attention mask, and cached per unique sequence and SMILES. The original embeds one item at a time. Values can differ at floating-point level only.
- Features for all unique items are computed once and shared across seeds and splits through the cache. Models are unchanged.
- `--epochs` (default 150) exists for smoke tests only.

## Commands

See `commands.txt`. Env: `/home/adhil/.conda/envs/mldb/bin/python`.

```
cd /home/adhil/github/EMULaToR/baselines/CataPro
python emulator_bench/launch_parallel_bench.py --tasks kcat km --gpus 0 1 2 3 --runs_per_gpu 2
python emulator_bench/aggregate_catapro_results.py
```

Sequential version for one task: `CUDA_VISIBLE_DEVICES=0 python emulator_bench/run_split_benchmarks.py --task km`.
Run count: 2 tasks x 5 split folders x 3 seeds = 30 runs (split folders counted per group; threshold groups add more).

## Smoke test

Data: Km, 64 train / 32 val / 32 test rows (sequences under 200 residues) for `random_splits_grouped_sequence` and `enzyme_sequence_splits/threshold_0.05`, in a temporary data root.
Run: `run_split_benchmarks.py` with seeds 666 777, 1 epoch, GPU 1. Then `launch_parallel_bench.py` with seed 888, 1 epoch.
Result: features built (1960 columns), all runs wrote `final_results_{val,test}.csv`, and the aggregator wrote mean and std.
Output: `/tmp/claude-1005/-home-adhil-github/92fb41ec-b482-467d-940d-08ea9f435509/scratchpad/bench_smoke/CataPro`.
Training was not run to completion. ProtT5 and MolT5 weights were downloaded to `~/.cache/huggingface` (about 10 GB).

## Changes after review

- Ki task removed from the runners, commands and README.
- `--out_root` defaults to `<EMULaToR>/outputs/baselines/CataPro` (also in the aggregator).
- Feature pickles are written to a temp file and moved with `os.replace`, so a crash leaves no partial file.
- Re-smoke (Km, 2 split folders, seed 666, 1 epoch, GPU 1): features, training and aggregation ran. `--task ki` is now rejected. Output: `bench_smoke/CataPro/out2`.

## Open issues

- kcat/Km (the second CataPro model) is not evaluated.
- Missing axes: none for kcat. Km split folders are found on disk only.

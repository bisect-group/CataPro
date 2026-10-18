# CataPro emulator_bench

This folder evaluates CataPro with its original settings on the default EMULaToR splits.
It trains one model per task, split, and seed. It does not tune hyperparameters.

Tasks: `kcat`, `km`. Both live in `data/processed/datasets/kinetic_dataset/<task>`.
The CataPro paper evaluates kcat and Km only.

Splits (found on disk, missing folders are skipped):
`random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`,
`enzyme_sequence_splits/threshold_*`, `substrate_splits/threshold_*`.

Seeds: 666, 777, 888.

## Files

- `model.py`: CataPro predictor (BatchNorm on ProtT5 and MolT5 parts, 1959 -> 256 -> 1).
- `utils.py`: metrics, RMSE loss, early stopping, epoch loops (from the original `training/` code).
- `feature_utils.py`: ProtT5, MolT5, and MACCS features with a persistent cache.
- `build_tvt_features.py`: builds one feature pickle from a split parquet file.
- `train_single_target_tvt.py`: trains on train, selects the best epoch on val, tests on test.
- `run_split_benchmarks.py`: runs all splits and seeds for one task, one after another.
- `launch_parallel_bench.py`: same runs, spread over several GPUs.
- `aggregate_catapro_results.py`: mean and std over seeds.

## Original settings

Adam (lr 1e-3, weight decay 0.01), batch size 64, up to 150 epochs, dropout 0,
RMSE loss, early stopping on val loss (patience 20, min delta 0.001).
These are the defaults of `train_single_target_tvt.py`. See `REPORT.md` for sources.

## Outputs

Everything is written to `--out_root` (default `<EMULaToR>/outputs/baselines/CataPro`). Nothing is written to the dataset tree.

```
<out_root>/<task>/<split_group>[/<threshold_dir>]/features/{train,val,test}_feats.pkl
<out_root>/<task>/<split_group>[/<threshold_dir>]/seed_<seed>/final_results_{val,test}.csv
```

## Run

See `commands.txt`. The ProtT5 and MolT5 weights download from Hugging Face on first use.
The embedding cache is `emulator_bench/.cache_embeddings` (`--cache_dir`).

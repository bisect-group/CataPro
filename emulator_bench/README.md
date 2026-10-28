# CataPro emulator_bench

This folder evaluates CataPro with its original settings on the default EMULaToR splits.
It trains one model per task, split, and seed. It does not tune hyperparameters.

Tasks: `kcat`, `km`. Both live in `data/processed/datasets/kinetic_dataset/<task>`.
The CataPro paper evaluates kcat and Km only.

Splits (found on disk, missing folders are skipped):
`random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`,
`enzyme_sequence_splits/threshold_*`, `enzyme_structure_splits/threshold_*`, `substrate_splits/threshold_*`, `conformer_cosine_splits/threshold_*`.

Seeds: 666, 777, 888.

## Files

- `model.py`: CataPro predictor (BatchNorm on ProtT5 and MolT5 parts, 1959 -> 256 -> 1).
- `utils.py`: metrics, RMSE loss, early stopping, epoch loops (from the original `training/` code).
- `feature_utils.py`: ProtT5, MolT5, and MACCS features with a persistent cache.
- `build_tvt_features.py`: builds one feature pickle from a split parquet file.
- `train_single_target_tvt.py`: trains on train, selects the best epoch on val, tests on test.
- `run_split_benchmarks.py`: runs all splits and seeds for one task, one after another.
- `launch_parallel_bench.py`: same runs, spread over several GPUs (`--gpus 0 1 --runs_per_gpu 2`). Both launchers take `--amp` (bf16, or fp16 on pre-Ampere GPUs). Default is the original fp32.
- `aggregate_catapro_results.py`: mean and std over seeds.

## Original settings

Adam (lr 1e-3, weight decay 0.01), batch size 64, up to 150 epochs, dropout 0,
RMSE loss, early stopping on val loss (patience 20, min delta 0.001).
These are the defaults of `train_single_target_tvt.py`. See `REPORT.md` for sources.

## Outputs

Everything is written to `--out_root` (default `<EMULaToR>/outputs/baselines/CataPro`). Nothing is written to the dataset tree.

```
<cache_dir>/{prot_t5,mol_t5,maccs}_v2/ (feature cache, see Run)
<out_root>/<task>/<split_group>/<split_name>/features/{train,val,test}_feats.pkl
<out_root>/<task>/<split_group>/<split_name>/seed_<seed>/{bestmodel.pth,final_results_{val,test}.csv,...}
<out_root>/aggregate_catapro_results.csv
```

`<split_name>` is `threshold_*` or `all` for flat split groups.

## Verify equivalence

```bash
emulator_bench/verify_equivalence.sh
```

Run it in the `mldb` env. It runs on CPU, needs no data or download, and fails if a test is skipped.
Production precision is fp32: `--amp` is off by default and was slower, so it is not used. The run folders hold
no precision record. The tests run fp32 on CPU, so the script has no gpu tests and `--gpu` only prints this.

`tests/test_original_equivalence.py` compares the wrapper with the original CataPro code:

- Model output, RMSE loss, gradients and batch-norm statistics for the four original model classes, in fp64,
  with and without dropout (rtol 1e-12, gradients 1e-10).
- Batch order of the wrapper against `DataLoader(shuffle=True)`, including the last partial batch.
- Two training epochs (loader, loss, Adam) against the original kcat loop and the Km loop (rtol 1e-4).
- Batched, padded ProtT5 and MolT5 features against the original per-item features (rtol 1e-4, atol 1e-5),
  using tiny randomly initialised T5 encoders and the tokenizer in `tests/fixtures/tiny_t5_tokenizer`.
- MACCS keys, exact.

The same two feature tests also run with the real cached ProtT5 and MolT5 weights:
`CUDA_VISIBLE_DEVICES= python -m pytest tests -m real_weights`. They skip without the Hugging Face cache and
are not part of the script.

## Run

See `commands.txt`. Run the commands from the `baselines/CataPro` folder. The ProtT5 and MolT5 weights download from Hugging Face on first use.
The embedding cache is `<EMULaToR>/outputs/baselines/CataPro/embeddings` (`--cache_dir`). Its keys cover the model id, dtype (float32) and the ProtT5 length cap.
The Hugging Face model revision is not pinned. Change `CACHE_VERSION` in `feature_utils.py` to force a rebuild.

"""Sequential multi-seed CataPro evaluation on the default EMULaToR splits.

Training uses the original CataPro settings (see train_single_target_tvt.py).
Inputs are read from the dataset tree. All outputs go under --out_root:
  <cache_dir>/ (default <EMULaToR>/outputs/baselines/CataPro/embeddings)
  <out_root>/<task>/<split_group>/<split_name>/features/{train,val,test}_feats.pkl
  <out_root>/<task>/<split_group>/<split_name>/seed_<seed>/
split_name is `all` for flat split groups.
"""

import argparse
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
EMULATOR_ROOT = BENCH_DIR.parents[2]
DEFAULT_DATA_ROOT = EMULATOR_ROOT / "data" / "processed" / "datasets"
DEFAULT_OUT_ROOT = EMULATOR_ROOT / "outputs" / "baselines" / "CataPro"
TASK_PARENT = {"kcat": "kinetic_dataset", "km": "kinetic_dataset"}
SPLIT_GROUPS = [
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "substrate_splits",
]
SEEDS = [666, 777, 888]


def add_common_args(parser):
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT), help="Folder holding kinetic_dataset/.")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--split_groups", nargs="+", default=SPLIT_GROUPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--cache_dir", default=str(DEFAULT_OUT_ROOT / "embeddings"))
    parser.add_argument("--prot_batch_size", type=int, default=32)
    parser.add_argument("--mol_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150, help="Original value is 150. Lower it for smoke tests only.")
    parser.add_argument("--amp", action="store_true", help="Mixed precision (bf16, or fp16 on pre-Ampere GPUs). Default is fp32.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")


def discover_split_jobs(task, args):
    """Return (name, split_dir, run_root) for every split folder found on disk."""
    task_root = Path(args.data_root) / TASK_PARENT[task] / task
    jobs = []
    for group in args.split_groups:
        group_dir = task_root / group
        if (group_dir / "train.parquet").exists():
            dirs = [(f"{group}/all", group_dir)]
        else:
            dirs = [(f"{group}/{d.name}", d) for d in sorted(group_dir.glob("threshold_*")) if d.is_dir()]
        jobs += [(name, d, Path(args.out_root) / task / name) for name, d in dirs]
    return jobs


def build_features(split_dir, run_root, args):
    feats_dir = run_root / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)
    for part in ("train", "val", "test"):
        out_pkl = feats_dir / f"{part}_feats.pkl"
        if out_pkl.exists() and not args.overwrite:
            continue
        subprocess.run(
            [
                sys.executable, str(BENCH_DIR / "build_tvt_features.py"),
                "--input_csv", str(split_dir / f"{part}.parquet"),
                "--output_pkl", str(out_pkl),
                "--target_col", "log10_value",
                "--prot_batch_size", str(args.prot_batch_size),
                "--mol_batch_size", str(args.mol_batch_size),
                "--cache_dir", args.cache_dir,
            ],
            check=True,
        )


def train_command(run_root, name, task, seed, device, args):
    feats_dir = run_root / "features"
    cmd = [
        sys.executable, str(BENCH_DIR / "train_single_target_tvt.py"),
        "--train_pkl", str(feats_dir / "train_feats.pkl"),
        "--val_pkl", str(feats_dir / "val_feats.pkl"),
        "--test_pkl", str(feats_dir / "test_feats.pkl"),
        "--out_dir", str(run_root / f"seed_{seed}"),
        "--task_name", f"{task}_{name}_seed{seed}",
        "--epochs", str(args.epochs),
        "--seed", str(seed),
        "--device", device,
    ]
    return cmd + ["--amp"] if args.amp else cmd


def is_done(run_root, seed):
    return (run_root / f"seed_{seed}" / "final_results_test.csv").exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(TASK_PARENT))
    parser.add_argument("--device", default="cuda:0")
    add_common_args(parser)
    args = parser.parse_args()

    jobs = discover_split_jobs(args.task, args)
    if not jobs:
        raise SystemExit(f"No split folders found for {args.task} under {args.data_root}")
    for name, split_dir, run_root in jobs:
        print(f"[{args.task}] {name} seeds={args.seeds}", flush=True)
        if args.dry_run:
            continue
        build_features(split_dir, run_root, args)
        for seed in args.seeds:
            if args.overwrite or not is_done(run_root, seed):
                subprocess.run(train_command(run_root, name, args.task, seed, args.device, args), check=True)


if __name__ == "__main__":
    main()

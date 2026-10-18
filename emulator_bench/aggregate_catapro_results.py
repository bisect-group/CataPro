"""Aggregate CataPro results over seeds.

Reads <out_root>/<task>/<split_group>[/<threshold_dir>]/seed_<seed>/final_results_{val,test}.csv
and writes <out_root>/aggregate_catapro_results.csv with mean and std per metric.
"""

import argparse
from pathlib import Path

import pandas as pd

METRICS = ["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"]
BENCH_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_root", default=str(BENCH_DIR.parents[2] / "outputs" / "baselines" / "CataPro"))
    args = parser.parse_args()
    out_root = Path(args.out_root)

    rows = []
    for test_csv in sorted(out_root.glob("**/seed_*/final_results_test.csv")):
        seed_dir = test_csv.parent
        for part in ("val", "test"):
            metrics = pd.read_csv(seed_dir / f"final_results_{part}.csv").iloc[0]
            task, *split = seed_dir.relative_to(out_root).parts[:-1]
            rows.append({"task": task, "split": "/".join(split), "part": part,
                         "seed": int(seed_dir.name.split("_")[1]), **metrics[METRICS]})
    if not rows:
        raise SystemExit(f"No finished runs under {out_root}")

    grouped = pd.DataFrame(rows).groupby(["task", "split", "part"])
    agg = grouped[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = grouped.size()
    out_path = out_root / "aggregate_catapro_results.csv"
    agg.reset_index().to_csv(out_path, index=False)
    print(f"Saved {len(agg)} rows to {out_path}")


if __name__ == "__main__":
    main()

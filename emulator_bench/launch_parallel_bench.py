"""Multi-GPU CataPro evaluation: build features one split at a time, then train all (split, seed) runs in parallel.

Example:
    python launch_parallel_bench.py --tasks kcat km --gpus 0 1 --runs_per_gpu 2
"""

import argparse
import os
import queue
import subprocess
import threading

from run_split_benchmarks import (
    TASK_PARENT,
    add_common_args,
    build_features,
    discover_split_jobs,
    is_done,
    train_command,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", required=True, choices=sorted(TASK_PARENT))
    parser.add_argument("--gpus", nargs="+", required=True, help="GPU ids, for example 0 1 2 3.")
    parser.add_argument("--runs_per_gpu", type=int, default=1)
    add_common_args(parser)
    args = parser.parse_args()

    experiments = []
    for task in args.tasks:
        for name, split_dir, run_root in discover_split_jobs(task, args):
            for seed in args.seeds:
                if args.overwrite or not is_done(run_root, seed):
                    experiments.append((task, name, split_dir, run_root, seed))
    print(f"{len(experiments)} runs to train")
    if args.dry_run:
        for task, name, _, _, seed in experiments:
            print(f"[{task}] {name} seed={seed}")
        return

    # One embedding cache is shared, so features are built one split at a time.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
    for split_dir, run_root in dict.fromkeys((d, r) for _, _, d, r, _ in experiments):
        build_features(split_dir, run_root, args)
    if args.features_only:
        return

    work = queue.Queue()
    for exp in experiments:
        work.put(exp)
    failed = []

    def worker(gpu):
        gpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        while True:
            try:
                task, name, _, run_root, seed = work.get_nowait()
            except queue.Empty:
                return
            cmd = train_command(run_root, name, task, seed, "cuda:0", args)
            if subprocess.run(cmd, env=gpu_env).returncode:
                failed.append(f"[{task}] {name} seed={seed}")

    threads = [
        threading.Thread(target=worker, args=(str(g),))
        for g in args.gpus
        for _ in range(args.runs_per_gpu)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        raise SystemExit("Failed runs:\n" + "\n".join(failed))


if __name__ == "__main__":
    main()

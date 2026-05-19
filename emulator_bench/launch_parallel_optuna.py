import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TUNE_SCRIPT = REPO_ROOT / "emulator_bench" / "tune_optuna.py"


def split_trials(total, workers):
    base = total // workers
    rem = total % workers
    return [base + (1 if idx < rem else 0) for idx in range(workers)]


def worker_cmd(args, trials, worker_index):
    cmd = [
        sys.executable,
        str(TUNE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--manifests_dir",
        args.manifests_dir,
        "--features_dir",
        args.features_dir,
        "--cache_dir",
        args.cache_dir,
        "--output_dir",
        args.output_dir,
        "--dict_dir",
        args.dict_dir,
        "--source_target_col",
        args.source_target_col,
        "--target_transform",
        args.target_transform,
        "--device",
        "cuda:0",
        "--iteration",
        str(args.iteration),
        "--metric",
        args.metric,
        "--eval_split",
        args.eval_split,
        "--n_trials",
        str(trials),
        "--sampler_seed",
        str(args.sampler_seed + worker_index),
        "--study_name",
        args.study_name,
        "--storage",
        args.storage,
        "--skip_prepare",
        "--skip_cache",
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.seeds:
        cmd.extend(["--seeds", *[str(seed) for seed in args.seeds]])
    if args.max_jobs:
        cmd.extend(["--max_jobs", str(args.max_jobs)])
    if args.overwrite_runs:
        cmd.append("--overwrite_runs")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch parallel single-GPU Optuna workers for DeepEnzyme.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--trials_per_gpu", type=int, default=1)
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument("--base_dir", default="/home/adhil/github/EMULaToR/data/processed/baselines/DeepEnzyme")
    parser.add_argument("--manifests_dir", default="/home/adhil/github/EMULaToR/data/processed/baselines/DeepEnzyme/deepenzyme_manifests")
    parser.add_argument("--features_dir", default="/home/adhil/github/EMULaToR/data/processed/baselines/DeepEnzyme/deepenzyme_features")
    parser.add_argument("--cache_dir", default="/home/adhil/github/EMULaToR/data/processed/baselines/DeepEnzyme/embeddings")
    parser.add_argument("--output_dir", default="/home/adhil/github/EMULaToR/data/processed/baselines/DeepEnzyme/deepenzyme_optuna_runs")
    parser.add_argument("--dict_dir", default="Data/Input")
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--source_target_col", default="log10_value")
    parser.add_argument("--target_transform", choices=["none", "log2_from_value"], default="none")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--max_jobs", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=80)
    parser.add_argument("--metric", choices=["MAE", "RMSE", "R2", "PCC"], default="RMSE")
    parser.add_argument("--eval_split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", default="deepenzyme_optuna")
    parser.add_argument("--storage", required=True)
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--stagger_seconds", type=float, default=3.0)
    args = parser.parse_args()

    slots = [(gpu, slot) for gpu in args.gpus for slot in range(args.trials_per_gpu)]
    counts = split_trials(args.n_trials, len(slots))
    procs = []
    try:
        for idx, ((gpu, slot), trials) in enumerate(zip(slots, counts)):
            if trials <= 0:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = worker_cmd(args, trials, idx)
            print(f"Launching Optuna worker {idx} on GPU {gpu} slot {slot} for {trials} trials", flush=True)
            procs.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env))
            if args.stagger_seconds > 0:
                time.sleep(args.stagger_seconds)
        failed = False
        for proc in procs:
            if proc.wait() != 0:
                failed = True
        if failed:
            raise RuntimeError("One or more Optuna workers failed.")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()


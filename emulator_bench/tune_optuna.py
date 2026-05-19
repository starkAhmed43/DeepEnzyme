import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_FEATURES_DIR, DEFAULT_MANIFESTS_DIR, DEFAULT_SPLIT_GROUPS, atomic_json, discover_split_jobs, feature_dir, normalize_threshold_args
from emulator_bench.run_split_benchmarks import maybe_cache, maybe_prepare


def metric_direction(metric: str):
    return "maximize" if metric.upper() in {"R2", "PCC"} else "minimize"


def objective_value(metric_row, metric: str):
    key = metric.upper()
    return float(metric_row[key])


def train_trial(args, job, seed, trial_number, hparams, env):
    out_dir = Path(args.output_dir) / f"trial_{trial_number}" / job["split_group"] / job["split_name"] / f"seed_{seed}"
    metric_path = out_dir / f"final_results_{args.eval_split}.csv"
    if not metric_path.exists() or args.overwrite_runs:
        cmd = [
            sys.executable,
            "emulator_bench/train_single_target_tvt.py",
            "--train_dir",
            str(feature_dir(args.features_dir, job, "train")),
            "--val_dir",
            str(feature_dir(args.features_dir, job, "val")),
            "--test_dir",
            str(feature_dir(args.features_dir, job, "test")),
            "--out_dir",
            str(out_dir),
            "--dict_dir",
            str(args.dict_dir),
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--lr",
            str(hparams["lr"]),
            "--weight_decay",
            str(hparams["weight_decay"]),
            "--dropout",
            str(hparams["dropout"]),
            "--iteration",
            str(args.iteration),
            "--dim",
            "64",
            "--layer_output",
            "3",
            "--hidden_dim1",
            "64",
            "--hidden_dim2",
            "64",
            "--nhead",
            "4",
            "--hid_size",
            "64",
            "--layers_trans",
            "3",
        ]
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    row = pd.read_csv(metric_path).iloc[0].to_dict()
    return objective_value(row, args.metric)


def main():
    parser = argparse.ArgumentParser(description="Optional Optuna tuning for DeepEnzyme retraining-safe knobs.")
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--manifests_dir", type=Path, default=DEFAULT_MANIFESTS_DIR)
    parser.add_argument("--features_dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_BASE_DIR / "embeddings")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_BASE_DIR / "deepenzyme_optuna_runs")
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--dict_dir", type=Path, default=Path("Data/Input"))
    parser.add_argument("--source_target_col", type=str, default="log10_value")
    parser.add_argument("--target_transform", choices=["none", "log2_from_value"], default="none")
    parser.add_argument("--identity_threshold", type=float, default=90.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--max_jobs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda_visible_devices", type=str, default="2")
    parser.add_argument("--iteration", type=int, default=80)
    parser.add_argument("--metric", choices=["MAE", "RMSE", "R2", "PCC"], default="RMSE")
    parser.add_argument("--eval_split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="deepenzyme_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=5e-3)
    parser.add_argument("--weight_decay_min", type=float, default=1e-8)
    parser.add_argument("--weight_decay_max", type=float, default=1e-4)
    parser.add_argument("--dropout_min", type=float, default=0.0)
    parser.add_argument("--dropout_max", type=float, default=0.5)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite_prepare", action="store_true")
    parser.add_argument("--overwrite_features", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    maybe_prepare(args)
    maybe_cache(args)
    jobs = discover_split_jobs(args.base_dir, args.split_groups, args.thresholds)
    if args.max_jobs and args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=metric_direction(args.metric),
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )

    def objective(trial):
        hparams = {
            "lr": trial.suggest_float("lr", args.lr_min, args.lr_max, log=True),
            "weight_decay": trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max, log=True),
            "dropout": trial.suggest_float("dropout", args.dropout_min, args.dropout_max),
        }
        values = []
        for job in jobs:
            for seed in args.seeds:
                values.append(train_trial(args, job, seed, trial.number, hparams, env))
        return float(sum(values) / len(values))

    study.optimize(objective, n_trials=args.n_trials)
    best_hparams = {
        "lr": study.best_params["lr"],
        "weight_decay": study.best_params["weight_decay"],
        "dropout": study.best_params["dropout"],
        "iteration": 200,
        "dim": 64,
        "layer_output": 3,
        "hidden_dim1": 64,
        "hidden_dim2": 64,
        "nhead": 4,
        "hid_size": 64,
        "layers_trans": 3,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_dir / f"{args.study_name}_best_hparams.json",
        {
            "metric": args.metric,
            "eval_split": args.eval_split,
            "best_trial_number": study.best_trial.number,
            "best_value": float(study.best_value),
            "best_hparams": best_hparams,
        },
    )
    study.trials_dataframe().to_csv(args.output_dir / f"{args.study_name}_trials.csv", index=False)


if __name__ == "__main__":
    main()

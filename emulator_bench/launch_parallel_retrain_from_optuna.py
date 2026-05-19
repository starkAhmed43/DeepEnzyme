import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_FEATURES_DIR, DEFAULT_MANIFESTS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_SPLIT_GROUPS, atomic_csv, discover_split_jobs, feature_dir, normalize_threshold_args, result_dir
from emulator_bench.run_split_benchmarks import maybe_cache, maybe_prepare


def load_hparams(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw = payload.get("best_hparams", payload)
    defaults = {
        "lr": 0.001,
        "iteration": 200,
        "weight_decay": 1e-6,
        "dropout": 0.3,
        "dim": 64,
        "layer_output": 3,
        "hidden_dim1": 64,
        "hidden_dim2": 64,
        "nhead": 4,
        "hid_size": 64,
        "layers_trans": 3,
    }
    defaults.update({key: raw[key] for key in defaults if key in raw})
    return defaults


def train_cmd(args, job, seed, hparams, gpu):
    out_dir = result_dir(args.output_root, job, seed)
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
        "cuda:0",
        "--seed",
        str(seed),
    ]
    for key, value in hparams.items():
        cmd.extend([f"--{key}", str(value)])
    if args.no_amp:
        cmd.append("--no_amp")
    if args.overwrite_runs:
        cmd.append("--overwrite")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return cmd, env, out_dir


def main():
    parser = argparse.ArgumentParser(description="Retrain DeepEnzyme split jobs in parallel from an Optuna best-hparams JSON.")
    parser.add_argument("--hparams_json", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["2"])
    parser.add_argument("--trials_per_gpu", type=int, default=1)
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--manifests_dir", type=Path, default=DEFAULT_MANIFESTS_DIR)
    parser.add_argument("--features_dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_BASE_DIR / "embeddings")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_BASE_DIR / "retrain_from_optuna")
    parser.add_argument("--dict_dir", type=Path, default=Path("Data/Input"))
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--source_target_col", type=str, default="log10_value")
    parser.add_argument("--target_transform", choices=["none", "log2_from_value"], default="none")
    parser.add_argument("--identity_threshold", type=float, default=90.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite_prepare", action="store_true")
    parser.add_argument("--overwrite_features", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_prepare(args)
    maybe_cache(args)
    hparams = load_hparams(args.hparams_json)
    jobs = discover_split_jobs(args.base_dir, args.split_groups, args.thresholds)
    tasks = [(job, seed) for job in jobs for seed in args.seeds]
    work = queue.Queue()
    for task in tasks:
        work.put(task)

    results, lock = [], threading.Lock()

    def worker(gpu, slot):
        while True:
            try:
                job, seed = work.get_nowait()
            except queue.Empty:
                return
            cmd, env, out_dir = train_cmd(args, job, seed, hparams, gpu)
            status = "completed"
            try:
                if not args.dry_run:
                    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
            except Exception as exc:
                status = f"failed: {exc}"
            with lock:
                results.append({"split_group": job["split_group"], "split_name": job["split_name"], "seed": seed, "gpu": gpu, "slot": slot, "run_dir": str(out_dir), "status": status})
            work.task_done()

    threads = []
    for gpu in args.gpus:
        for slot in range(args.trials_per_gpu):
            thread = threading.Thread(target=worker, args=(gpu, slot), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        row = dict(result)
        for split in ("train", "val", "test"):
            metrics_path = Path(result["run_dir"]) / f"final_results_{split}.csv"
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
                for key, value in metrics.items():
                    row[f"{split}_{key}"] = value
        rows.append(row)
    atomic_csv(args.output_root / "retrain_summary_runs.csv", pd.DataFrame(rows))


if __name__ == "__main__":
    main()

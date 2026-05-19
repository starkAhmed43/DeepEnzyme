import argparse
import json
import subprocess
import sys
from pathlib import Path


def _to_cli_key(k):
    return f"--{k}"


def _append_arg(cmd, key, value):
    cli_key = _to_cli_key(key)
    if isinstance(value, bool):
        if value:
            cmd.append(cli_key)
        return
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        if not value:
            return
        cmd.append(cli_key)
        cmd.extend([str(v) for v in value])
        return
    cmd.extend([cli_key, str(value)])


def main():
    parser = argparse.ArgumentParser(description="Launch DeepEnzyme emulator_bench from a JSON config.")
    parser.add_argument("--config", required=True, type=str, help="Path to benchmark JSON config.")
    parser.add_argument("--dry_run", action="store_true", help="Print the resolved command and exit.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    cmd = [sys.executable, "emulator_bench/run_split_benchmarks.py"]

    # Preserve insertion order from config for readability/reproducibility.
    for k, v in cfg.items():
        _append_arg(cmd, k, v)

    print("Resolved command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

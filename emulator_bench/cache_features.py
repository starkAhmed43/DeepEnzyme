import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
try:
    from src.utils.rich_progress import progress, write
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress, write

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_FEATURES_DIR,
    DEFAULT_MANIFESTS_DIR,
    DEFAULT_SPLIT_GROUPS,
    atomic_json,
    discover_split_jobs,
    feature_dir,
    manifest_path,
    normalize_threshold_args,
    read_table,
    require_columns,
)
from emulator_bench.feature_utils import (
    FeatureCache,
    atomic_save_object_array,
    create_adjacency,
    create_atoms,
    create_ijbonddict,
    extract_fingerprints,
    load_pickle,
    protein_contact_map,
    split_sequence,
)


REQUIRED_FEATURE_FILES = [
    "fingerprint.npy",
    "smileadjacencies.npy",
    "sequences.npy",
    "proteinadjacencies.npy",
    "logkcat.npy",
    "metadata.csv",
]

REQUIRED_MANIFEST_COLUMNS = {"sequence", "smiles", "structure_path", "chain_id", "target"}


def _feature_complete(out_dir: Path) -> bool:
    return all((out_dir / name).exists() for name in REQUIRED_FEATURE_FILES)


def _label_for(row: dict, target_col: str, target_transform: str) -> float:
    if target_transform == "none":
        return float(row[target_col])
    if target_transform == "log2_from_value":
        value = float(row["value"])
        if value <= 0:
            raise ValueError(f"value must be positive for log2 transform, got {value}")
        return float(np.log2(value))
    raise ValueError(f"Unsupported target_transform: {target_transform}")


def build_feature_split(manifest: Path, out_dir: Path, args, dictionaries: dict) -> dict:
    if _feature_complete(out_dir) and not args.overwrite:
        return {"manifest": str(manifest), "out_dir": str(out_dir), "status": "skipped_exists"}

    frame = read_table(manifest)
    if not REQUIRED_MANIFEST_COLUMNS.issubset(frame.columns):
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(frame.columns))
        raise ValueError(
            f"Prepared manifest has old or invalid schema; missing {missing} in {manifest}. "
            "Run emulator_bench/prepare_splits.py first, or rerun run_split_benchmarks.py after this fix."
        )
    required = ["sequence", "smiles", "structure_path", "chain_id", args.target_col]
    if args.target_transform == "log2_from_value":
        required.append("value")
    require_columns(frame, required, manifest)
    if args.limit_rows:
        frame = frame.head(args.limit_rows).copy()

    cache = FeatureCache(args.cache_dir)
    fingerprints, smileadjacencies, sequences, proteinadjacencies, labels = [], [], [], [], []
    kept_rows, failed_rows = [], []
    stats = {"sequence_hits": 0, "sequence_writes": 0, "smiles_hits": 0, "smiles_writes": 0, "structure_hits": 0, "structure_writes": 0}

    iterator = progress(frame.to_dict("records"), desc=f"Caching {manifest.parent.parent.name}/{manifest.parent.name}/{manifest.stem}", unit="row")
    for idx, row in enumerate(iterator):
        try:
            seq = str(row["sequence"])
            smiles = str(row["smiles"])
            structure_path = str(row["structure_path"])
            chain_id = row.get("chain_id", "A")

            seq_key = cache.seq_key(seq, args.ngram)
            seq_ids = cache.load_sequence(seq_key)
            if seq_ids is None:
                seq_ids = split_sequence(seq, args.ngram, dictionaries["word_dict"])
                cache.save_sequence(seq_key, seq_ids)
                stats["sequence_writes"] += 1
            else:
                stats["sequence_hits"] += 1

            smiles_key = cache.smiles_key(smiles, args.radius)
            fp_ids, smiles_adj = cache.load_smiles(smiles_key)
            if fp_ids is None or smiles_adj is None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {smiles}")
                mol = Chem.AddHs(mol)
                atoms = create_atoms(mol, dictionaries["atom_dict"])
                i_jbond = create_ijbonddict(mol, dictionaries["bond_dict"])
                fp_ids = extract_fingerprints(
                    atoms,
                    i_jbond,
                    args.radius,
                    dictionaries["fingerprint_dict"],
                    dictionaries["edge_dict"],
                )
                smiles_adj = create_adjacency(mol)
                cache.save_smiles(smiles_key, fp_ids, smiles_adj)
                stats["smiles_writes"] += 1
            else:
                stats["smiles_hits"] += 1

            struct_key = cache.struct_key(structure_path, seq, args.dist_thres, chain_id)
            prot_adj = cache.load_structure(struct_key)
            if prot_adj is None:
                prot_adj = protein_contact_map(structure_path, seq, dist_thres=args.dist_thres, chain_id=chain_id)
                cache.save_structure(struct_key, prot_adj)
                stats["structure_writes"] += 1
            else:
                stats["structure_hits"] += 1

            fingerprints.append(np.asarray(fp_ids, dtype=np.int64))
            smileadjacencies.append(np.asarray(smiles_adj, dtype=np.float32))
            sequences.append(np.asarray(seq_ids, dtype=np.int64))
            proteinadjacencies.append(prot_adj)
            labels.append(np.asarray([_label_for(row, args.target_col, args.target_transform)], dtype=np.float32))
            kept_rows.append(idx)
        except Exception as exc:
            failed_rows.append({"row_index": idx, "sample_id": row.get("sample_id", ""), "error": str(exc)})
            if args.fail_on_error:
                raise

    if not kept_rows:
        raise RuntimeError(f"No valid rows were featurized for {manifest}")

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_object_array(out_dir / "fingerprint.npy", fingerprints)
    atomic_save_object_array(out_dir / "smileadjacencies.npy", smileadjacencies)
    atomic_save_object_array(out_dir / "sequences.npy", sequences)
    atomic_save_object_array(out_dir / "proteinadjacencies.npy", proteinadjacencies)
    atomic_save_object_array(out_dir / "logkcat.npy", labels)
    frame.iloc[kept_rows].to_csv(out_dir / "metadata.csv", index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(out_dir / "failed_rows.csv", index=False)

    summary = {
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "rows_in": int(len(frame)),
        "rows_out": int(len(kept_rows)),
        "rows_failed": int(len(failed_rows)),
        "target_col": args.target_col,
        "target_transform": args.target_transform,
        "radius": int(args.radius),
        "ngram": int(args.ngram),
        "dist_thres": float(args.dist_thres),
        **stats,
    }
    atomic_json(out_dir / "feature_summary.json", summary)
    return summary


def load_dictionaries(dict_dir: Path):
    return {
        "atom_dict": load_pickle(dict_dir / "atom_dict_0612.pickle"),
        "bond_dict": load_pickle(dict_dir / "bond_dict_0612.pickle"),
        "edge_dict": load_pickle(dict_dir / "edge_dict_0612.pickle"),
        "fingerprint_dict": load_pickle(dict_dir / "fingerprint_dict_0612.pickle"),
        "word_dict": load_pickle(dict_dir / "sequence_dict_0612.pickle"),
    }


def main():
    parser = argparse.ArgumentParser(description="Cache DeepEnzyme static features for prepared EMULaToR manifests.")
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--manifests_dir", type=Path, default=DEFAULT_MANIFESTS_DIR)
    parser.add_argument("--features_dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--dict_dir", type=Path, default=Path("Data/Input"))
    parser.add_argument("--target_col", type=str, default="target")
    parser.add_argument("--target_transform", choices=["none", "log2_from_value"], default="none")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--ngram", type=int, default=4)
    parser.add_argument("--dist_thres", type=float, default=10.0)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    jobs = discover_split_jobs(args.base_dir, args.split_groups, args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    dictionaries = load_dictionaries(args.dict_dir)
    summaries = []
    for job in jobs:
        for split in ("train", "val", "test"):
            manifest = manifest_path(args.manifests_dir, job, split)
            if not manifest.exists():
                raise FileNotFoundError(f"Missing prepared manifest: {manifest}. Run prepare_splits.py first.")
            summaries.append(build_feature_split(manifest, feature_dir(args.features_dir, job, split), args, dictionaries))

    payload = {"jobs": len(jobs), "summaries": summaries}
    atomic_json(args.features_dir / "cache_features_summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

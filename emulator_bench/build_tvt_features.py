import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm.auto import tqdm

from feature_utils import (
    FeatureCache,
    create_adjacency,
    create_atoms,
    create_ijbonddict,
    extract_fingerprints,
    load_pickle,
    protein_contact_map,
    resolve_structure_path,
    split_sequence,
)


def _require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_split_features(args):
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = FeatureCache(args.cache_dir)

    df = pd.read_csv(input_csv)
    required = [args.sequence_col, args.smiles_col, args.structure_col, args.target_col]
    if args.sample_id_col:
        required.append(args.sample_id_col)
    _require_columns(df, required)

    atom_dict = load_pickle(args.atom_dict)
    bond_dict = load_pickle(args.bond_dict)
    edge_dict = load_pickle(args.edge_dict)
    fingerprint_dict = load_pickle(args.fingerprint_dict)
    word_dict = load_pickle(args.sequence_dict)

    fingerprints, smileadjacencies, sequences, proteinadjacencies, labels = [], [], [], [], []

    rows_ok = 0
    rows_failed = 0
    kept_indices = []
    failed_rows = []

    iterator = tqdm(df.itertuples(index=False), total=len(df), desc=f"Feature build: {input_csv.name}")
    for row_idx, row in enumerate(iterator):
        try:
            seq = str(getattr(row, args.sequence_col))
            smiles = str(getattr(row, args.smiles_col))
            structure_raw = getattr(row, args.structure_col)
            structure_path = resolve_structure_path(structure_raw, input_csv)

            y = float(getattr(row, args.target_col))
            if not args.target_is_log2:
                if y <= 0:
                    raise ValueError(f"Target must be > 0 for log2 transform, got: {y}")
                y = float(np.log2(y))

            seq_key = cache.seq_key(seq, args.ngram)
            smi_key = cache.smiles_key(smiles, args.radius)
            struct_key = cache.struct_key(structure_path, seq, args.dist_thres, args.chain_id)

            # Sequence tokens
            seq_ids = cache.load_sequence(seq_key) if args.cache_read else None
            if seq_ids is None:
                seq_ids = split_sequence(seq, args.ngram, word_dict)
                if args.cache_write:
                    cache.save_sequence(seq_key, seq_ids)

            # SMILES graph features
            fp_ids, smi_adj = cache.load_smiles(smi_key) if args.cache_read else (None, None)
            if fp_ids is None or smi_adj is None:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {smiles}")
                atoms = create_atoms(mol, atom_dict)
                i_jbond = create_ijbonddict(mol, bond_dict)
                fp_ids = extract_fingerprints(atoms, i_jbond, args.radius, fingerprint_dict, edge_dict)
                smi_adj = create_adjacency(mol)
                if args.cache_write:
                    cache.save_smiles(smi_key, fp_ids, smi_adj)

            # Protein contact-map adjacency
            prot_adj = cache.load_structure(struct_key) if args.cache_read else None
            if prot_adj is None:
                prot_adj = protein_contact_map(
                    structure_path,
                    seq,
                    dist_thres=args.dist_thres,
                    chain_id=args.chain_id,
                )
                if args.cache_write:
                    cache.save_structure(struct_key, prot_adj)

            fingerprints.append(np.asarray(fp_ids, dtype=np.int64))
            smileadjacencies.append(np.asarray(smi_adj, dtype=np.float32))
            sequences.append(np.asarray(seq_ids, dtype=np.int64))
            proteinadjacencies.append(prot_adj)
            labels.append(np.asarray([y], dtype=np.float32))
            kept_indices.append(row_idx)
            rows_ok += 1
        except Exception as exc:
            rows_failed += 1
            failed_rows.append({"row_index": row_idx, "error": str(exc)})
            if args.fail_on_error:
                raise
            iterator.write(f"[warn] skipping row due to: {exc}")

    if rows_ok == 0:
        raise RuntimeError("No valid rows were featurized.")

    np.save(output_dir / "fingerprint.npy", np.array(fingerprints, dtype=object), allow_pickle=True)
    np.save(output_dir / "smileadjacencies.npy", np.array(smileadjacencies, dtype=object), allow_pickle=True)
    np.save(output_dir / "sequences.npy", np.array(sequences, dtype=object), allow_pickle=True)
    np.save(output_dir / "proteinadjacencies.npy", np.array(proteinadjacencies, dtype=object), allow_pickle=True)
    np.save(output_dir / "logkcat.npy", np.array(labels, dtype=object), allow_pickle=True)

    meta = df.iloc[kept_indices, :].copy()
    meta.insert(0, "_source_row_index", kept_indices)
    meta.to_csv(output_dir / "metadata.csv", index=False)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(output_dir / "failed_rows.csv", index=False)

    stats = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "rows_total": int(len(df)),
        "rows_ok": int(rows_ok),
        "rows_failed": int(rows_failed),
        "target_col": args.target_col,
        "target_is_log2": bool(args.target_is_log2),
        "radius": int(args.radius),
        "ngram": int(args.ngram),
        "dist_thres": float(args.dist_thres),
        "chain_id": args.chain_id,
    }
    with open(output_dir / "build_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved features to: {output_dir}")
    print(f"Rows ok: {rows_ok} | Rows failed: {rows_failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DeepEnzyme TVT features with persistent caching.")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)
    parser.add_argument("--structure_col", default="structure_path", type=str)
    parser.add_argument("--target_col", default="kcat_value", type=str)
    parser.add_argument("--sample_id_col", default="sample_id", type=str)
    parser.add_argument("--target_is_log2", action="store_true", help="Set when target_col is already log2(kcat).")

    parser.add_argument("--radius", default=2, type=int)
    parser.add_argument("--ngram", default=4, type=int)
    parser.add_argument("--dist_thres", default=10.0, type=float)
    parser.add_argument("--chain_id", default="A", type=str, help="Use 'None' to include all chains.")

    parser.add_argument("--dict_dir", default="Data/Input", type=str)
    parser.add_argument("--atom_dict", default=None, type=str)
    parser.add_argument("--bond_dict", default=None, type=str)
    parser.add_argument("--edge_dict", default=None, type=str)
    parser.add_argument("--fingerprint_dict", default=None, type=str)
    parser.add_argument("--sequence_dict", default=None, type=str)

    parser.add_argument("--cache_dir", default="emulator_bench/.cache_features", type=str)
    parser.add_argument("--no_cache_read", action="store_true")
    parser.add_argument("--no_cache_write", action="store_true")

    parser.add_argument("--fail_on_error", action="store_true")

    args = parser.parse_args()

    if args.chain_id == "None":
        args.chain_id = None

    dict_dir = Path(args.dict_dir)
    args.atom_dict = args.atom_dict or str(dict_dir / "atom_dict_0612.pickle")
    args.bond_dict = args.bond_dict or str(dict_dir / "bond_dict_0612.pickle")
    args.edge_dict = args.edge_dict or str(dict_dir / "edge_dict_0612.pickle")
    args.fingerprint_dict = args.fingerprint_dict or str(dict_dir / "fingerprint_dict_0612.pickle")
    args.sequence_dict = args.sequence_dict or str(dict_dir / "sequence_dict_0612.pickle")

    args.cache_read = not args.no_cache_read
    args.cache_write = not args.no_cache_write

    build_split_features(args)

import hashlib
import json
import os
import pickle
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from rdkit import Chem
from sklearn.metrics import pairwise_distances

CACHE_VERSION = "v2"


def load_pickle(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def _file_signature(path):
    p = Path(path)
    st = p.stat()
    payload = {
        "path": str(p.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    return json.dumps(payload, sort_keys=True)


def split_sequence(sequence, ngram, word_dict):
    sequence = "--" + str(sequence) + "="
    words = []
    for i in range(len(sequence) - ngram + 1):
        token = sequence[i : i + ngram]
        words.append(word_dict.get(token, 0))
    return np.array(words, dtype=np.int64)


def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        idx = a.GetIdx()
        atoms[idx] = (atoms[idx], "aromatic")
    atoms = [atom_dict.get(a, 0) for a in atoms]
    return np.array(atoms, dtype=np.int64)


def create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict.get(str(b.GetBondType()), 0)
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):
            fingerprints = [0 for _ in range(len(nodes))]
            for i, j_edge in i_jedge_dict.items():
                if i >= len(nodes):
                    continue
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints[i] = fingerprint_dict.get(fingerprint, 0)
            nodes = fingerprints

            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                if i >= len(nodes):
                    continue
                for j, edge in j_edge:
                    if j >= len(nodes):
                        continue
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict.get((both_side, edge), 0)
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints, dtype=np.int64)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency, dtype=np.float32)


def get_ca_coords(pdb_path, chain_id="A"):
    with open(pdb_path, "r") as file:
        lines = file.readlines()

    out = []
    for line in lines:
        if not line.startswith("ATOM "):
            continue

        toks = line.split()
        # if len(toks) < 9:
        #     continue

        if len(toks) < 9:
            continue
        atom_name = toks[2]
        current_chain = toks[4]
        if atom_name != "CA":
            continue
        if chain_id is not None and current_chain != chain_id:
            continue

        res_num = toks[5]
        res_name = toks[3]
        x = toks[6]
        y = toks[7]
        z = toks[8]

        if len(x) > 8:
            x = toks[6][:-8]
            y = toks[6][-8:]
            z = toks[7]
        elif len(y) > 8:
            x = toks[6]
            y = toks[7][:-8]
            z = toks[7][-8:]
        elif len(res_num) > 4:
            x = toks[5][-8:]
            y = toks[6]
            z = toks[7]
            res_num = toks[5][:-8]

        out.append([res_num, res_name, x, y, z])

    return pd.DataFrame(out, columns=["res_num", "res_name", "x", "y", "z"])


def protein_contact_map(pdb_path, seq, dist_thres=10.0, chain_id="A"):
    if chain_id in {"", "None", "none", "null"}:
        chain_id = None
    ca_coords = get_ca_coords(pdb_path, chain_id=chain_id)

    if len(ca_coords) == 0 and chain_id is not None:
        ca_coords = get_ca_coords(pdb_path, chain_id=None)

    if len(ca_coords) == 0:
        raise ValueError(f"No C-alpha atoms found in PDB: {pdb_path}")

    xyz = ca_coords[["x", "y", "z"]].astype(float).values
    dist_arr = pairwise_distances(xyz)
    cont_arr = (dist_arr < dist_thres).astype(np.int32)

    target_len = len(seq)
    if cont_arr.shape[0] == target_len:
        proteinadjacency = sparse.csr_matrix(cont_arr)
    else:
        if cont_arr.shape[0] > target_len:
            cont_arr = cont_arr[:target_len, :target_len]
        else:
            a = np.zeros((cont_arr.shape[0], target_len - cont_arr.shape[0]), dtype=np.int32)
            cont_arr = np.column_stack((cont_arr, a))
            b = np.zeros((target_len - cont_arr.shape[0], target_len), dtype=np.int32)
            cont_arr = np.row_stack((cont_arr, b))
        row, col = np.diag_indices_from(cont_arr)
        cont_arr[row, col] = 1
        proteinadjacency = sparse.csr_matrix(cont_arr)

    return proteinadjacency


class FeatureCache:
    def __init__(self, cache_dir):
        self.cache_root = _ensure_dir(cache_dir)
        self.seq_dir = _ensure_dir(self.cache_root / "sequence")
        self.smiles_dir = _ensure_dir(self.cache_root / "smiles")
        self.struct_dir = _ensure_dir(self.cache_root / "structure")

    def _path(self, namespace_dir, key, suffix):
        return namespace_dir / key[:2] / f"{key}.{suffix}"

    def seq_key(self, sequence, ngram):
        payload = f"{CACHE_VERSION}|seq|ngram={ngram}|{sequence}"
        return _sha256_text(payload)

    def smiles_key(self, smiles, radius):
        can = _canonical_smiles(smiles)
        payload = f"{CACHE_VERSION}|smiles|radius={radius}|{can}"
        return _sha256_text(payload)

    def struct_key(self, structure_path, sequence, dist_thres, chain_id):
        sig = _file_signature(structure_path)
        payload = f"{CACHE_VERSION}|struct|dist={dist_thres}|chain={chain_id}|seq_len={len(sequence)}|{sig}"
        return _sha256_text(payload)

    def load_sequence(self, key):
        fpath = self._path(self.seq_dir, key, "npy")
        if not fpath.exists():
            return None
        try:
            return np.load(fpath, allow_pickle=False)
        except Exception:
            return None

    def save_sequence(self, key, arr):
        fpath = self._path(self.seq_dir, key, "npy")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = fpath.with_suffix(f".tmp.{os.getpid()}.npy")
        np.save(tmp, np.asarray(arr, dtype=np.int64), allow_pickle=False)
        os.replace(tmp, fpath)

    def load_smiles(self, key):
        fpath = self._path(self.smiles_dir, key, "npz")
        if not fpath.exists():
            return None, None
        try:
            data = np.load(fpath, allow_pickle=False)
            return data["fingerprints"], data["adjacency"]
        except Exception:
            return None, None

    def save_smiles(self, key, fingerprints, adjacency):
        fpath = self._path(self.smiles_dir, key, "npz")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = fpath.with_suffix(f".tmp.{os.getpid()}.npz")
        np.savez_compressed(
            tmp,
            fingerprints=np.asarray(fingerprints, dtype=np.int64),
            adjacency=np.asarray(adjacency, dtype=np.float32),
        )
        os.replace(tmp, fpath)

    def load_structure(self, key):
        fpath = self._path(self.struct_dir, key, "npz")
        if not fpath.exists():
            return None
        try:
            return sparse.load_npz(fpath)
        except Exception:
            return None

    def save_structure(self, key, matrix):
        fpath = self._path(self.struct_dir, key, "npz")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = fpath.with_suffix(f".tmp.{os.getpid()}.npz")
        sparse.save_npz(tmp, matrix)
        os.replace(tmp, fpath)


def resolve_structure_path(value, csv_path):
    p = Path(str(value))
    if p.is_absolute():
        return str(p)
    return str((Path(csv_path).parent / p).resolve())


def atomic_save_object_array(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.npy")
    np.save(tmp, np.array(values, dtype=object), allow_pickle=True)
    os.replace(tmp, path)

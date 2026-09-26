"""Frozen, verifiable train/validation split on Source-1 entity IDs.

The split files are gitignored, so every load is checked against the committed
SHA-256 manifest (`experiments/results/validation_split_manifest.json`). A
mismatch raises instead of silently evaluating on a different split.
Regenerating from the ground truth reproduces the manifest byte-for-byte.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .. import config

SPLIT_FILES = {"train": "train_ids.csv", "val": "val_ids.csv"}
MANIFEST_KEYS = {"train": "train_id_file_sha256", "val": "validation_id_file_sha256"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_split_files(split_dir=None, manifest_path=None) -> None:
    split_dir, manifest_path = Path(split_dir or config.SPLIT_DIR), Path(manifest_path or config.SPLIT_MANIFEST)
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    for part, name in SPLIT_FILES.items():
        got, want = _sha256(split_dir / name), manifest[MANIFEST_KEYS[part]]
        if got != want:
            raise RuntimeError(
                f"{split_dir / name} does not match the frozen manifest (sha256 {got} != {want}). "
                "Delete the split files to regenerate them, or get them from Mudit.")


def create_validation_split(ground_truth_df: pd.DataFrame = None, test_size=0.2, random_state=None,
                            split_dir=None, manifest_path=None) -> tuple:
    """Return (train_s1_ids, val_s1_ids) as sets; load frozen files or create them.

    `ground_truth_df` is only needed the first time (to create the files).
    """
    split_dir = Path(split_dir or config.SPLIT_DIR)
    paths = {part: split_dir / name for part, name in SPLIT_FILES.items()}
    if not all(p.exists() for p in paths.values()):
        if ground_truth_df is None:
            raise FileNotFoundError(f"No split files in {split_dir}; pass the ground truth to create them.")
        # Appearance-ordered unique IDs as a plain numpy array: identical order to the
        # original pandas-2 implementation (pandas 3 returns an Arrow array sklearn cannot index).
        ids = np.asarray(ground_truth_df["source1_entity_id"].unique(), dtype=object)
        train_ids, val_ids = train_test_split(
            ids, test_size=test_size, random_state=config.SEED if random_state is None else random_state)
        split_dir.mkdir(parents=True, exist_ok=True)
        for part, arr in (("train", train_ids), ("val", val_ids)):
            # lineterminator fixed so the bytes (and hash) are identical on every OS
            pd.DataFrame({"entity_id": arr}).to_csv(paths[part], index=False, lineterminator="\r\n")
    verify_split_files(split_dir, manifest_path)
    return tuple(set(pd.read_csv(paths[p], dtype=str)["entity_id"]) for p in ("train", "val"))


def apply_validation_split(df: pd.DataFrame, s1_id_col: str, train_ids: set, val_ids: set) -> tuple:
    """Split a frame holding Source-1 IDs into (train_df, val_df)."""
    return df[df[s1_id_col].isin(train_ids)].copy(), df[df[s1_id_col].isin(val_ids)].copy()


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Verify the frozen split without creating it")
    args = parser.parse_args()
    
    if args.verify:
        split_dir = Path(config.SPLIT_DIR)
        manifest_path = Path(config.SPLIT_MANIFEST)
        if not manifest_path.exists():
            print(f"Manifest not found at {manifest_path}")
            sys.exit(1)
        
        manifest = json.loads(manifest_path.read_text())
        
        for part, name in SPLIT_FILES.items():
            file_path = split_dir / name
            if not file_path.exists():
                print(f"Missing {file_path}")
                sys.exit(1)
            got, want = _sha256(file_path), manifest[MANIFEST_KEYS[part]]
            if got != want:
                print(f"Hash mismatch on {file_path}: got {got}, want {want}")
                sys.exit(1)
                
        train_df = pd.read_csv(split_dir / SPLIT_FILES["train"], dtype=str)
        val_df = pd.read_csv(split_dir / SPLIT_FILES["val"], dtype=str)
        
        train_ids = set(train_df["entity_id"])
        val_ids = set(val_df["entity_id"])
        
        print(f"Train IDs: {len(train_ids)}")
        print(f"Val IDs: {len(val_ids)}")
        
        overlap = train_ids.intersection(val_ids)
        if overlap:
            print(f"CRITICAL: Found {len(overlap)} overlapping IDs!")
            sys.exit(1)
            
        print("Validation split verified successfully.")
        sys.exit(0)
    else:
        print("Use --verify to verify the split.")
        sys.exit(1)

import json
import hashlib
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()

def verify_and_extract_s1(tsv_path, out_csv_path, expected_n_s1=None):
    logging.info(f"Verifying TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", compression="gzip")
    
    row_count = len(df)
    duplicates = df.duplicated(subset=["source1_entity_id", "candidate_entity_id"]).sum()
    n_s1 = df["source1_entity_id"].nunique()

    logging.info(f"Row count: {row_count}")
    logging.info(f"Unique S1 count: {n_s1}")
    logging.info(f"Duplicate pairs: {duplicates}")

    if expected_n_s1 is not None:
        assert n_s1 == expected_n_s1, f"Expected {expected_n_s1} unique S1s, got {n_s1}"
    assert duplicates == 0, f"Expected 0 duplicates, got {duplicates}"
    
    logging.info(f"Writing {out_csv_path}")
    s1_ids = pd.DataFrame(df["source1_entity_id"].unique(), columns=["source1_entity_id"])
    s1_ids.to_csv(out_csv_path, index=False)
    
    return row_count, n_s1, sha256(tsv_path), sha256(out_csv_path)

def main():
    repo_root = Path(__file__).resolve().parent.parent
    in_dir = repo_root / "artifacts" / "handoff" / "BLK-020"
    if not in_dir.exists():
        logging.error(f"Directory {in_dir} not found. Please place Ayush's artifacts there.")
        return
        
    train_tsv = in_dir / "train_candidate_pairs.tsv.gz"
    val_tsv = in_dir / "validation_candidate_pairs.tsv.gz"
    meta_json = in_dir / "metadata.json"
    
    for f in [train_tsv, val_tsv, meta_json]:
        if not f.exists():
            logging.error(f"Missing file: {f}")
            return
            
    # Verify train
    train_csv = in_dir / "train_source_ids.csv"
    train_rows, train_s1, train_tsv_hash, train_csv_hash = verify_and_extract_s1(train_tsv, train_csv)
    
    # Verify val (should be 441,365 S1s for frozen validation split)
    val_csv = in_dir / "validation_source_ids.csv"
    val_rows, val_s1, val_tsv_hash, val_csv_hash = verify_and_extract_s1(val_tsv, val_csv, expected_n_s1=441365)
    
    logging.info("Verification complete. Please ensure metadata.json contains the correct hashes.")

if __name__ == "__main__":
    main()

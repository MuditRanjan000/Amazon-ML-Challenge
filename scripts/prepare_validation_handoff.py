import json
import hashlib
import shutil
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

def main():
    repo_root = Path(__file__).resolve().parent.parent
    src_tsv = repo_root / "artifacts" / "blocking" / "validation_candidate_pairs.tsv"
    out_dir = repo_root / "artifacts" / "handoff" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_tsv = out_dir / "validation_candidate_pairs.tsv"
    out_csv = out_dir / "validation_source_ids.csv"
    out_meta = out_dir / "metadata.json"

    # 1. Copy TSV
    logging.info(f"Copying {src_tsv} to {out_tsv}")
    shutil.copy2(src_tsv, out_tsv)

    # 2. Verify Schema, Row count, and Duplicates
    logging.info("Verifying TSV (row count, duplicates, schema)...")
    df = pd.read_csv(out_tsv, sep="\t")
    
    row_count = len(df)
    duplicates = df.duplicated(subset=["source1_entity_id", "candidate_entity_id"]).sum()
    n_s1 = df["source1_entity_id"].nunique()

    logging.info(f"Row count: {row_count}")
    logging.info(f"Unique source1_entity_id count: {n_s1}")
    logging.info(f"Duplicate pairs: {duplicates}")

    # The validation split has exactly 441,365 queries.
    assert n_s1 == 441365, f"Expected 441365 unique S1s, got {n_s1}"
    assert duplicates == 0, f"Expected 0 duplicates, got {duplicates}"
    
    # 3. Write source IDs to CSV
    logging.info(f"Writing {out_csv}")
    s1_ids = pd.DataFrame(df["source1_entity_id"].unique(), columns=["source1_entity_id"])
    s1_ids.to_csv(out_csv, index=False)
    
    # 4. Hash files
    logging.info("Calculating hashes...")
    tsv_hash = sha256(out_tsv)
    csv_hash = sha256(out_csv)
    
    # 5. Extract metadata from blocking_metadata.json
    blocking_meta_path = repo_root / "artifacts" / "blocking" / "blocking_metadata.json"
    with open(blocking_meta_path) as f:
        blocking_meta = json.load(f)
    
    val_meta = blocking_meta["splits"]["validation"]
    
    metadata = {
        "generator_commit": val_meta["generator_commit"],
        "config_hash": val_meta["config_sha"],
        "blocker": val_meta["blocker"],
        "fields": val_meta["config"]["text_fields"],
        "K": 200,
        "validation_sample_size": n_s1,
        "candidate_pair_count": row_count,
        "recall_at_200": val_meta["recall_at_200"],
        "ceiling_f05_at_200": val_meta["ceiling_f05_at_200"],
        "files": {
            "validation_candidate_pairs.tsv": {"sha256": tsv_hash},
            "validation_source_ids.csv": {"sha256": csv_hash}
        }
    }
    
    logging.info(f"Writing {out_meta}")
    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)
        
    logging.info("Done!")

if __name__ == "__main__":
    main()

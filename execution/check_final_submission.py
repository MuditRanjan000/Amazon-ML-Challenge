import argparse
import sys
import pandas as pd
from pathlib import Path
import subprocess

def run_checks(matching_tsv, test_dir):
    print(f"--- Running Integrity Checks on {matching_tsv} ---")
    
    # 1. Run official validator
    print("\n1. Running validate_submission.py...")
    res = subprocess.run(
        f"python scripts/validate_submission.py --submission {matching_tsv} --test-dir {test_dir} --check-ids",
        shell=True, capture_output=True, text=True
    )
    if res.returncode != 0:
        print(f"FAILED: Official validator failed.\n{res.stdout}\n{res.stderr}")
        sys.exit(1)
    else:
        print("PASSED: Official validator succeeded.")

    # 2. Check every S1 test ID exists exactly once
    print("\n2. Checking every S1 test ID exists exactly once...")
    df = pd.read_csv(matching_tsv, sep="\t", dtype=str)
    test_s1_path = Path(test_dir) / "test_source1.tsv"
    test_s1 = pd.read_csv(test_s1_path, sep="\t", dtype=str)
    
    missing_ids = set(test_s1["record_id"]) - set(df["source1_entity_id"])
    extra_ids = set(df["source1_entity_id"]) - set(test_s1["record_id"])
    duplicates = df[df.duplicated(subset=["source1_entity_id"])]
    
    if missing_ids or extra_ids or not duplicates.empty:
        print(f"FAILED: ID mismatch.")
        print(f"Missing IDs: {len(missing_ids)}")
        print(f"Extra IDs: {len(extra_ids)}")
        print(f"Duplicate IDs: {len(duplicates)}")
        sys.exit(1)
    else:
        print("PASSED: Exact S1 test ID match, no duplicates.")

    # 3. Check singleton formatting
    print("\n3. Checking singleton formatting...")
    # Singletons must have an empty matched_entity_ids (pandas reads empty string as NaN unless handled)
    invalid_singletons = df[df["matched_entity_ids"].str.contains(r"^\s*$", na=False)]
    if not invalid_singletons.empty:
        print("FAILED: Found whitespace instead of empty strings for singletons.")
        sys.exit(1)
    
    # Check for trailing commas
    invalid_commas = df[df["matched_entity_ids"].str.endswith(",", na=False)]
    if not invalid_commas.empty:
        print("FAILED: Found trailing commas in matched_entity_ids.")
        sys.exit(1)
    print("PASSED: Singleton formatting is correct.")

    # 4. Check TSV header
    print("\n4. Checking TSV header...")
    expected_header = ["source1_entity_id", "matched_entity_ids"]
    if list(df.columns) != expected_header:
        print(f"FAILED: Invalid header {list(df.columns)}. Expected {expected_header}.")
        sys.exit(1)
    else:
        print("PASSED: TSV header matches exact specification.")

    # 5. Calculate final row counts
    print("\n5. Calculating final row counts...")
    total_rows = len(df)
    singletons = df["matched_entity_ids"].isna().sum()
    non_singletons = total_rows - singletons
    print(f"Total Rows: {total_rows}")
    print(f"Singletons (Empty matches): {singletons}")
    print(f"Matched entities (1 or more matches): {non_singletons}")
    
    print("\nALL CHECKS PASSED. Submission is safe to upload.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--test-dir", default=".tmp/artifacts_unzip/data/test")
    args = parser.parse_args()
    
    run_checks(args.submission, args.test_dir)

import argparse
import logging
import json
import time
import pandas as pd
from pathlib import Path
from entity_resolution import config
from entity_resolution.tracking import log_run, git_commit

# For merge, we will use the streaming models
from entity_resolution.models.streaming import (
    assemble_score_output, build_threshold_decisions, evaluate_with_shared_evaluator
)

def main():
    p = argparse.ArgumentParser(description="Fan-out wrapper for streaming matcher")
    p.add_argument("--split", choices=["validation", "test"], required=True)
    p.add_argument("--part", help="i/n: run only this contiguous slice of Source-1 IDs")
    p.add_argument("--merge", type=int, help="n: merge n parts into final decisions")
    
    # Paths required for both
    p.add_argument("--run-dir", type=Path, default=Path(".tmp/runs/AWS-MATCHER"))
    p.add_argument("--exp-id", default="AWS-MATCHER-01")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for final decisions")
    
    # Paths required for --part
    p.add_argument("--candidates", type=Path)
    p.add_argument("--source1-ids", type=Path)
    p.add_argument("--store", type=Path)
    p.add_argument("--feature-artifact", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--ground-truth", type=Path)
    
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    
    a.run_dir.mkdir(parents=True, exist_ok=True)
    
    if a.merge:
        logging.info(f"Merging {a.merge} parts for split {a.split}...")
        # Assume parts are in args.run_dir/parts/part01/validation_logistic_scores.tsv etc
        parts_dir = a.run_dir / "parts"
        merged_scores = a.run_dir / f"{a.split}_logistic_scores.tsv"
        
        # Merge scores
        with open(merged_scores, "wb") as dst:
            header_written = False
            for i in range(1, a.merge + 1):
                part_file = parts_dir / f"part{i:02d}" / "validation_logistic_scores.tsv" # Always named validation_ inside the streaming script
                if not part_file.exists():
                    raise FileNotFoundError(f"Missing part {i}: {part_file}")
                with open(part_file, "rb") as src:
                    header = src.readline()
                    if not header_written:
                        dst.write(header)
                        header_written = True
                    for block in iter(lambda: src.read(1024 * 1024 * 16), b""):
                        dst.write(block)
                        
        logging.info(f"Merged scores to {merged_scores}")
        
        # Now run decisions
        decisions_path = a.run_dir / f"{a.split}_logistic_decisions.tsv"
        decisions = build_threshold_decisions(
            score_path=merged_scores,
            source1_ids_path=a.source1_ids,
            output_path=decisions_path,
            threshold=a.threshold,
            working_database=a.run_dir / "decisions.sqlite"
        )
        
        if a.split == "validation" and a.ground_truth:
            metrics = evaluate_with_shared_evaluator(
                ground_truth_path=a.ground_truth,
                source1_ids_path=a.source1_ids,
                decisions_path=decisions_path
            )
            logging.info(f"Metrics: {metrics}")
            log_run({
                "experiment_id": a.exp_id,
                "stage": "matching",
                "split": a.split,
                "threshold": a.threshold,
                "metrics": metrics
            })
            
        logging.info("Merge complete.")
        return

    # Part execution
    if not a.part:
        p.error("Must specify --part i/n or --merge n")
        
    i_str, n_str = a.part.split("/")
    i, n = int(i_str), int(n_str)
    
    logging.info(f"Running matcher part {i}/{n} on {a.split}")
    
    # Import and run the streaming matcher
    from entity_resolution.models.streaming import (
        CandidatePairSpool, StreamingConfig, score_logistic_stream, SQLiteRecordStore
    )
    import pickle
    
    part_dir = a.run_dir / "parts" / f"part{i:02d}"
    part_dir.mkdir(parents=True, exist_ok=True)
    
    spool = CandidatePairSpool(part_dir / "candidate_spool.sqlite")
    spool.build(a.candidates, a.source1_ids)
    
    with open(a.feature_artifact, "rb") as f:
        extractor = pickle.load(f)
    with open(a.model, "rb") as f:
        model = pickle.load(f)
        
    score_logistic_stream(
        spool=spool,
        record_store=SQLiteRecordStore(a.store),
        extractor=extractor,
        model=model,
        run_dir=part_dir,
        config=StreamingConfig(pair_batch_size=5000, write_rule_raw_scores=False, source1_partition_count=n, source1_partition_index=i-1),
        model_path=a.model,
        feature_artifact_path=a.feature_artifact
    )
    
    # Assemble part scores
    assemble_score_output(part_dir, part_dir / "validation_logistic_scores.tsv")
    logging.info(f"Part {i} complete.")

if __name__ == "__main__":
    main()

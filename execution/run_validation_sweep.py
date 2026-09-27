import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import shutil
import numpy as np

from entity_resolution.models.streaming import build_threshold_decisions, evaluate_with_shared_evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True, help="Path to merged validation_logistic_scores.tsv")
    parser.add_argument("--candidates", type=Path, required=True, help="Path to validation_candidate_pairs.tsv.gz")
    parser.add_argument("--source1-ids", type=Path, required=True, help="Path to validation_ids.csv")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Path to train_ground_truth.tsv")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory to save sweep results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [round(t, 2) for t in np.arange(0.50, 0.81, 0.01)]
    ks = [200, 100]

    logging.info(f"Loading candidates to build rank filters from {args.candidates}...")
    # Load rank <= 100
    df_cands = pd.read_csv(args.candidates, sep="\t", usecols=["source1_entity_id", "candidate_entity_id", "rank"])
    valid_k100 = set(df_cands[df_cands["rank"] <= 100][["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None))
    del df_cands
    logging.info(f"Loaded {len(valid_k100):,} pairs for K=100 filter.")

    results = []

    for k in ks:
        logging.info(f"--- Processing K={k} ---")
        
        # We process the scores file once per K to create a filtered version
        scores_k_path = args.run_dir / f"scores_K{k}.tsv"
        
        if k == 200:
            # K=200 is the original score file (since candidates were generated at K=200)
            # Just symlink or copy
            scores_k_path = args.scores
        else:
            logging.info(f"Filtering scores for K={k}...")
            # We filter the full score file
            with open(args.scores, "r", encoding="utf-8") as src, open(scores_k_path, "w", encoding="utf-8") as dst:
                header = src.readline()
                dst.write(header)
                cols = header.strip().split("\t")
                idx_s1 = cols.index("source1_entity_id")
                idx_c2 = cols.index("candidate_entity_id")
                
                kept = 0
                for line in src:
                    parts = line.strip("\r\n").split("\t")
                    if (parts[idx_s1], parts[idx_c2]) in valid_k100:
                        dst.write(line)
                        kept += 1
            logging.info(f"Kept {kept:,} scored pairs for K={k}.")

        for t in thresholds:
            logging.info(f"Threshold = {t}")
            db_path = args.run_dir / f"decisions_K{k}_T{t}.sqlite"
            db_path.unlink(missing_ok=True)
            
            decisions_tsv = args.run_dir / f"decisions_K{k}_T{t}.tsv"
            
            build_threshold_decisions(
                score_path=scores_k_path,
                source1_ids_path=args.source1_ids,
                output_path=decisions_tsv,
                threshold=t,
                working_database=db_path
            )
            
            metrics = evaluate_with_shared_evaluator(
                ground_truth_path=args.ground_truth,
                source1_ids_path=args.source1_ids,
                decisions_path=decisions_tsv
            )
            
            logging.info(f"Result for K={k}, T={t} -> F0.5: {metrics['macro_f05']:.5f}")
            
            results.append({
                "k": k,
                "threshold": t,
                "metrics": metrics,
                "decisions_file": str(decisions_tsv)
            })

    # Save summary
    summary_path = args.run_dir / "sweep_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Identify best config
    best = max(results, key=lambda x: x["metrics"]["macro_f05"])
    logging.info(f"Best configuration: K={best['k']}, Threshold={best['threshold']} with F0.5={best['metrics']['macro_f05']:.5f}")
    
    # Convert best decisions TSV to final matching_results.tsv (they are identical formats)
    final_output = args.run_dir / "matching_results.tsv"
    shutil.copy2(best["decisions_file"], final_output)
    logging.info(f"Saved best submission to {final_output}")

if __name__ == "__main__":
    main()

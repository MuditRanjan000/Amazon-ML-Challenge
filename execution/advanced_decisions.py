import argparse
import pandas as pd
from pathlib import Path
import json
import logging
import sqlite3
import sys
import numpy as np

sys.path.insert(0, "src")
from entity_resolution.models.streaming import evaluate_with_shared_evaluator

def evaluate_mask(df, mask, name, run_dir, ground_truth, source1_ids):
    accepted = df[mask]
    
    db_path = run_dir / f"adv_{name}.sqlite"
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    
    s1_ids = pd.read_csv(source1_ids, sep=",", dtype=str)["source1_entity_id"].tolist()
    
    conn.execute("CREATE TABLE source1_ids (ordinal INTEGER PRIMARY KEY, source1_entity_id TEXT UNIQUE NOT NULL)")
    conn.execute("CREATE TABLE accepted (source1_entity_id TEXT NOT NULL, candidate_entity_id TEXT NOT NULL)")
    
    conn.executemany("INSERT INTO source1_ids (ordinal, source1_entity_id) VALUES (?, ?)", enumerate(s1_ids))
    conn.executemany("INSERT INTO accepted VALUES (?, ?)", accepted[["source1_entity_id", "candidate_entity_id"]].values.tolist())
    
    out_tsv = run_dir / f"adv_{name}.tsv"
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for source, matches in conn.execute(
            "SELECT s.source1_entity_id, COALESCE(GROUP_CONCAT(a.candidate_entity_id, ','), '') "
            "FROM source1_ids AS s LEFT JOIN accepted AS a ON a.source1_entity_id = s.source1_entity_id "
            "GROUP BY s.ordinal, s.source1_entity_id ORDER BY s.ordinal"
        ):
            f.write(f"{source}\t{matches}\n")
    
    conn.close()
    
    metrics = evaluate_with_shared_evaluator(
        ground_truth_path=ground_truth,
        source1_ids_path=source1_ids,
        decisions_path=out_tsv
    )
    logging.info(f"{name} -> F0.5: {metrics['macro_f05']:.5f} | Precision: {metrics.get('precision', 0):.5f} | Recall: {metrics.get('recall', 0):.5f}")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Path to validation_logistic_scores.tsv")
    parser.add_argument("--stats", required=True, help="Path to record_stats_trainval.parquet")
    parser.add_argument("--source1-ids", required=True, help="Path to validation_ids.csv")
    parser.add_argument("--ground-truth", required=True, help="Path to train_ground_truth.tsv")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading scores...")
    scores = pd.read_csv(args.scores, sep="\t", dtype={"source1_entity_id": str, "candidate_entity_id": str})
    
    logging.info("Loading stats...")
    stats = pd.read_parquet(args.stats)
    
    logging.info("Merging stats into scores...")
    df = scores.merge(stats, on="candidate_entity_id", how="left")
    
    # Advanced features for decisions
    # 'is_owner' = true if this S1 had the highest blocking score for the candidate
    df["is_owner"] = df["source1_entity_id"] == df["best_s1"]
    
    # Margin of logistic match probability between highest and second highest
    # We will compute rank 1 and rank 2 PROBABILITIES for each candidate
    prob_ranks = df.sort_values(["candidate_entity_id", "match_probability"], ascending=[True, False])
    prob_ranks["prob_rank"] = prob_ranks.groupby("candidate_entity_id").cumcount()
    
    best_probs = prob_ranks[prob_ranks["prob_rank"] == 0].set_index("candidate_entity_id")["match_probability"].rename("best_prob")
    second_probs = prob_ranks[prob_ranks["prob_rank"] == 1].set_index("candidate_entity_id")["match_probability"].rename("second_prob")
    
    df = df.join(best_probs, on="candidate_entity_id", how="left")
    df = df.join(second_probs, on="candidate_entity_id", how="left").fillna({"second_prob": 0.0})
    
    df["is_prob_owner"] = df["match_probability"] == df["best_prob"]
    df["prob_margin"] = df["best_prob"] - df["second_prob"]

    results = {}
    
    thresholds = np.arange(0.50, 0.81, 0.01)

    logging.info("--- Baseline Raw Thresholds ---")
    for t in thresholds:
        t_round = round(t, 2)
        results[f"raw_{t_round}"] = evaluate_mask(df, df["match_probability"] >= t, f"raw_{t_round}", run_dir, args.ground_truth, args.source1_ids)

    logging.info("--- One-Owner-Per-Candidate ---")
    for t in thresholds:
        t_round = round(t, 2)
        results[f"one_owner_{t_round}"] = evaluate_mask(df, (df["match_probability"] >= t) & df["is_prob_owner"], f"one_owner_{t_round}", run_dir, args.ground_truth, args.source1_ids)

    logging.info("--- Probability Margin Filtering ---")
    # Margin sweep: t1 (strict raw), t2 (lenient + owner + margin), margin
    # To keep the grid manageable, we sweep the gap and strict threshold, tying lenient to strict
    margins = [0.05, 0.10, 0.15, 0.20]
    
    # We will test t2 from 0.50 to 0.70, and t1 = t2 + gap
    for t2 in np.arange(0.50, 0.71, 0.02):
        t2_round = round(t2, 2)
        for gap in margins:
            t1_round = round(t2_round + gap, 2)
            if t1_round > 0.80: continue
            
            name = f"margin_t1_{t1_round}_t2_{t2_round}_gap_{gap}"
            mask = (df["match_probability"] >= t1_round) | ((df["match_probability"] >= t2_round) & df["is_prob_owner"] & (df["prob_margin"] >= gap))
            results[name] = evaluate_mask(df, mask, name, run_dir, args.ground_truth, args.source1_ids)

    best_name = max(results, key=lambda k: results[k]["macro_f05"])
    logging.info(f"BEST CONFIGURATION: {best_name} with F0.5={results[best_name]['macro_f05']:.5f}")
    
    with open(run_dir / "advanced_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

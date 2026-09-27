import argparse
import time
import logging
import gc
import pandas as pd
import numpy as np
from rapidfuzz import distance, fuzz
from sklearn.linear_model import LogisticRegression

from entity_resolution.data.loader import DataLoader, explode_ground_truth
from entity_resolution.evaluation.evaluator import Evaluator
from entity_resolution.tracking import log_run
from entity_resolution.models.decision import DecisionLayer

def extract_features(pairs: pd.DataFrame, s1_dict: dict, cand_dict: dict) -> pd.DataFrame:
    # We will build numpy arrays for speed
    n = len(pairs)
    f_name_exact = np.zeros(n, dtype=np.float32)
    f_addr_exact = np.zeros(n, dtype=np.float32)
    f_name_jw = np.zeros(n, dtype=np.float32)
    f_addr_jw = np.zeros(n, dtype=np.float32)
    f_name_tsr = np.zeros(n, dtype=np.float32)

    s1_ids = pairs["source1_entity_id"].to_numpy()
    cand_ids = pairs["candidate_entity_id"].to_numpy()
    
    for i in range(n):
        s1 = s1_dict.get(s1_ids[i])
        c = cand_dict.get(cand_ids[i])
        if not s1 or not c:
            continue
            
        n1, a1 = s1
        n2, a2 = c
        
        if n1 == n2 and n1 != "":
            f_name_exact[i] = 1.0
        if a1 == a2 and a1 != "":
            f_addr_exact[i] = 1.0
            
        f_name_jw[i] = distance.JaroWinkler.normalized_similarity(n1, n2) if n1 and n2 else 0.0
        f_addr_jw[i] = distance.JaroWinkler.normalized_similarity(a1, a2) if a1 and a2 else 0.0
        f_name_tsr[i] = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        
    return pd.DataFrame({
        "name_exact": f_name_exact,
        "addr_exact": f_addr_exact,
        "name_jw": f_name_jw,
        "addr_jw": f_addr_jw,
        "name_tsr": f_name_tsr,
    })

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--one-owner", type=int, default=1)
    p.add_argument("--margin", type=float, default=0.0)
    args = p.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    
    loader = DataLoader()
    logging.info("Loading S1 dict...")
    s1_df = loader.load_source("train", 1)
    s1_dict = {
        row.entity_id: (row.business_name.lower().strip(), row.business_address.lower().strip())
        for row in s1_df.itertuples()
    }
    del s1_df; gc.collect()
    
    logging.info("Loading Cand dict...")
    cand_df = loader.load_candidates_pool("train")
    cand_dict = {
        row.entity_id: (row.business_name.lower().strip(), row.business_address.lower().strip())
        for row in cand_df.itertuples()
    }
    del cand_df; gc.collect()
    
    logging.info("Loading training candidates (using 500k sample)...")
    # Actually we can just load the 500k sample to train LR
    train_pairs = pd.read_csv("artifacts/blocking/train_sample_candidate_pairs.tsv", sep="\t", dtype="string")
    
    logging.info("Extracting train features...")
    X_train = extract_features(train_pairs, s1_dict, cand_dict)
    
    logging.info("Loading ground truth...")
    gt = explode_ground_truth(loader.load_ground_truth())
    gt["is_match"] = 1
    
    train_labeled = train_pairs.merge(gt, on=["source1_entity_id", "candidate_entity_id"], how="left")
    y_train = train_labeled["is_match"].fillna(0).astype(int).to_numpy()
    
    logging.info(f"Training LR on {len(X_train)} pairs (Positives: {y_train.sum()})...")
    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    logging.info(f"LR Coefficients: {lr.coef_}")
    
    del X_train, train_labeled, train_pairs
    gc.collect()
    
    logging.info("Evaluating on Validation pairs in chunks...")
    val_gt = loader.load_ground_truth()
    val_ids = pd.read_csv("artifacts/validation_split/val_ids.csv", sep="\t")["source1_entity_id"].tolist()
    val_gt = val_gt[val_gt["source1_entity_id"].isin(val_ids)]
    
    chunk_size = 5_000_000
    reader = pd.read_csv("artifacts/handoff/BLK-020/validation_candidate_pairs.tsv.gz", sep="\t", dtype="string", chunksize=chunk_size)
    
    decision_layer = DecisionLayer(threshold=args.threshold, one_owner=bool(args.one_owner), margin=args.margin)
    
    all_decisions = []
    
    for i, chunk in enumerate(reader):
        logging.info(f"Processing chunk {i+1}...")
        X_val = extract_features(chunk, s1_dict, cand_dict)
        chunk["probability"] = lr.predict_proba(X_val)[:, 1]
        
        # We can pre-filter here before appending to save memory
        filtered = chunk[chunk["probability"] >= args.threshold]
        all_decisions.append(filtered)
        gc.collect()
        
    logging.info("Applying global decision layer...")
    scored_val = pd.concat(all_decisions, ignore_index=True)
    final_decisions = decision_layer.decide(scored_val)
    
    logging.info("Formatting for evaluation...")
    pred = (final_decisions.groupby("source1_entity_id")["candidate_entity_id"]
            .agg(",".join).rename("matched_entity_ids").reset_index())
            
    metrics = Evaluator().evaluate(val_gt, pred)
    
    rec = log_run({
        "experiment_id": "MUDIT-LR-BASELINE",
        "stage": "matching", 
        "split": "val",
        "threshold": args.threshold,
        "one_owner": args.one_owner,
        "margin": args.margin,
        "metrics": metrics,
        "runtime_s": round(time.time() - t0, 1)
    })
    
    for k, v in metrics.items():
        print(f"{k:22s} {v}")
        
    print(f"runtime {rec['runtime_s']}s | peak RSS {rec['peak_rss_gb']} GB")

if __name__ == "__main__":
    main()

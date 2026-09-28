import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pickle
import time
import json
from sklearn.ensemble import HistGradientBoostingClassifier

from entity_resolution.features.competition import add_competition_features
from entity_resolution.models.logistic import LogisticRegressionConfig, fit_logistic_regression, predict_match_probabilities
from entity_resolution.models.experiments import threshold_sweep

def main():
    base = Path("/opt/v2-diag")
    
    print("Loading train pairs...")
    train_pairs = pd.read_csv(base / "train_sample_candidate_pairs.tsv", sep="\t", dtype=str)
    num_train = len(train_pairs)
    
    with open(base / "features.pkl", "rb") as f:
        extractor = pickle.load(f)
    feature_order = list(extractor.feature_order)
    
    print("Loading mmaps...")
    train_feat_mmap = np.memmap(base / "training_features.float32.mmap", dtype=np.float32, mode="r", shape=(num_train, 43))
    train_labels = np.memmap(base / "training_labels.uint8.mmap", dtype=np.uint8, mode="r", shape=(num_train,))
    
    print("Loading competition stats...")
    stats = pd.read_parquet(base / "record_stats_trainval.parquet")
    train_features = pd.DataFrame(train_feat_mmap, columns=feature_order)
    train_features["source1_entity_id"] = train_pairs["source1_entity_id"]
    train_features["candidate_entity_id"] = train_pairs["candidate_entity_id"]
    
    print("Joining stats...")
    train_joined, final_cols = add_competition_features(
        pair_features=train_features,
        candidate_pairs=train_pairs,
        competition_stats=stats
    )
    
    # Split into train/val
    split_idx = 400000
    
    train_df = train_joined.iloc[:split_idx]
    y_train = np.array(train_labels[:split_idx])
    X_train = train_df[final_cols].to_numpy()
    
    val_df = train_joined.iloc[split_idx:].copy()
    y_val = np.array(train_labels[split_idx:])
    X_val = val_df[final_cols].to_numpy()
    
    print("Training Logistic Regression...")
    lr_config = LogisticRegressionConfig(c=1.0)
    lr_model = fit_logistic_regression(train_df, pd.Series(y_train), final_cols, lr_config)
    
    print("Training HistGradientBoostingClassifier...")
    gbdt_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    gbdt_model.fit(X_train, y_train)
    
    print("Evaluating models...")
    
    lr_probs = predict_match_probabilities(lr_model, val_df, final_cols)
    gbdt_probs = gbdt_model.predict_proba(X_val)[:, 1]
    
    # Aggregate val truth into comma-separated list of matches
    val_truth = (
        val_df[y_val == 1]
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda x: ",".join(x))
        .reset_index(name="matched_entity_ids")
    )
    val_ids = pd.Series(val_df["source1_entity_id"].unique())
    
    lr_scores = pd.DataFrame({
        "source1_entity_id": val_df["source1_entity_id"],
        "candidate_entity_id": val_df["candidate_entity_id"],
        "match_probability": lr_probs
    })
    
    gbdt_scores = pd.DataFrame({
        "source1_entity_id": val_df["source1_entity_id"],
        "candidate_entity_id": val_df["candidate_entity_id"],
        "match_probability": gbdt_probs
    })
    
    from entity_resolution.models.experiments import BoundedExperimentConfig
    config = BoundedExperimentConfig(threshold_start=0.50, threshold_stop=0.95, threshold_step=0.05)
    
    lr_sweep = threshold_sweep(lr_scores, val_truth, val_ids, "match_probability", config)
    lr_best = lr_sweep.loc[lr_sweep["macro_f05"].idxmax()].to_dict()
    
    gbdt_sweep = threshold_sweep(gbdt_scores, val_truth, val_ids, "match_probability", config)
    gbdt_best = gbdt_sweep.loc[gbdt_sweep["macro_f05"].idxmax()].to_dict()
    
    report = {
        "diagnostic_bounded_evaluation": True,
        "logistic_regression": lr_best,
        "hist_gradient_boosting": gbdt_best
    }
    
    Path("/opt/v2-diag/report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

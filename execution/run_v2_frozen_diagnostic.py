import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

from entity_resolution.config import DATA_DIR
from entity_resolution.models.experiments import (
    BoundedExperimentConfig,
    read_candidate_groups,
    read_source1_ids,
    select_ground_truth,
    labels_for_candidates,
    threshold_sweep,
    sha256_file
)
from entity_resolution.models.logistic import LogisticRegressionConfig, fit_logistic_regression, predict_match_probabilities
from entity_resolution.features.pairwise import PairwiseFeatureExtractor
from entity_resolution.features.competition import add_competition_features
from entity_resolution.matching.records import SQLiteRecordStore

def _joined(record_adapter, pairs, batch_size):
    results = []
    for joined, _ in record_adapter.iter_joined_batches(pairs, batch_size):
        results.append(joined)
    return pd.concat(results, ignore_index=True)

def main():
    base = Path("D:/Amazon-ML-Challenge/artifacts/blocking/BLK-020_word_unigram_69a91f1")
    store_path = Path("D:/Amazon-ML-Challenge/.tmp/train_records.sqlite")
    stats_path = Path("D:/Amazon-ML-Challenge/record_stats_trainval.parquet")
    
    print("Loading train pairs...")
    train_pairs = pd.read_csv(base / "train_sample_candidate_pairs.tsv", sep="\t", dtype=str)
    
    print("Resolving train ground truth...")
    ground_truth = pd.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype="string")
    train_s1_ids = pd.Series(train_pairs["source1_entity_id"].unique())
    train_truth = select_ground_truth(ground_truth, train_s1_ids)
    train_labels = labels_for_candidates(train_pairs, train_truth)
    
    print("Extracting features for training sample...")
    record_adapter = SQLiteRecordStore(store_path)
    train_resolved = _joined(record_adapter, train_pairs, batch_size=5000)
    extractor = PairwiseFeatureExtractor().fit(train_resolved)
    train_features_base = extractor.transform(train_resolved)
    
    print("Joining competition stats...")
    stats = pd.read_parquet(stats_path)
    train_joined, final_cols = add_competition_features(
        pair_features=train_features_base,
        candidate_pairs=train_pairs,
        competition_stats=stats
    )
    
    print("Training models on complete frozen-train S1 groups...")
    X_train = train_joined[final_cols].to_numpy()
    y_train = np.array(train_labels)
    
    lr_config = LogisticRegressionConfig(c=1.0)
    lr_model = fit_logistic_regression(train_joined, pd.Series(y_train), final_cols, lr_config)
    
    gbdt_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    gbdt_model.fit(X_train, y_train)
    
    print("Loading reproducible subset of validation S1 IDs...")
    val_ids_all = read_source1_ids(Path("D:/Amazon-ML-Challenge/artifacts/validation_split/val_ids.csv"))
    val_ids = val_ids_all.iloc[:5000].reset_index(drop=True)
    print(f"Testing on {len(val_ids)} S1 validation entities.")
    
    print("Extracting validation candidate pairs from stream...")
    val_candidates = read_candidate_groups(base / "validation_candidate_pairs.tsv.gz", val_ids, max_pairs=1_000_000)
    print(f"Extracted {len(val_candidates)} candidate pairs.")
    
    print("Resolving val ground truth...")
    val_truth = select_ground_truth(ground_truth, val_ids)
    
    print("Extracting features for validation pairs...")
    val_resolved = _joined(record_adapter, val_candidates, batch_size=5000)
    val_features_base = extractor.transform(val_resolved)
    
    print("Joining competition stats to validation pairs...")
    val_features_joined, _ = add_competition_features(
        pair_features=val_features_base,
        candidate_pairs=val_candidates,
        competition_stats=stats
    )
    X_val = val_features_joined[final_cols].to_numpy()
    
    print("Predicting match probabilities...")
    lr_probs = predict_match_probabilities(lr_model, val_features_joined, final_cols)
    gbdt_probs = gbdt_model.predict_proba(X_val)[:, 1]
    
    lr_scores = pd.DataFrame({
        "source1_entity_id": val_features_joined["source1_entity_id"].astype("string"),
        "candidate_entity_id": val_features_joined["candidate_entity_id"].astype("string"),
        "match_probability": lr_probs
    })
    
    gbdt_scores = pd.DataFrame({
        "source1_entity_id": val_features_joined["source1_entity_id"].astype("string"),
        "candidate_entity_id": val_features_joined["candidate_entity_id"].astype("string"),
        "match_probability": gbdt_probs
    })
    
    print("Sweeping thresholds and calculating official F0.5...")
    config = BoundedExperimentConfig(threshold_start=0.50, threshold_stop=0.95, threshold_step=0.05)
    
    lr_sweep = threshold_sweep(lr_scores, val_truth, val_ids, "match_probability", config)
    lr_best = lr_sweep.loc[lr_sweep["macro_f05"].idxmax()].to_dict()
    
    gbdt_sweep = threshold_sweep(gbdt_scores, val_truth, val_ids, "match_probability", config)
    gbdt_best = gbdt_sweep.loc[gbdt_sweep["macro_f05"].idxmax()].to_dict()
    
    report = {
        "diagnostic_frozen_validation": True,
        "val_entities": len(val_ids),
        "val_ids_hash": sha256_file(Path("D:/Amazon-ML-Challenge/artifacts/validation_split/val_ids.csv")),
        "logistic_regression": lr_best,
        "hist_gradient_boosting": gbdt_best
    }
    
    print(json.dumps(report, indent=2))
    Path("D:/Amazon-ML-Challenge/.tmp/v2_frozen_diagnostic_report.json").write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()


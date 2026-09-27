import argparse
import subprocess
import time
import sys
from pathlib import Path
import json

def run_cmd(cmd, check=True):
    print(f"\n=> Running: {cmd}")
    subprocess.run(cmd, shell=True, check=check)

def poll_s3_status(s3_path, expected_file, interval=30):
    print(f"Polling {s3_path} for {expected_file}...")
    while True:
        res = subprocess.run(f"aws s3 ls {s3_path}", shell=True, capture_output=True, text=True)
        if expected_file in res.stdout:
            print(f"Found {expected_file}!")
            return True
        if "FAILED" in res.stdout or "TIMEOUT" in res.stdout:
            print(f"ERROR: Found FAILED/TIMEOUT status in {s3_path}:\n{res.stdout}")
            return False
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Run ID e.g. mudit-final/runs/train_XXX")
    parser.add_argument("--bucket", default="amazon-ml-challenge-2026-data-655285961749")
    parser.add_argument("--n-parts", type=int, default=10)
    args = parser.parse_args()

    s3_status_dir = f"s3://{args.bucket}/{args.run_id}/status/"
    
    # 1. Wait for train.DONE
    success = poll_s3_status(s3_status_dir, "train.DONE")
    if not success:
        sys.exit(1)
        
    print("\n--- VERIFYING MODEL ARTIFACTS ---")
    models_dir = f"s3://{args.bucket}/{args.run_id}/models/"
    for artifact in ["features.pkl", "logistic_regression.pkl"]:
        res = subprocess.run(f"aws s3 ls {models_dir}{artifact}", shell=True, capture_output=True, text=True)
        if artifact not in res.stdout:
            print(f"ERROR: {artifact} is missing in {models_dir}!")
            sys.exit(1)
    print("Model artifacts verified.")
        
    # 2. Launch validation fanout
    print("\n--- LAUNCHING VALIDATION FANOUT ---")
    run_cmd(f"python scripts/aws/launch_validation.py --run-id {args.run_id} --split validation --n-parts {args.n_parts}")
    
    # 3. Wait for all parts to finish
    print("\n--- WAITING FOR FANOUT PARTS ---")
    for i in range(1, args.n_parts + 1):
        part_flag = f"part{i:02d}.DONE"
        success = poll_s3_status(s3_status_dir, part_flag, interval=20)
        if not success:
            sys.exit(1)
            
    print("\n--- DOWNLOADING AND MERGING PARTS ---")
    local_parts_dir = Path(f".tmp/runs/{args.run_id}/parts")
    local_parts_dir.mkdir(parents=True, exist_ok=True)
    
    run_cmd(f"aws s3 cp s3://{args.bucket}/{args.run_id}/parts/ {local_parts_dir}/ --recursive")
    
    merged_path = Path(f".tmp/runs/{args.run_id}/merged_validation_scores.tsv")
    with open(merged_path, "wb") as dst:
        header_written = False
        for i in range(1, args.n_parts + 1):
            part_tsv = local_parts_dir / f"part{i:02d}" / "validation_logistic_scores.tsv"
            with open(part_tsv, "rb") as src:
                header = src.readline()
                if not header_written:
                    dst.write(header)
                    header_written = True
                while True:
                    chunk = src.read(1024 * 1024 * 16)
                    if not chunk:
                        break
                    dst.write(chunk)
                    
    print(f"\nMerged TSV written to {merged_path}")
    
    print("\n--- RUNNING THRESHOLD & K SWEEP ---")
    sweep_dir = Path(f".tmp/runs/{args.run_id}/sweep_results")
    run_cmd(f"python execution/run_validation_sweep.py "
            f"--scores {merged_path} "
            f"--candidates artifacts/handoff/BLK-020/validation_candidate_pairs.tsv.gz "
            f"--source1-ids output/validation_split/val_ids.csv "
            f"--ground-truth 6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv "
            f"--run-dir {sweep_dir}")
            
    print("\n--- RUNNING ADVANCED DECISION EXPERIMENTS ---")
    run_cmd(f"python execution/advanced_decisions.py "
            f"--scores {merged_path} "
            f"--stats artifacts/blocking/BLK-020_word_unigram_69a91f1/record_stats_trainval.parquet "
            f"--source1-ids output/validation_split/val_ids.csv "
            f"--ground-truth 6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv "
            f"--run-dir {sweep_dir}/adv_k200")
            
    run_cmd(f"python execution/advanced_decisions.py "
            f"--scores {sweep_dir}/scores_K100.tsv "
            f"--stats artifacts/blocking/BLK-020_word_unigram_69a91f1/record_stats_trainval.parquet "
            f"--source1-ids output/validation_split/val_ids.csv "
            f"--ground-truth 6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv "
            f"--run-dir {sweep_dir}/adv_k100")

    print("\n--- SELECTING BEST CONFIGURATION ---")
    all_configs = []

    # Parse basic sweep
    with open(sweep_dir / "sweep_results.json") as f:
        basic_results = json.load(f)
        for res in basic_results:
            all_configs.append({
                "type": "basic",
                "k": res["k"],
                "threshold": res["threshold"],
                "name": "basic",
                "margin": "N/A",
                "f05": res["metrics"]["macro_f05"],
                "precision": res.get("metrics", {}).get("macro_precision", 0.0),
                "recall": res.get("metrics", {}).get("macro_recall", 0.0),
                "fp": res.get("metrics", {}).get("total_fp", 0),
                "fn": res.get("metrics", {}).get("total_fn", 0),
                "tsv_path": str(res["decisions_file"]),
                "config_data": {"type": "basic", "k": res["k"], "threshold": res["threshold"]}
            })

    # Parse advanced sweeps
    for k_val in [200, 100]:
        adv_path = sweep_dir / f"adv_k{k_val}" / "advanced_sweep_results.json"
        if adv_path.exists():
            with open(adv_path) as f:
                adv_results = json.load(f)
                for key, res in adv_results.items():
                    margin = "N/A"
                    if "margin" in key or "gap" in key:
                        try:
                            margin = key.split("_gap_")[-1]
                        except:
                            pass
                            
                    all_configs.append({
                        "type": "advanced",
                        "k": k_val,
                        "threshold": "N/A",  # Complex threshold
                        "name": key,
                        "margin": margin,
                        "f05": res.get("macro_f05", 0.0),
                        "precision": res.get("macro_precision", 0.0),
                        "recall": res.get("macro_recall", 0.0),
                        "fp": res.get("total_fp", 0),
                        "fn": res.get("total_fn", 0),
                        "tsv_path": str(sweep_dir / f"adv_k{k_val}" / f"adv_{key}.tsv"),
                        "config_data": {"type": "advanced", "name": key, "k": k_val}
                    })

    # Sort and pick top 10
    all_configs.sort(key=lambda x: x["f05"], reverse=True)
    top10 = all_configs[:10]
    best_config = top10[0]
    
    # Calculate baseline F0.5 (let's say basic K=200, T=0.60 as a common baseline)
    baseline_f05 = next((c["f05"] for c in all_configs if c["type"] == "basic" and c["k"] == 200 and round(float(c["threshold"]), 2) == 0.60), 0.0)

    print(f"\nOverall Best Configuration: {best_config['name']} (K={best_config['k']}) with F0.5={best_config['f05']:.5f}")
    
    # Generate Top 10 Report
    report = "### Validation Sweep Top 10 Configurations\\n\\n"
    report += "| Rank | Strategy | K | Threshold | Margin | F0.5 | Precision | Recall | FP | FN | +vs Baseline |\\n"
    report += "|---|---|---|---|---|---|---|---|---|---|---|\\n"
    
    # Identify bests
    best_precision = max(all_configs, key=lambda x: x["precision"])
    best_recall = max(all_configs, key=lambda x: x["recall"])
    baseline_config = next((c for c in all_configs if c["type"] == "basic" and c["k"] == 200 and round(float(c["threshold"]), 2) == 0.60), None)
    
    for i, c in enumerate(top10):
        improvement = c["f05"] - 0.8928  # Fixed baseline
        report += f"| {i+1} | {c['name']} | {c['k']} | {c['threshold']} | {c['margin']} | **{c['f05']:.5f}** | {c['precision']:.5f} | {c['recall']:.5f} | {c['fp']} | {c['fn']} | {improvement:+.5f} |\\n"
        
    report += "\\n### Compact Summary\\n"
    report += f"- **Best F0.5 Config:** {best_config['name']} (K={best_config['k']}, T={best_config['threshold']}, F0.5={best_config['f05']:.5f})\\n"
    report += f"- **Best Precision Config:** {best_precision['name']} (K={best_precision['k']}, T={best_precision['threshold']}, Precision={best_precision['precision']:.5f})\\n"
    report += f"- **Best Recall Config:** {best_recall['name']} (K={best_recall['k']}, T={best_recall['threshold']}, Recall={best_recall['recall']:.5f})\\n"
    report += f"- **Difference from 0.8928 baseline:** {best_config['f05'] - 0.8928:+.5f}\\n"
    
    if baseline_config:
        fp_reduction = baseline_config["fp"] - best_config["fp"]
        fn_reduction = baseline_config["fn"] - best_config["fn"]
        report += f"- **FP Reduction vs Baseline (K=200, T=0.60):** {fp_reduction:+d} false positives\\n"
        report += f"- **FN Reduction vs Baseline (K=200, T=0.60):** {fn_reduction:+d} false negatives\\n"
    
    print(report.replace("\\n", "\n"))
    with open("artifacts/top10_report.md", "w") as f:
        f.write(report.replace("\\n", "\n"))
    
    with open(sweep_dir / "best_config.json", "w") as f:
        json.dump(best_config["config_data"], f)
    
    final_output = sweep_dir / "matching_results.tsv"
    run_cmd(f"cp {best_config['tsv_path']} {final_output}")
            
    print("\n--- VALIDATION PIPELINE COMPLETE ---")
    print(f"Final best submission configuration saved to: {final_output}")
    print("Waiting for manual inspection of Top 10 Report before test inference.")

if __name__ == "__main__":
    main()

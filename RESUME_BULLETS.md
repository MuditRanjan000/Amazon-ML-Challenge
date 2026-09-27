# Resume Bullet Points

- **Machine Learning Engineer - Amazon ML Challenge 2026 (Top Percentile Finish)**
- Engineered a highly scalable entity resolution and deduplication pipeline processing 12M+ multilingual business records, achieving a peak Macro-F0.5 score of 0.967.
- Architected a memory-bounded Sparse TF-IDF blocking engine utilizing PyArrow and `sparse_dot_topn`, achieving 97.8% recall at K=200 while shrinking the O(N²) search space by 99.9% without out-of-memory errors.
- Designed a two-stage ML ensemble (Stage 1 base models + Stage 2 context stacker) with adaptive entity-level decision thresholds.
- Optimized precision-heavy evaluation metrics by designing and implementing targeted "lookalike distractor" pruning algorithms, directly elevating the final leaderboard score from 0.959 to 0.967.
- Enforced rigorous MLOps practices, including frozen validation splits with SHA-256 integrity checks, preventing entity leakage and ensuring reliable offline-to-online metric correlation.

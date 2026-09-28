# Data Directory

Raw data is never committed (`*.tsv`, `*.csv`, `*.parquet` are gitignored).
Place the challenge resource at `<repo>/6ab10eb3b23ba_student_resource/`, or set `ER_DATA_DIR` to the folder containing `train/` and `test/`.
Parsed parquet copies are cached automatically under `output/cache/`. They are rebuilt if a TSV is newer.

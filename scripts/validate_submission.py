"""Validate output files with the organisers' official validator.

  python scripts/validate_submission.py                       # output/matching_results.tsv + candidate_pairs.tsv
  python scripts/validate_submission.py --check-ids           # also verify every ID exists (more RAM)
"""
import argparse
import sys

from entity_resolution import config
from entity_resolution.submission.validator import SubmissionValidator


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submission", "--matching", dest="matching", default=str(config.OUTPUT_DIR / "matching_results.tsv"))
    p.add_argument("--candidate", default=str(config.OUTPUT_DIR / "candidate_pairs.tsv"))
    p.add_argument("--test-dir", default=str(config.DATA_DIR / "test"))
    p.add_argument("--check-ids", action="store_true")
    a = p.parse_args()
    ok = SubmissionValidator(a.test_dir).validate(a.matching, a.candidate, check_ids=a.check_ids)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

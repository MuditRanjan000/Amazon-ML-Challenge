import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.entity_resolution.submission.validator import SubmissionValidator

def main():
    parser = argparse.ArgumentParser(description="Validate submission TSV for Amazon ML Challenge 2026")
    parser.add_argument("--submission", required=True, help="Path to the generated submission file (e.g., output/matching_results.tsv)")
    parser.add_argument("--test-dir", required=True, help="Directory containing the test TSV files")
    
    args = parser.parse_args()
    
    validator = SubmissionValidator(test_dir=args.test_dir)
    is_valid = validator.validate(args.submission)
    
    if not is_valid:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

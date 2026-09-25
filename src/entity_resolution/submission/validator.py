import pandas as pd
import os

class SubmissionValidator:
    def __init__(self, test_dir: str):
        self.test_dir = test_dir
        
    def validate(self, submission_path: str) -> bool:
        """Validate submission file against official rules."""
        if not os.path.exists(submission_path):
            print(f"Error: Submission file {submission_path} does not exist.")
            return False
            
        print("Loading test datasets for validation...")
        s1_path = os.path.join(self.test_dir, "test_source1.tsv")
        s2_path = os.path.join(self.test_dir, "test_source2.tsv")
        s3_path = os.path.join(self.test_dir, "test_source3.tsv")
        
        test_s1 = pd.read_csv(s1_path, sep='\t', usecols=['entity_id'], dtype=str)
        test_s2 = pd.read_csv(s2_path, sep='\t', usecols=['entity_id'], dtype=str)
        test_s3 = pd.read_csv(s3_path, sep='\t', usecols=['entity_id'], dtype=str)
        
        valid_s1_ids = set(test_s1['entity_id'])
        valid_s2_ids = set(test_s2['entity_id'])
        valid_s3_ids = set(test_s3['entity_id'])
        valid_target_ids = valid_s2_ids.union(valid_s3_ids)
        
        print("Loading submission file...")
        try:
            sub = pd.read_csv(submission_path, sep='\t', dtype=str)
        except Exception as e:
            print(f"Error: Invalid TSV format - {str(e)}")
            return False
            
        # Check columns
        expected_cols = ['source1_entity_id', 'matched_entity_ids']
        if list(sub.columns) != expected_cols:
            print(f"Error: Columns must be {expected_cols}. Found {list(sub.columns)}")
            return False
            
        # Check every Source 1 entity exists
        sub_s1_ids = set(sub['source1_entity_id'])
        if sub_s1_ids != valid_s1_ids:
            missing = valid_s1_ids - sub_s1_ids
            extra = sub_s1_ids - valid_s1_ids
            print(f"Error: Source 1 entity mismatch. Missing: {len(missing)}, Extra: {len(extra)}")
            return False
            
        # Check no duplicate Source 1 rows
        if len(sub['source1_entity_id']) != len(sub_s1_ids):
            print("Error: Duplicate source1_entity_id rows found.")
            return False
            
        errors = 0
        for idx, row in sub.iterrows():
            s1 = row['source1_entity_id']
            matches_str = row['matched_entity_ids']
            
            if pd.isna(matches_str) or matches_str.strip() == "":
                continue
                
            matches = matches_str.split(',')
            
            # Check duplicates in match list
            if len(matches) != len(set(matches)):
                print(f"Error on row {idx}: Duplicate IDs in match list.")
                errors += 1
                
            # Check S1 self match
            if s1 in matches:
                print(f"Error on row {idx}: Source 1 self-match found.")
                errors += 1
                
            # Check IDs exist in S2/S3
            for m in matches:
                if m not in valid_target_ids:
                    print(f"Error on row {idx}: Target ID {m} not found in Source 2 or 3.")
                    errors += 1
                    
            if errors > 10:
                print("Too many errors. Aborting validation.")
                return False
                
        if errors > 0:
            return False
            
        print("Validation PASSED! The submission file is valid and compliant.")
        return True

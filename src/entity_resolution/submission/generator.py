import pandas as pd
import os

class SubmissionGenerator:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate(self, test_s1_ids: set, predicted_matches: dict, filename: str = "matching_results.tsv"):
        """
        Generate the final submission TSV.
        test_s1_ids: Set of all Source 1 entity IDs in the test set.
        predicted_matches: Dict mapping Source1 ID to a list/set of matched S2/S3 IDs.
        """
        output_path = os.path.join(self.output_dir, filename)
        
        results = []
        for s1_id in test_s1_ids:
            matches = predicted_matches.get(s1_id, [])
            # Rule 4: No duplicate IDs
            unique_matches = list(set(matches))
            # Rule 3: Only S2/S3 IDs (we assume predicted_matches correctly holds these)
            
            # Rule 2: If no matches, matched_entity_ids must be empty
            matches_str = ",".join(unique_matches) if len(unique_matches) > 0 else ""
            
            results.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": matches_str
            })
            
        # Rule 1: Every Source1 test entity must have exactly one row
        df = pd.DataFrame(results)
        
        # Rule 5: Preserve TSV formatting exactly
        df.to_csv(output_path, sep='\t', index=False)
        print(f"Submission generated at {output_path}")
        return output_path

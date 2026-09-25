import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_source(self, split: str, source: int) -> pd.DataFrame:
        """
        Load specific source file memory-efficiently.
        split: 'train' or 'test'
        source: 1, 2, or 3
        """
        file_path = os.path.join(self.data_dir, split, f"{split}_source{source}.tsv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found: {file_path}")
            
        df = pd.read_csv(file_path, sep='\t', dtype={
            'entity_id': 'string',
            'business_name': 'string',
            'business_address': 'string',
            'country': 'string'
        })
        return df
        
    def load_ground_truth(self) -> pd.DataFrame:
        """Load ground truth matching labels for the training set."""
        file_path = os.path.join(self.data_dir, "train", "train_ground_truth.tsv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ground truth file not found: {file_path}")
            
        df = pd.read_csv(file_path, sep='\t', dtype={
            'source1_entity_id': 'string',
            'matched_entity_ids': 'string'
        })
        return df

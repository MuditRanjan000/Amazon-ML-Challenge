import pandas as pd
from sklearn.model_selection import train_test_split

def create_validation_split(ground_truth_df: pd.DataFrame, test_size=0.2, random_state=42) -> tuple:
    """
    Split the dataset based on Source 1 entity IDs ensuring no leakage.
    Returns (train_s1_ids, val_s1_ids) as sets.
    """
    unique_s1_ids = ground_truth_df['source1_entity_id'].unique()
    train_ids, val_ids = train_test_split(unique_s1_ids, test_size=test_size, random_state=random_state)
    return set(train_ids), set(val_ids)

def apply_validation_split(df: pd.DataFrame, s1_id_col: str, train_ids: set, val_ids: set) -> tuple:
    """
    Split a DataFrame containing a source1 entity ID column into train and validation subsets.
    """
    train_df = df[df[s1_id_col].isin(train_ids)].copy()
    val_df = df[df[s1_id_col].isin(val_ids)].copy()
    return train_df, val_df

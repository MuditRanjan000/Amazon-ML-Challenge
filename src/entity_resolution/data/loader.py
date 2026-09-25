class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_source(self, split: str, source: int):
        """Load specific source file memory-efficiently."""
        raise NotImplementedError

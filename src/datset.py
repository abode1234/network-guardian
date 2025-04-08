import pandas as pd
from .utils import reduce_memory_usage

class DatasetLoader:
    def __init__(self, logger):
        self.logger = logger
    
    def load_in_chunks(self, file_path, chunksize=500000):
        """Load large dataset in chunks."""
        self.logger.info(f"Loading data from {file_path}")
        chunks = []
        
        total_rows = sum(1 for _ in open(file_path, 'r')) - 1
        
        with tqdm(total=total_rows) as pbar:
            for chunk in pd.read_csv(file_path, chunksize=chunksize):
                chunk = reduce_memory_usage(chunk, self.logger)
                chunks.append(chunk)
                pbar.update(len(chunk))
                
        df = pd.concat(chunks, ignore_index=True)
        self.logger.info(f"Loaded {df.shape[0]} rows with {df.shape[1]} features")
        
        return df

from pathlib import Path
import pickle
from typing import List

from rank_bm25 import BM25Okapi as BM25

class CachedBM25(BM25):
    def __init__(self, cache_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_path = cache_path
        self.save(cache_path)

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"No cached model found at {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)

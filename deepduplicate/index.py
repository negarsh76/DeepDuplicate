"""Nearest-neighbour index over L2-normalised embeddings (cosine = inner product)."""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class VectorIndex:
    """FAISS when installed (flat for < 50k vectors, HNSW above), exact NumPy search otherwise."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = self.vectors.shape
        self.backend = "numpy"
        if faiss is not None and n:
            if n < 50_000:
                self.index = faiss.IndexFlatIP(d)
            else:
                self.index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efSearch = 128
            self.index.add(self.vectors)
            self.backend = f"faiss:{type(self.index).__name__}"

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        k = min(k, len(self.vectors))
        if self.backend.startswith("faiss"):
            return self.index.search(q, k)
        sims = q @ self.vectors.T
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        part = np.take_along_axis(sims, idx, 1)
        order = np.argsort(-part, axis=1)
        return np.take_along_axis(part, order, 1), np.take_along_axis(idx, order, 1)

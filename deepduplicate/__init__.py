"""DeepDuplicate — find exact, near and visual duplicates in large image collections.

A cascade of increasingly expensive checks:

1. **Embedding search** (OpenCLIP / DINOv2 / ResNet, or an offline classic descriptor)
   with an approximate-nearest-neighbour index (FAISS, NumPy fallback).
2. **Perceptual hash** (pHash, rotation-aware) and **SSIM** re-ranking.
3. **Geometric verification** with ORB/SIFT keypoints + RANSAC homography.

Nothing is ever deleted: duplicates are reported (CSV + HTML gallery) and can
optionally be *moved* to a quarantine folder.
"""
__version__ = "1.0.0"

from .pipeline import DuplicateFinder, Match  # noqa: F401

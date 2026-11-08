"""Pixel-level checks: SSIM and keypoint geometric verification."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _gray(im: Image.Image, size: int) -> np.ndarray:
    g = im.convert("L")
    g.thumbnail((size, size), Image.LANCZOS)
    return np.asarray(g)


def ssim(a: Image.Image, b: Image.Image, size: int = 256) -> Optional[float]:
    """SSIM after resizing both images to the same grid (scikit-image)."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError:  # pragma: no cover
        return None
    ga = _gray(a, size)
    gb = np.asarray(Image.fromarray(_gray(b, size)).resize(ga.shape[::-1], Image.LANCZOS))
    return float(structural_similarity(ga, gb, data_range=255))


def geometric_match(a: Image.Image, b: Image.Image, size: int = 640, min_inliers: int = 15,
                    ratio: float = 0.75, use_sift: bool = True) -> Tuple[bool, int]:
    """Keypoints + Lowe ratio test + RANSAC homography. Robust to crops, rotation and scale.

    Returns ``(is_match, n_inliers)``.
    """
    if cv2 is None:
        return False, 0
    ga, gb = _gray(a, size), _gray(b, size)
    det = None
    if use_sift and hasattr(cv2, "SIFT_create"):
        det, norm = cv2.SIFT_create(nfeatures=2000), cv2.NORM_L2
    if det is None:
        det, norm = cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING
    ka, da = det.detectAndCompute(ga, None)
    kb, db = det.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return False, 0
    pairs = cv2.BFMatcher(norm).knnMatch(da, db, k=2)
    good = [m for m, *n in pairs if n and m.distance < ratio * n[0].distance]
    if len(good) < 8:
        return False, len(good)
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers >= min_inliers, inliers

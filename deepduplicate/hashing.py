"""Rotation-aware perceptual hashing."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PIL import Image, ImageOps

try:
    import imagehash
except ImportError:  # pragma: no cover
    imagehash = None

ROTATIONS = (0, 90, 180, 270)


def _normalise(im: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(ImageOps.grayscale(im))


def phash(im: Image.Image, hash_size: int = 8) -> int:
    """64-bit pHash as an int. Uses ``imagehash`` if available, else a NumPy DCT."""
    g = _normalise(im)
    if imagehash is not None:
        return int(str(imagehash.phash(g, hash_size=hash_size)), 16)
    from scipy.fft import dctn
    a = np.asarray(g.resize((hash_size * 4, hash_size * 4), Image.LANCZOS), dtype=np.float64)
    d = dctn(a, norm="ortho")[:hash_size, :hash_size]
    bits = (d > np.median(d)).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def phash_rotations(im: Image.Image, flips: bool = True) -> List[int]:
    """Hashes of the image under 0/90/180/270° rotation (and mirror), for orientation-robust matching."""
    variants = [im.rotate(-r, expand=True) for r in ROTATIONS]
    if flips:
        variants += [ImageOps.mirror(v) for v in variants]
    return [phash(v) for v in variants]


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def min_hamming(query: int, candidates: List[int]) -> int:
    return min(hamming(query, c) for c in candidates)


def phash_or_none(im: Optional[Image.Image]) -> Optional[int]:
    return None if im is None else phash(im)

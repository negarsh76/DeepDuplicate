"""The cascade that turns candidate pairs into confirmed duplicates."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .embed import embed_paths, make_embedder
from .hashing import hamming, phash, phash_rotations
from .index import VectorIndex
from .io import list_images, load_rgb
from .verify import geometric_match, ssim


class HashBands:
    """Multi-index hashing: split 64-bit hashes into 8 bytes. Two hashes within Hamming
    distance 7 must agree on at least one byte (pigeonhole), so bucket look-ups find
    every near-copy without comparing all pairs."""

    def __init__(self, hashes_per_item: Sequence[Sequence[int]]):
        self.buckets: Dict[tuple, set] = {}
        for i, hs in enumerate(hashes_per_item):
            for h in hs:
                for b in range(8):
                    self.buckets.setdefault((b, (h >> (8 * b)) & 0xFF), set()).add(i)

    def candidates(self, h: int) -> set:
        out = set()
        for b in range(8):
            out |= self.buckets.get((b, (h >> (8 * b)) & 0xFF), set())
        return out


@dataclass
class Config:
    backbone: str = "auto"
    top_k: int = 10                 # neighbours examined per image
    cos_candidate: Optional[float] = None  # below this, a pair is never considered (auto: 0.55 classic, 0.75 deep)
    cos_accept: float = 0.96        # above this, accept on embedding alone
    phash_accept: int = 6           # Hamming distance (of 64 bits) for a near-copy
    ssim_accept: float = 0.82       # pixel structure check for the grey zone
    min_inliers: int = 20           # RANSAC inliers for crop / rotate / perspective edits
    geometric: bool = True
    max_edge: int = 1024            # images are downscaled to this before any check


@dataclass
class Match:
    query: str
    match: str
    kind: str                       # exact | near | visual | edited
    cosine: float
    phash_dist: Optional[int] = None
    ssim: Optional[float] = None
    inliers: Optional[int] = None
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


class DuplicateFinder:
    """Find duplicates inside one collection, or between new images and a reference collection.

    >>> finder = DuplicateFinder(Config(backbone="classic"))
    >>> matches = finder.find_within("photos/")
    """

    def __init__(self, config: Optional[Config] = None, embedder=None, progress: bool = True):
        self.cfg = config or Config()
        self.embedder = embedder or make_embedder(self.cfg.backbone)
        if self.cfg.cos_candidate is None:
            self.cfg.cos_candidate = 0.55 if getattr(self.embedder, "name", "") == "classic" else 0.75
        self.progress = progress
        self._cache: Dict[Path, object] = {}

    # ------------------------------------------------------------------ helpers
    def _img(self, p: Path):
        if p not in self._cache:
            self._cache[p] = load_rgb(p, self.cfg.max_edge)
        return self._cache[p]

    def _embed(self, paths: Sequence[Path]) -> np.ndarray:
        return embed_paths(list(paths), self.embedder, self._img, progress=self.progress)

    def _hashes(self, paths):
        ims = [self._img(p) for p in paths]
        h0 = [phash(im) if im is not None else 0 for im in ims]
        hr = [phash_rotations(im) if im is not None else [] for im in ims]
        flat = [im is None or float(np.asarray(im.convert("L"), float).std()) < 12 for im in ims]
        return h0, hr, flat

    def _hash_pairs(self, h0, hr, flat, ref_hr=None) -> Dict[tuple, int]:
        """(query, ref) → min Hamming distance, for every pair within ``phash_accept``."""
        ref_hr = hr if ref_hr is None else ref_hr
        bands = HashBands(ref_hr)
        out = {}
        for i, h in enumerate(h0):
            if flat[i]:
                continue
            for j in bands.candidates(h):
                if ref_hr is hr and j == i:
                    continue
                d = min(hamming(h, x) for x in ref_hr[j])
                if d <= self.cfg.phash_accept:
                    out[(i, j)] = d
        return out

    def _judge(self, q: Path, r: Path, cos: float, sha: Dict[Path, str]) -> Optional[Match]:
        m = Match(str(q), str(r), "", float(cos))
        if sha.get(q) and sha.get(q) == sha.get(r):
            m.kind, m.evidence = "exact", ["identical bytes (SHA-1)"]
            return m
        a, b = self._img(q), self._img(r)
        if a is None or b is None:
            return None
        m.phash_dist = min(hamming(phash(a), h) for h in phash_rotations(b))
        flat = min(float(np.asarray(x.convert("L"), float).std()) for x in (a, b)) < 12
        if m.phash_dist <= self.cfg.phash_accept and not flat:
            m.kind = "near"; m.evidence.append(f"pHash distance {m.phash_dist} ≤ {self.cfg.phash_accept}")
            return m
        if cos >= self.cfg.cos_accept:
            m.kind = "visual"; m.evidence.append(f"cosine {cos:.3f} ≥ {self.cfg.cos_accept}")
            return m
        m.ssim = ssim(a, b)
        if m.ssim is not None and m.ssim >= self.cfg.ssim_accept:
            m.kind = "near"; m.evidence.append(f"SSIM {m.ssim:.2f} ≥ {self.cfg.ssim_accept}")
            return m
        if self.cfg.geometric:
            ok, m.inliers = geometric_match(a, b, min_inliers=self.cfg.min_inliers)
            if ok:
                m.kind = "edited"; m.evidence.append(f"{m.inliers} RANSAC inliers (crop / rotate / perspective)")
                return m
        return None

    # ------------------------------------------------------------------ public API
    def find_within(self, folder, paths: Optional[Sequence[Path]] = None) -> List[Match]:
        """All duplicate pairs inside one collection (each pair reported once)."""
        paths = [Path(p) for p in (paths or list_images(folder))]
        if len(paths) < 2:
            return []
        feats = self._embed(paths)
        sims, idx = VectorIndex(feats).search(feats, self.cfg.top_k + 1)
        sha = {p: _sha1(p) for p in paths}
        cand = set()
        for i, (row_s, row_i) in enumerate(zip(sims, idx)):            # 1) embedding neighbours
            cand |= {(min(i, j), max(i, j)) for s, j in zip(row_s, row_i) if j != i and j >= 0 and s >= self.cfg.cos_candidate}
        cand |= {(min(i, j), max(i, j)) for i, j in self._hash_pairs(*self._hashes(paths))}   # 2) hash buckets
        by_hash = {}
        for p in paths:
            by_hash.setdefault(sha[p], []).append(p)
        pos = {p: k for k, p in enumerate(paths)}
        for grp in by_hash.values():                                    # 3) identical files
            cand |= {(pos[a], pos[b]) for a, b in zip(grp, grp[1:])}
        out = []
        for i, j in sorted(cand):
            m = self._judge(paths[j], paths[i], float(feats[i] @ feats[j]), sha)
            if m:
                out.append(m)
        return out

    def find_against(self, new_folder, reference_folder, new_paths=None, ref_paths=None) -> List[Match]:
        """For each image in ``new_folder``, its best confirmed match in ``reference_folder`` (if any)."""
        new = [Path(p) for p in (new_paths or list_images(new_folder))]
        ref = [Path(p) for p in (ref_paths or list_images(reference_folder))]
        if not new or not ref:
            return []
        fr, fn = self._embed(ref), self._embed(new)
        sims, idx = VectorIndex(fr).search(fn, self.cfg.top_k)
        sha = {p: _sha1(p) for p in new + ref}
        n0, _, nflat = self._hashes(new)
        _, rr, _ = self._hashes(ref)
        hash_c = {}
        for (i, j), d in self._hash_pairs(n0, [[]] * len(new), nflat, ref_hr=rr).items():
            hash_c.setdefault(i, []).append(j)
        ref_sha = {}
        for k, p in enumerate(ref):
            ref_sha.setdefault(sha[p], k)
        out = []
        for qi, (row_s, row_i) in enumerate(zip(sims, idx)):
            cands = ([ref_sha[sha[new[qi]]]] if sha[new[qi]] in ref_sha else []) + hash_c.get(qi, [])
            cands += [int(j) for s, j in zip(row_s, row_i) if j >= 0 and s >= self.cfg.cos_candidate]
            for j in dict.fromkeys(cands):
                m = self._judge(new[qi], ref[j], float(fn[qi] @ fr[j]), sha)
                if m:
                    out.append(m); break
        return out


def group_matches(matches: Sequence[Match]) -> List[List[str]]:
    """Union-find: merge pairwise matches into duplicate groups."""
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for m in matches:
        parent[find(m.query)] = find(m.match)
    groups: Dict[str, List[str]] = {}
    for x in parent:
        groups.setdefault(find(x), []).append(x)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=len, reverse=True)

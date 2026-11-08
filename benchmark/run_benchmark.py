"""Score pHash-only, embedding-only and the full cascade on the synthetic benchmark.

    python benchmark/make_benchmark.py         # once
    python benchmark/run_benchmark.py --backbone classic   # or dinov2 / openclip-l14 / resnet50
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepduplicate.hashing import hamming, phash, phash_rotations  # noqa: E402
from deepduplicate.io import load_rgb  # noqa: E402
from deepduplicate.pipeline import Config, DuplicateFinder  # noqa: E402

HERE = Path(__file__).parent


def prf(pred: set, truth: set):
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", default="classic"); a = ap.parse_args()
    labels = json.loads((HERE / "data" / "labels.json").read_text())
    names = sorted(labels); paths = [HERE / "data" / "images" / n for n in names]
    group = [labels[n]["group"] for n in names]; kind = [labels[n]["kind"] for n in names]
    truth = {(i, j) for i, j in itertools.combinations(range(len(names)), 2) if group[i] == group[j]}
    print(f"{len(names)} images, {len(truth)} true duplicate pairs, "
          f"{len(names) * (len(names) - 1) // 2 - len(truth)} non-duplicate pairs")
    results = {}

    # 1) pHash only (rotation-aware), brute force over all pairs
    t = time.time()
    ims = [load_rgb(p, 512) for p in paths]
    h0 = [phash(im) for im in ims]; hr = [phash_rotations(im) for im in ims]
    for thr in (6, 10):
        pred = {(i, j) for i, j in itertools.combinations(range(len(ims)), 2)
                if min(hamming(h0[i], x) for x in hr[j]) <= thr}
        results[f"pHash only (≤{thr})"] = (*prf(pred, truth), time.time() - t, pred)

    # 2) embedding only, best threshold
    finder = DuplicateFinder(Config(backbone=a.backbone), progress=False)
    t = time.time(); F = finder._embed(paths); S = F @ F.T; te = time.time() - t
    best = max(((thr, prf({(i, j) for i, j in itertools.combinations(range(len(names)), 2) if S[i, j] >= thr}, truth))
                for thr in np.arange(0.70, 0.995, 0.005)), key=lambda x: x[1][2])
    pred = {(i, j) for i, j in itertools.combinations(range(len(names)), 2) if S[i, j] >= best[0]}
    results[f"Embedding only ({finder.embedder.name}, best cos={best[0]:.3f})"] = (*best[1], te, pred)

    # 3) full cascade
    t = time.time()
    ms = finder.find_within(None, paths=paths)
    idx = {str(p): i for i, p in enumerate(paths)}
    pred = {tuple(sorted((idx[m.query], idx[m.match]))) for m in ms}
    results[f"Cascade ({finder.embedder.name} → pHash → SSIM → RANSAC)"] = (*prf(pred, truth), time.time() - t, pred)

    # per-transform recall: pair (original, edited copy)
    orig = {group[i]: i for i in range(len(names)) if kind[i] == "original"}
    kinds = [k for k in dict.fromkeys(kind) if k != "original"]
    table = {}
    for name, (p, r, f1, secs, pr) in results.items():
        per = {}
        for k in kinds:
            pairs = [tuple(sorted((orig[group[i]], i))) for i in range(len(names)) if kind[i] == k]
            per[k] = float(np.mean([pp in pr for pp in pairs]))
        table[name] = {"precision": p, "recall": r, "F1": f1, "seconds": secs, "recall_by_edit": per}
        print(f"{name:55s} P={p:.3f} R={r:.3f} F1={f1:.3f}  ({secs:.1f}s)  " + "  ".join(f"{k}={v:.2f}" for k, v in per.items()))
    (HERE / f"results_{a.backbone}.json").write_text(json.dumps(table, indent=2))
    kinds_count = {}
    for m in ms:
        kinds_count[m.kind] = kinds_count.get(m.kind, 0) + 1
    print("cascade decisions by stage:", kinds_count)


if __name__ == "__main__":
    main()

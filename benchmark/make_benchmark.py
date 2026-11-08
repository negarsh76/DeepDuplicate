"""Build a small, fully reproducible near-duplicate benchmark from scikit-image sample images.

Each *original* is a crop from a public-domain sample image. Every original gets
five edited copies, one per transform family below. Crops taken from the same
source image act as **hard negatives**: similar texture and colour, but they are
not duplicates.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from skimage import data

SOURCES = ["chelsea", "coffee", "rocket", "hubble_deep_field", "immunohistochemistry", "retina",
           "brick", "grass", "gravel", "moon", "coins", "page", "horse", "clock", "colorwheel", "logo", "microaneurysms", "text", "cell"]


def _jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")


TRANSFORMS = {
    "recompressed":  lambda im, r: _jpeg(im.resize((im.width // 2, im.height // 2), Image.LANCZOS), 25),
    "color-edited":  lambda im, r: ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(r.uniform(.65, 1.4))).enhance(r.uniform(.7, 1.4)).filter(ImageFilter.GaussianBlur(1.2)),
    "rotated+flipped": lambda im, r: ImageOps.mirror(im.rotate(int(r.choice([90, 180, 270])), expand=True)),
    "cropped":       lambda im, r: (lambda w, h: im.crop((int(w*.12), int(h*.1), int(w*.86), int(h*.92))).resize((w, h)))(*im.size),
    "perspective":   lambda im, r: (lambda w, h: im.rotate(r.uniform(6, 12), resample=Image.BICUBIC).transform(
                         im.size, Image.PERSPECTIVE, (1, .06, -8, .04, 1, -6, .0002, .0001), Image.BICUBIC)
                         .crop((int(w*.15), int(h*.15), int(w*.85), int(h*.85))).resize((w, h), Image.BICUBIC))(*im.size),
}


def to_rgb(a):
    a = np.asarray(a)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    if a.dtype != np.uint8:
        a = (255 * (a - a.min()) / max(np.ptp(a), 1e-9)).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return Image.fromarray(a[..., :3])


def build(out: Path, crops_per_source: int = 5, size: int = 224, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    import random
    r = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    labels = {}
    gid = 0
    for name in SOURCES:
        src = to_rgb(getattr(data, name)())
        if min(src.size) < 3 * size:                                     # room for several disjoint crops
            k = 3 * size / min(src.size)
            src = src.resize((int(src.width * k) + 1, int(src.height * k) + 1), Image.LANCZOS)
        boxes, tries = [], 0
        while len(boxes) < crops_per_source and tries < 500:
            tries += 1
            x = int(rng.integers(0, src.width - size + 1)); y = int(rng.integers(0, src.height - size + 1))
            base = src.crop((x, y, x + size, y + size))
            if np.asarray(base.convert("L"), float).std() < 25:           # skip blank / flat regions
                continue
            if any(abs(x - bx) < size and abs(y - by) < size for bx, by in boxes):
                continue                                                   # crops never overlap
            boxes.append((x, y))
            g = f"g{gid:03d}"; gid += 1
            base.save(out / f"{g}_original.jpg", quality=95); labels[f"{g}_original.jpg"] = {"group": g, "kind": "original"}
            for tname, fn in TRANSFORMS.items():
                fn(base, r).convert("RGB").save(out / f"{g}_{tname}.jpg", quality=92)
                labels[f"{g}_{tname}.jpg"] = {"group": g, "kind": tname}
    (out.parent / "labels.json").write_text(json.dumps(labels, indent=1))
    return labels


if __name__ == "__main__":
    here = Path(__file__).parent
    lab = build(here / "data" / "images")
    print(f"{len(lab)} images in {len(set(v['group'] for v in lab.values()))} duplicate groups")

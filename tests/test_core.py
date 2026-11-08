import numpy as np
from PIL import Image, ImageOps

from deepduplicate.hashing import hamming, phash, phash_rotations
from deepduplicate.index import VectorIndex
from deepduplicate.pipeline import Config, DuplicateFinder, group_matches


def _img(seed=0, size=256):
    rng = np.random.default_rng(seed)
    a = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
    return Image.fromarray(a).resize((size, size), Image.BICUBIC)


def test_phash_is_rotation_and_flip_aware():
    im = _img()
    h = phash(ImageOps.mirror(im.rotate(90, expand=True)))
    assert min(hamming(h, x) for x in phash_rotations(im)) <= 2


def test_phash_separates_different_images():
    assert hamming(phash(_img(0)), phash(_img(1))) > 16


def test_numpy_index_matches_bruteforce():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 8)).astype(np.float32); v /= np.linalg.norm(v, axis=1, keepdims=True)
    s, i = VectorIndex(v).search(v[:5], 3)
    assert (i[:, 0] == np.arange(5)).all() and np.allclose(s[:, 0], 1, atol=1e-5)


def test_pipeline_finds_planted_duplicates(tmp_path):
    for k in range(6):
        _img(k).save(tmp_path / f"img{k}.png")
    _img(2).resize((128, 128)).save(tmp_path / "copy_of_2.jpg", quality=40)          # resized + recompressed
    ImageOps.mirror(_img(4).rotate(180)).save(tmp_path / "flipped_4.png")             # rotated + mirrored
    ms = DuplicateFinder(Config(backbone="classic"), progress=False).find_within(tmp_path)
    groups = [{p.split("/")[-1] for p in g} for g in group_matches(ms)]
    assert {"img2.png", "copy_of_2.jpg"} in groups
    assert {"img4.png", "flipped_4.png"} in groups
    assert len(groups) == 2


def test_against_mode(tmp_path):
    ref, new = tmp_path / "ref", tmp_path / "new"; ref.mkdir(); new.mkdir()
    for k in range(5):
        _img(k).save(ref / f"r{k}.png")
    _img(3).rotate(90, expand=True).save(new / "seen_before.png")
    _img(42).save(new / "brand_new.png")
    ms = DuplicateFinder(Config(backbone="classic"), progress=False).find_against(new, ref)
    assert [(m.query.split("/")[-1], m.match.split("/")[-1]) for m in ms] == [("seen_before.png", "r3.png")]

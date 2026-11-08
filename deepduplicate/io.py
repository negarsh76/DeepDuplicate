"""Image listing and robust loading."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".heic"}

try:  # optional HEIC support
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass


def list_images(folder: str | Path, recursive: bool = True) -> List[Path]:
    folder = Path(folder)
    it: Iterable[Path] = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def load_rgb(path: str | Path, max_edge: Optional[int] = None) -> Optional[Image.Image]:
    """Open an image, honour EXIF orientation, convert to RGB. Returns None if unreadable."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max_edge and max(im.size) > max_edge:
                im.thumbnail((max_edge, max_edge), Image.LANCZOS)
            return im.copy()
    except Exception:
        return None


def move_unique(src: Path, dst_dir: Path) -> Path:
    """Move ``src`` into ``dst_dir`` without overwriting (appends _1, _2 …)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    i = 1
    while dst.exists():
        dst = dst_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    shutil.move(str(src), str(dst))
    return dst

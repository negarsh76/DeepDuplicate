"""Image embedders: deep backbones when available, an offline classic descriptor otherwise."""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np
from PIL import Image

BACKBONES = ("auto", "openclip-l14", "openclip-b32", "dinov2", "resnet50", "classic")


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class ClassicEmbedder:
    """Download-free descriptor: HSV colour histogram + low-frequency DCT of the grey image.

    Much weaker than a CNN for semantic near-duplicates, but works offline and is
    surprisingly good for re-encoded / resized / lightly edited copies.
    """
    name = "classic"
    dim = 8 * 4 * 4 + 16 * 16

    def __call__(self, images: Sequence[Image.Image]) -> np.ndarray:
        from scipy.fft import dctn
        out = []
        for im in images:
            hsv = np.asarray(im.convert("HSV").resize((128, 128)))
            hist, _ = np.histogramdd(hsv.reshape(-1, 3), bins=(8, 4, 4), range=((0, 256),) * 3)
            hist = np.sqrt(hist.ravel() / hist.sum())
            g = np.asarray(im.convert("L").resize((64, 64), Image.LANCZOS), dtype=np.float64)
            d = dctn(g - g.mean(), norm="ortho")[:16, :16].ravel()
            d /= np.linalg.norm(d) + 1e-12
            out.append(np.concatenate([hist, d]))
        return l2n(np.asarray(out, dtype=np.float32))


class TorchEmbedder:
    """Wraps OpenCLIP, DINOv2 (timm) or torchvision ResNet-50 behind one interface."""

    def __init__(self, backbone: str = "auto", device: Optional[str] = None, pretrained: bool = True):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        errors = []
        order = [backbone] if backbone != "auto" else ["openclip-l14", "dinov2", "resnet50"]
        for b in order:
            try:
                self._build(b, pretrained); self.name = b; break
            except Exception as e:  # try the next one
                errors.append(f"{b}: {e}")
        else:
            raise RuntimeError("no deep backbone could be loaded:\n" + "\n".join(errors))
        self.model.eval().to(self.device)

    def _build(self, b: str, pretrained: bool):
        if b.startswith("openclip"):
            import open_clip
            arch = "ViT-L-14" if b == "openclip-l14" else "ViT-B-32"
            model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained="openai" if pretrained else None)
            self.model, self._fwd = model, model.encode_image
        elif b == "dinov2":
            import timm
            self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=0, img_size=224)
            cfg = timm.data.resolve_data_config({"input_size": (3, 224, 224), "crop_pct": 1.0}, model=self.model)
            self.preprocess = timm.data.create_transform(**cfg)
            self._fwd = self.model
        elif b == "resnet50":
            import torchvision
            from torchvision.models import ResNet50_Weights, resnet50
            w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            m = resnet50(weights=w); m.fc = self.torch.nn.Identity()
            self.model, self._fwd = m, m
            self.preprocess = (w.transforms() if w else torchvision.transforms.Compose([
                torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
        else:
            raise ValueError(f"unknown backbone {b!r}")

    def __call__(self, images: Sequence[Image.Image]) -> np.ndarray:
        x = self.torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with self.torch.inference_mode():
            f = self._fwd(x)
        return l2n(f.float().cpu().numpy())


def make_embedder(backbone: str = "auto", **kw) -> Callable[[Sequence[Image.Image]], np.ndarray]:
    if backbone == "classic":
        return ClassicEmbedder()
    try:
        return TorchEmbedder(backbone, **kw)
    except Exception as e:
        if backbone != "auto":
            raise
        print(f"[deepduplicate] deep backbone unavailable ({e.__class__.__name__}); falling back to 'classic'")
        return ClassicEmbedder()


def embed_paths(paths: List, embedder, loader, batch_size: int = 32, progress: bool = True) -> np.ndarray:
    feats, batch = [], []
    it = paths
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(paths, desc=f"embedding ({getattr(embedder, 'name', '?')})", unit="img")
        except ImportError:
            pass
    for p in it:
        im = loader(p)
        batch.append(im if im is not None else Image.new("RGB", (32, 32)))
        if len(batch) == batch_size:
            feats.append(embedder(batch)); batch = []
    if batch:
        feats.append(embedder(batch))
    return np.concatenate(feats) if feats else np.zeros((0, 1), np.float32)

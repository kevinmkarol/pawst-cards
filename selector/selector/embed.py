# ABOUTME: Turns a photo into the list of numbers the selector's model judges: the image embedding
# ABOUTME: of a pretrained CLIP network, computed on this machine and cached on disk.

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

# The pretrained network whose image embeddings are the model's input. Its weights are
# downloaded once (about 350 MB) and reused from the local cache afterwards.
BACKBONE = ("ViT-B-32-quickgelu", "openai")
BATCH = 16
# A region of a photo as fractions of its width and height: left, top, right, bottom.
Region = Tuple[float, float, float, float]
# Numbers in one embedding.
WIDTH = 512


def load_row(cached: Optional[Path]) -> Optional[np.ndarray]:
    """A cached embedding, or None when there is none or the file was cut short by a kill."""
    if cached is None or not cached.is_file():
        return None
    try:
        return np.load(cached)
    except (OSError, ValueError, EOFError):
        return None


def save_row(cached: Path, row: np.ndarray) -> None:
    """Caches an embedding whole: written beside its place, then moved into it."""
    scratch = cached.with_name(cached.stem + ".new.npy")
    np.save(scratch, row)
    os.replace(scratch, cached)


def _pixels(region: Region, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """The region as a pixel box, grown to a square around its centre where the photo allows,
    so that the network's square view takes in all of it."""
    width, height = size
    left, top, right, bottom = (
        region[0] * width,
        region[1] * height,
        region[2] * width,
        region[3] * height,
    )
    side = max(right - left, bottom - top, 32.0)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    return (
        int(max(0, cx - side / 2)),
        int(max(0, cy - side / 2)),
        int(min(width, cx + side / 2)),
        int(min(height, cy + side / 2)),
    )


class Embedder(Protocol):
    def embed(
        self, paths: Sequence[Path], regions: Optional[Sequence[Optional[Region]]] = None
    ) -> np.ndarray: ...


class ClipEmbedder:
    """Embeds photos with CLIP on the CPU; a few dozen frames take a couple of seconds. Results
    are cached in `cache_dir`, when there is one, by the photo's path, size and modification
    time."""

    def __init__(self, cache_dir: Optional[Path]) -> None:
        self._cache_dir = cache_dir
        self._model = None
        self._preprocess = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch

        torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 2))
        name, pretrained = BACKBONE
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        self._model = model.eval()
        self._preprocess = preprocess

    def _cache_path(self, path: Path, region: Optional[Region]) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        stat = path.stat()
        where = "" if region is None else "|" + ",".join(f"{edge:.3f}" for edge in region)
        key = f"{'/'.join(BACKBONE)}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}{where}"
        return self._cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.npy"

    def embed(
        self, paths: Sequence[Path], regions: Optional[Sequence[Optional[Region]]] = None
    ) -> np.ndarray:
        """One unit-length row per photo, in the order given. With `regions`, a photo that has
        one is embedded from that part of it alone, and a photo that has None gets a row of
        zeros."""
        import torch
        from PIL import Image

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        rows: List[np.ndarray | None] = []
        missing: List[int] = []
        for index, path in enumerate(paths):
            if regions is not None and regions[index] is None:
                rows.append(np.zeros(WIDTH, dtype=np.float32))
                continue
            cached = self._cache_path(path, regions[index] if regions is not None else None)
            row = load_row(cached)
            rows.append(row)
            if row is None:
                missing.append(index)
        if missing:
            self._load()
        for start in range(0, len(missing), BATCH):
            chunk = missing[start : start + BATCH]
            images = []
            for index in chunk:
                with Image.open(paths[index]) as image:
                    if regions is None:
                        # Decoding at reduced size makes 5 MP JPEGs quick to read.
                        image.draft("RGB", (448, 448))
                        image = image.convert("RGB")
                    else:
                        image = image.convert("RGB").crop(_pixels(regions[index], image.size))
                    images.append(self._preprocess(image))
            with torch.no_grad():
                features = self._model.encode_image(torch.stack(images))
                features = features / features.norm(dim=-1, keepdim=True)
            for index, row in zip(chunk, features.numpy().astype(np.float32)):
                if self._cache_dir is not None:
                    region = regions[index] if regions is not None else None
                    save_row(self._cache_path(paths[index], region), row)
                rows[index] = row
        return np.stack(rows) if rows else np.zeros((0, WIDTH), dtype=np.float32)

# ABOUTME: Describes each photo for the selector: where the pet is, how sharp it is, and the numbers
# ABOUTME: the model judges (embeddings of the frame and the pet, and the collector's measurements).

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from selector.detect import Detector, PetBox
from selector.embed import WIDTH, Embedder, Region
from selector.measures import MEASURED, measures_of

# Recorded with every trained model; a model made from other features is not used.
FEATURES = "frame+pet-crop+measures/2"
# Features whose columns are the first columns of today's, in the same order. A model trained
# on them is still used, reading the columns it knows, until one is trained on today's.
EARLIER_FEATURES = ("frame+pet-crop/1",)
# A detection at least this sure counts as the pet being in the photo.
PET_SCORE = 0.5
# The pet's crop takes in this much more than its box on every side.
CROP_MARGIN = 0.15
# A pet filling this share of the photo is as large as it needs to be.
AMPLE_AREA = 0.08


@dataclass(frozen=True)
class Described:
    path: Path
    box: Optional[PetBox]
    # Edge contrast inside the pet's box (or the whole photo without one); higher is crisper.
    sharpness: float
    # What the collector measured when it took the frame, by name (MEASURED); empty for a frame
    # from before it measured any, and without whatever it could not measure of this one.
    measures: Mapping[str, float] = field(default_factory=dict)

    @property
    def has_pet(self) -> bool:
        return self.box is not None and self.box.score >= PET_SCORE

    @property
    def centredness(self) -> float:
        if self.box is None:
            return 0.0
        cx, cy = (self.box.left + self.box.right) / 2, (self.box.top + self.box.bottom) / 2
        return max(0.0, 1.0 - (((cx - 0.5) / 0.5) ** 2 + ((cy - 0.5) / 0.5) ** 2) ** 0.5)


def crop_region(box: PetBox) -> Region:
    width, height = box.right - box.left, box.bottom - box.top
    return (
        max(0.0, box.left - CROP_MARGIN * width),
        max(0.0, box.top - CROP_MARGIN * height),
        min(1.0, box.right + CROP_MARGIN * width),
        min(1.0, box.bottom + CROP_MARGIN * height),
    )


def sharpness(path: Path, box: Optional[PetBox]) -> float:
    with Image.open(path) as image:
        image.draft("L", (1296, 972))
        gray = image.convert("L")
        if box is not None:
            width, height = gray.size
            gray = gray.crop(
                (
                    int(box.left * width),
                    int(box.top * height),
                    max(int(box.right * width), int(box.left * width) + 8),
                    max(int(box.bottom * height), int(box.top * height) + 8),
                )
            )
        gray.thumbnail((320, 320))
        pixels = np.asarray(gray, dtype=np.float64)
    if min(pixels.shape) < 3:
        return 0.0
    laplacian = (
        -4 * pixels[1:-1, 1:-1]
        + pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
    )
    return float(laplacian.var())


class Featurizer:
    def __init__(self, embedder: Embedder, detector: Detector) -> None:
        self.embedder = embedder
        self.detector = detector

    def describe(self, paths: Sequence[Path]) -> List[Described]:
        boxes = self.detector.detect(paths)
        return [
            Described(path, box, sharpness(path, box), measures_of(path))
            for path, box in zip(paths, boxes)
        ]

    def features(self, described: Sequence[Described]) -> np.ndarray:
        """One row per photo: the whole frame's embedding, the pet crop's embedding (zeros when
        no pet was found), a few plain measurements of the pet, and the collector's measurements
        of blur and movement, each NaN where the collector did not measure it."""
        if not described:
            return np.zeros((0, 2 * WIDTH + 6 + len(MEASURED)), dtype=np.float32)
        paths = [photo.path for photo in described]
        frames = self.embedder.embed(paths)
        crops = self.embedder.embed(
            paths, [crop_region(photo.box) if photo.has_pet else None for photo in described]
        )
        plain = np.array(
            [
                [
                    float(photo.has_pet),
                    photo.box.score if photo.box else 0.0,
                    photo.box.area if photo.has_pet else 0.0,
                    photo.centredness if photo.has_pet else 0.0,
                    float(photo.box.cut_off) if photo.has_pet else 0.0,
                    np.log1p(photo.sharpness),
                ]
                for photo in described
            ],
            dtype=np.float32,
        )
        measured = np.array(
            [
                [np.log1p(max(0.0, photo.measures[name])) if name in photo.measures else np.nan
                 for name in MEASURED]
                for photo in described
            ],
            dtype=np.float32,
        )  # fmt: skip
        return np.hstack([frames, crops, plain, measured]).astype(np.float32)


def rule_of_thumb(described: Sequence[Described]) -> np.ndarray:
    """A 0 to 1 guess at each photo's worth before any model has been trained: a pet that is
    surely there, large, central, wholly in shot and sharp next to the visit's other frames."""
    sharpest = max((photo.sharpness for photo in described if photo.has_pet), default=0.0) or 1.0
    scores = []
    for photo in described:
        if not photo.has_pet:
            scores.append(0.0)
            continue
        size = min(1.0, photo.box.area / AMPLE_AREA)
        scores.append(
            photo.box.score
            * (0.4 + 0.6 * size)
            * (0.6 + 0.4 * photo.centredness)
            * (0.6 if photo.box.cut_off else 1.0)
            * (0.5 + 0.5 * min(1.0, photo.sharpness / sharpest))
        )
    return np.array(scores)

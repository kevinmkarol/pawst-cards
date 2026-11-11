# ABOUTME: Finds the pet in a photo with a pretrained object detector run on this machine, so
# ABOUTME: frames can be judged by the animal in them and shown with a box around it.

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

# COCO classes that count as the pet unless PETCAM_PETS or `pet_classes` in the selector's
# state.json says otherwise. A cat seen from an odd angle is often called a dog.
PET_CLASSES = ("cat", "dog")
# Detections less sure than this are not kept at all.
LOWEST_SCORE = 0.2
# The detector sees photos reduced to about this size; it shrinks them further itself.
DETECT_SIZE = (1296, 972)


@dataclass(frozen=True)
class PetBox:
    """Where the pet is, as fractions of the photo's width and height, and how sure the
    detector is that it is one."""

    left: float
    top: float
    right: float
    bottom: float
    score: float

    @property
    def area(self) -> float:
        return max(0.0, self.right - self.left) * max(0.0, self.bottom - self.top)

    @property
    def cut_off(self) -> bool:
        """Whether the box runs into an edge of the photo: part of the pet is out of shot."""
        return min(self.left, self.top) < 0.01 or max(self.right, self.bottom) > 0.99

    def overlap(self, other: "PetBox") -> float:
        """Shared area over combined area, 0 to 1."""
        width = min(self.right, other.right) - max(self.left, other.left)
        height = min(self.bottom, other.bottom) - max(self.top, other.top)
        shared = max(0.0, width) * max(0.0, height)
        combined = self.area + other.area - shared
        return shared / combined if combined > 0 else 0.0


def coco_classes() -> List[str]:
    """The names of everything the detector can tell apart."""
    from torchvision.models import detection

    return list(detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"])


def known_classes() -> List[str]:
    """The animals and things the detector can be asked to look for."""
    return [name for name in coco_classes() if name != "N/A" and not name.startswith("__")]


def check_classes(names: Sequence[str]) -> Sequence[str]:
    """`names` if the detector knows them all; ValueError naming the ones it does not."""
    known = known_classes()
    if not names:
        raise ValueError("choose at least one animal")
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"the detector does not know {', '.join(unknown)}; " f"it knows: {', '.join(known)}"
        )
    return names


class Detector(Protocol):
    def detect(self, paths: Sequence[Path]) -> List[Optional[PetBox]]: ...


class PetDetector:
    """Faster R-CNN (ResNet-50, COCO) from torchvision; its weights are downloaded once, about
    170 MB. Runs on the Mac's GPU when there is one, about 0.3 s a photo, else on the CPU.
    Results are cached in `cache_dir`, when there is one."""

    def __init__(self, cache_dir: Optional[Path], classes: Sequence[str] = PET_CLASSES) -> None:
        self._cache_dir = cache_dir
        self._classes = tuple(classes)
        self._model = None
        self._transform = None
        self._pet_labels: List[int] = []
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from torchvision.models import detection

        weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=LOWEST_SCORE)
        self._pet_labels = [coco_classes().index(name) for name in check_classes(self._classes)]
        self._transform = weights.transforms()
        self._model = model.eval()
        if torch.backends.mps.is_available():
            try:
                self._model.to("mps")
                self._device = "mps"
            except Exception:
                self._model.to("cpu")

    def _cache_path(self, path: Path) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        stat = path.stat()
        key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        if self._classes != PET_CLASSES:
            # What was found while looking for other animals says nothing about these.
            key += "|" + ",".join(self._classes)
        return self._cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.json"

    def _detect_one(self, path: Path) -> Optional[PetBox]:
        import torch
        from PIL import Image

        with Image.open(path) as image:
            image.draft("RGB", DETECT_SIZE)
            image = image.convert("RGB")
            width, height = image.size
            tensor = self._transform(image)
        with torch.no_grad():
            try:
                found = self._model([tensor.to(self._device)])[0]
            except Exception:
                if self._device == "cpu":
                    raise
                # The GPU refused; carry on without it.
                self._model.to("cpu")
                self._device = "cpu"
                found = self._model([tensor])[0]
        best: Optional[PetBox] = None
        for box, label, score in zip(
            found["boxes"].tolist(), found["labels"].tolist(), found["scores"].tolist()
        ):
            if label in self._pet_labels and (best is None or score > best.score):
                best = PetBox(
                    box[0] / width, box[1] / height, box[2] / width, box[3] / height, float(score)
                )
        return best

    def detect(self, paths: Sequence[Path]) -> List[Optional[PetBox]]:
        """The most confident pet in each photo, or None where the detector saw none."""
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        boxes: List[Optional[PetBox]] = []
        for path in paths:
            cached = self._cache_path(path)
            if cached is not None and cached.is_file():
                try:
                    saved = json.loads(cached.read_text())
                    boxes.append(PetBox(**saved) if saved else None)
                    continue
                except (OSError, ValueError, TypeError):
                    # A cache entry cut short by a kill: detect again and replace it.
                    pass
            self._load()
            box = self._detect_one(path)
            if cached is not None:
                scratch = cached.with_name(cached.name + ".new")
                scratch.write_text(json.dumps(asdict(box) if box else None))
                os.replace(scratch, cached)
            boxes.append(box)
        return boxes

# ABOUTME: Keeps a day's postcards from repeating themselves: measures how alike two photos look
# ABOUTME: and picks, for the visits waiting to be posted, frames unlike what was posted before.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from selector.detect import Detector, PetBox
from selector.embed import Embedder, Region, _pixels, load_row, save_row
from selector.features import PET_SCORE, crop_region

# The network whose embeddings say how alike two photos look: DreamSim, a DINO ViT-B/16 tuned
# on people's judgements of which pictures look more alike. Its weights are downloaded once
# (about 1.9 GB with everything it unpacks) into the selector's data directory.
BACKBONE = "dino_vitb16"
BATCH = 16
# How alike two photos are runs from about 0 to 1. Measured on this camera, photos of
# different visits seen from the same spot mostly fall between 0.45 and 0.75, and two frames
# of one moment are around 0.9. A photo no more like an earlier one than FAMILIAR loses
# nothing; one as alike as SAME_SHOT loses a whole point of score, which no photo survives.
FAMILIAR = 0.62
SAME_SHOT = 0.86
# Each visit's best few frames are the ones considered; looking at more costs time for
# frames that would rarely be chosen.
FRAMES_PER_VISIT = 6


class ShotEmbedder:
    """Embeds photos with DreamSim, on the Mac's GPU when there is one. Results are cached in
    `cache_dir`, when there is one, by the photo's path, size and modification time."""

    def __init__(self, weights_dir: Path, cache_dir: Optional[Path]) -> None:
        self._weights_dir = weights_dir
        self._cache_dir = cache_dir
        self._model = None
        self._preprocess = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import contextlib
        import io

        import torch
        from dreamsim import dreamsim

        self._weights_dir.mkdir(parents=True, exist_ok=True)
        if torch.backends.mps.is_available():
            self._device = "mps"
        # The library points torch's download directory at its own and leaves it there; the
        # detector's weights live in the usual one.
        hub_dir = torch.hub.get_dir()
        try:
            # The library prints its progress; the selector's commands print only their answer.
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    model, preprocess = self._create(dreamsim)
                except Exception:
                    if self._device == "cpu":
                        raise
                    self._device = "cpu"
                    model, preprocess = self._create(dreamsim)
        finally:
            torch.hub.set_dir(hub_dir)
        self._model = model.eval()
        self._preprocess = preprocess

    def _create(self, dreamsim):
        return dreamsim(
            pretrained=True,
            device=self._device,
            dreamsim_type=BACKBONE,
            cache_dir=str(self._weights_dir),
        )

    def _cache_path(self, path: Path, region: Optional[Region]) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        import hashlib

        stat = path.stat()
        where = "" if region is None else "|" + ",".join(f"{edge:.3f}" for edge in region)
        key = f"dreamsim/{BACKBONE}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}{where}"
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
        rows: List[Optional[np.ndarray]] = []
        missing: List[int] = []
        for index, path in enumerate(paths):
            if regions is not None and regions[index] is None:
                rows.append(None)
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
                        image.draft("RGB", (640, 640))
                        image = image.convert("RGB")
                    else:
                        image = image.convert("RGB").crop(_pixels(regions[index], image.size))
                    images.append(self._preprocess(image)[0])
            with torch.no_grad():
                features = self._model.embed(torch.stack(images).to(self._device))
                features = features.float().reshape(len(chunk), -1).cpu().numpy()
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            for index, row in zip(chunk, features.astype(np.float32)):
                if self._cache_dir is not None:
                    region = regions[index] if regions is not None else None
                    save_row(self._cache_path(paths[index], region), row)
                rows[index] = row
        width = next((len(row) for row in rows if row is not None), 1)
        return np.stack(
            [row if row is not None else np.zeros(width, dtype=np.float32) for row in rows]
        ).reshape(len(paths), width)


@dataclass(frozen=True)
class Shot:
    """What a photo looks like, for comparing with other photos: the whole frame, and the pet
    alone when there is one."""

    whole: np.ndarray
    pet: Optional[np.ndarray]


def alikeness(a: Shot, b: Shot) -> float:
    """How alike two photos look, about 0 to 1. The whole frame says mostly where the camera
    was pointing; the pet alone says what the pet was doing. Both count equally when both
    photos have a pet."""
    whole = float(a.whole @ b.whole)
    if a.pet is None or b.pet is None:
        return whole
    return (whole + float(a.pet @ b.pet)) / 2


class Looker:
    """Turns photos into `Shot`s with a detector, to find the pet, and an embedder."""

    def __init__(self, embedder: Embedder, detector: Detector) -> None:
        self.embedder = embedder
        self.detector = detector

    def shots(self, paths: Sequence[Path]) -> List[Shot]:
        if not paths:
            return []
        boxes: List[Optional[PetBox]] = self.detector.detect(paths)
        regions = [
            crop_region(box) if box is not None and box.score >= PET_SCORE else None
            for box in boxes
        ]
        wholes = self.embedder.embed(paths)
        pets = self.embedder.embed(paths, regions)
        return [
            Shot(wholes[index], pets[index] if regions[index] is not None else None)
            for index in range(len(paths))
        ]


@dataclass(frozen=True)
class Pick:
    # Which of the visit's frames to post; None when every frame repeats what was posted.
    frame: Optional[int]
    # The frame's score less what it loses for looking like an earlier photo.
    worth: float
    # How alike the frame is to the most similar earlier photo, 0 when there was none.
    alike: float


def repeat_cost(alike: float, strictness: float = 1.0) -> float:
    """What a frame's score loses for being `alike` an earlier photo."""
    return strictness * min(1.0, max(0.0, (alike - FAMILIAR) / (SAME_SHOT - FAMILIAR)))


def pick_fresh(
    scores: Sequence[Sequence[float]],
    shots: Sequence[Sequence[Shot]],
    earlier: Sequence[Shot],
    post_at_least: float,
    strictness: float = 1.0,
) -> List[Pick]:
    """For each waiting visit, the frame to post, given its frames' scores and looks and the
    photos posted `earlier`. The frame worth most across all the visits is taken first and
    joins the earlier photos, then the next, so of several visits that look alike the best
    photo is the one kept. A frame is worth its score less `repeat_cost`; a visit with no
    frame worth `post_at_least` is not posted. Frames that score below `post_at_least` to
    begin with are never chosen."""
    alike = [
        np.array([max((alikeness(shot, old) for old in earlier), default=0.0) for shot in visit])
        for visit in shots
    ]
    picks: List[Optional[Pick]] = [None] * len(scores)

    def best_of(visit: int) -> Pick:
        best = Pick(None, float("-inf"), 0.0)
        for index, score in enumerate(scores[visit]):
            if score < post_at_least:
                continue
            worth = score - repeat_cost(float(alike[visit][index]), strictness)
            if worth > best.worth:
                best = Pick(index, float(worth), float(alike[visit][index]))
        return best

    waiting = set(range(len(scores)))
    while waiting:
        offers = {visit: best_of(visit) for visit in waiting}
        visit = max(waiting, key=lambda v: (offers[v].worth, -v))
        offer = offers[visit]
        if offer.frame is None or offer.worth < post_at_least:
            # Earlier photos only accumulate, so no waiting visit can do better later.
            for left in waiting:
                picks[left] = Pick(None, offers[left].worth, offers[left].alike)
            break
        picks[visit] = offer
        waiting.discard(visit)
        chosen = shots[visit][offer.frame]
        for other in waiting:
            for index, shot in enumerate(shots[other]):
                alike[other][index] = max(alike[other][index], alikeness(shot, chosen))
    return [pick if pick is not None else Pick(None, float("-inf"), 0.0) for pick in picks]


def fresh_picks(request: dict, looker: Looker) -> dict:
    """Answers the collector's question, both sides as JSON-ready values. The request:

        {"visits": [[{"frame": path, "score": number}, ...], ...],
         "earlier": [path, ...], "post_at_least": number, "strictness": number}

    with one list of frames per waiting visit and the photos posted `earlier`. The answer has
    a pick per visit, in the same order: the frame to post (null when the visit repeats what
    was posted), what it is worth, and how alike it is to the nearest earlier photo. Frames
    and earlier photos whose files are gone are left out of the reckoning."""
    post_at_least = float(request.get("post_at_least", 0.0))
    strictness = float(request.get("strictness", 1.0))
    visits: List[List[tuple]] = []
    for frames in request["visits"]:
        worthy = [
            (Path(entry["frame"]), float(entry["score"]))
            for entry in frames
            if float(entry["score"]) >= post_at_least and Path(entry["frame"]).is_file()
        ]
        worthy.sort(key=lambda frame: -frame[1])
        visits.append(worthy[:FRAMES_PER_VISIT])
    earlier = [Path(photo) for photo in request.get("earlier", []) if Path(photo).is_file()]
    paths = [path for visit in visits for path, _ in visit]
    looked = looker.shots([*paths, *earlier])
    shots, start = [], 0
    for visit in visits:
        shots.append(looked[start : start + len(visit)])
        start += len(visit)
    picks = pick_fresh(
        [[score for _, score in visit] for visit in visits],
        shots,
        looked[len(paths) :],
        post_at_least,
        strictness,
    )
    return {
        "picks": [
            {
                "frame": str(visit[pick.frame][0]) if pick.frame is not None else None,
                "worth": round(pick.worth, 4) if pick.frame is not None else None,
                "alike": round(pick.alike, 4),
            }
            for visit, pick in zip(visits, picks)
        ]
    }

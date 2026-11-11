# ABOUTME: The selector's data directory: your labels and votes, a kept copy of every labelled
# ABOUTME: frame, trained models, and which selector the collector is told to use.

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image

from selector.measures import keep_measures

# A frame's rating. Good and best are photos worth receiving; okay is the pet, but nothing
# to send; best is the pick of its visit.
BAD, OKAY, GOOD, BEST = 0, 1, 2, 3


def rating(value: object) -> int:
    """A stored label as a rating; the first labels were the words good and bad."""
    if isinstance(value, str):
        return GOOD if value == "good" else BAD
    return int(value)


SELECTORS = ("heuristic", "model")
# Labelled frames are kept at this width: plenty for training, a fraction of the 5 MP original.
KEPT_WIDTH = 1280


def default_data_dir() -> Path:
    return Path(os.environ.get("PETCAM_SELECTOR_DIR") or Path.home() / "petcam" / "selector")


class Store:
    """Everything the selector keeps on disk, under one directory:

    labels.json   how you rated the frames of each visit, and your comparison votes
    frames/       a copy of each labelled frame, so labels outlive the archive's keep time,
                  and with them what the collector measured of each
    models/       trained models, one .npz and one .json per training run
    state.json    which selector the collector uses, which model is the current one, the
                  posting thresholds and the animals looked for
    collector/    the collector's settings, status and latest log lines
    views/        the collector's map of the camera's surroundings and the sweep settings
    embeddings/, detections/, shots/
                  what the models worked out of each frame, kept so it is worked out once
    dreamsim/     the downloaded model that compares photos
    """

    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir
        self.frames_dir = data_dir / "frames"
        self.models_dir = data_dir / "models"
        self._labels_path = data_dir / "labels.json"
        self._state_path = data_dir / "state.json"
        self._lock = threading.Lock()

    def create(self) -> None:
        for directory in (self.dir, self.frames_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # Labels and votes.

    def labels(self) -> dict:
        data = _read(self._labels_path)
        data.setdefault("visits", {})
        data.setdefault("votes", {})
        for entry in data["visits"].values():
            entry["frames"] = {name: rating(value) for name, value in entry["frames"].items()}
        return data

    def rating_counts(self) -> List[int]:
        """How many frames carry each rating, bad first."""
        counts = [0, 0, 0, 0]
        for entry in self.labels()["visits"].values():
            for value in entry["frames"].values():
                counts[value] += 1
        return counts

    def label_visit(
        self,
        key: str,
        started_at: float,
        posted: Optional[str],
        frames: Dict[str, Path],
        labels: Dict[str, int],
        heuristic: Optional[str] = None,
    ) -> None:
        """Record `labels` (frame name -> rating) as the whole verdict on visit `key`, and keep
        a copy of each rated frame. `frames` maps frame names to where they are now. `posted`
        is the frame that was sent and `heuristic` the built-in scoring's pick, which is a
        different frame when the trained model did the choosing."""
        for label in labels.values():
            if label not in (BAD, OKAY, GOOD, BEST) or isinstance(label, bool):
                raise ValueError(f"unknown rating {label!r}")
        for name in labels:
            keep_measures(frames[name], self.keep_frame(key, name, frames[name]))
        with self._lock:
            data = self.labels()
            data["visits"][key] = {
                "started_at": started_at,
                "labelled_at": time.time(),
                "posted": posted,
                "heuristic": heuristic or posted,
                "frames": labels,
            }
            _write(self._labels_path, data)

    def vote(self, key: str, vote: str) -> None:
        """Record which pick of visit `key` you preferred. The vote is on the current model's
        pick, so it is kept with that model's name and stops counting once another is trained."""
        if vote not in (*SELECTORS, "same"):
            raise ValueError(f"unknown vote {vote!r}")
        model = self.current_model()
        with self._lock:
            data = self.labels()
            data["votes"][key] = {"vote": vote, "model": model.stem if model else None}
            _write(self._labels_path, data)

    def votes(self) -> Dict[str, str]:
        """Visit -> your vote, for the votes cast on the current model's picks."""
        model = self.current_model()
        if model is None:
            return {}
        return {
            key: cast["vote"]
            for key, cast in self.labels()["votes"].items()
            # A bare string is a vote from before the model was recorded with it.
            if isinstance(cast, dict) and cast.get("model") == model.stem
        }

    def kept_frame(self, key: str, name: str) -> Path:
        return self.frames_dir / key / name

    def keep_frame(self, key: str, name: str, source: Path) -> Path:
        kept = self.kept_frame(key, name)
        if kept.is_file() or source == kept:
            return kept
        kept.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((KEPT_WIDTH, KEPT_WIDTH))
            # Saved under another name first: a copy cut short must not pass for the frame.
            scratch = kept.with_name(kept.name + ".new")
            image.save(scratch, "JPEG", quality=90)
            os.replace(scratch, kept)
        return kept

    # Selector state and models.

    def state(self) -> dict:
        state = _read(self._state_path)
        if state.get("active") not in SELECTORS:
            state["active"] = "heuristic"
        return state

    def pet_classes(self) -> tuple:
        """What the detector looks for in each photo: `pet_classes` in state.json, or the
        animals named in PETCAM_PETS, or the built-in ones."""
        from selector.detect import PET_CLASSES

        names = self.state().get("pet_classes")
        if isinstance(names, list) and names and all(isinstance(name, str) for name in names):
            return tuple(names)
        # The camera also takes class numbers there; the detector goes by names alone.
        named = [name for name in os.environ.get("PETCAM_PETS", "").split() if not name.isdigit()]
        return tuple(named) or PET_CLASSES

    def set_pet_classes(self, names: Sequence[str]) -> None:
        from selector.detect import check_classes

        self._update_state(pet_classes=list(check_classes(list(names))))

    def set_numbers(self, numbers: Dict[str, float]) -> None:
        """Keep the thresholds the collector reads from state.json."""
        self._update_state(**numbers)

    def current_model(self) -> Optional[Path]:
        """The trained model the collector and the tools use, if there is one. A model trained
        on features this version cannot compute does not count."""
        from selector.features import EARLIER_FEATURES, FEATURES

        name = self.state().get("model")
        path = self.models_dir / f"{name}.npz" if name else None
        if path is None or not path.is_file():
            return None
        made_from = _read(path.with_suffix(".json")).get("features")
        return path if made_from in (FEATURES, *EARLIER_FEATURES) else None

    def outdated_model(self) -> bool:
        """Whether a model was trained but cannot be used, because an earlier version made it
        or its file is gone, so it has to be trained again."""
        name = self.state().get("model")
        return bool(name) and self.current_model() is None

    def model_meta(self, path: Optional[Path] = None) -> Optional[dict]:
        path = path or self.current_model()
        if path is None:
            return None
        return _read(path.with_suffix(".json")) or None

    def models(self) -> List[dict]:
        metas = [_read(path) for path in sorted(self.models_dir.glob("*.json"))]
        return [meta for meta in metas if meta]

    def history(self) -> List[dict]:
        """The record of every training run, oldest first, each marked with whether it is the
        model in use now."""
        current = self.current_model()
        runs = sorted(self.models(), key=lambda meta: meta.get("trained_at", 0.0))
        return [
            {**meta, "current": current is not None and meta.get("name") == current.stem}
            for meta in runs
            if isinstance(meta.get("metrics"), dict)
        ]

    def set_model(self, name: str) -> None:
        if not (self.models_dir / f"{name}.npz").is_file():
            raise ValueError(f"no trained model named {name}")
        self._update_state(model=name)

    def use(self, selector: str) -> None:
        if selector not in SELECTORS:
            raise ValueError(f"selector must be one of {', '.join(SELECTORS)}")
        if selector == "model" and self.current_model() is None:
            raise ValueError("there is no trained model yet; label some visits and train first")
        self._update_state(active=selector)

    def _update_state(self, **changes) -> None:
        with self._lock:
            state = self.state()
            state.update(changes)
            _write(self._state_path, state)


def _read(path: Path) -> dict:
    """What the file holds; nothing when there is no file. One that cannot be made sense of is
    moved aside, not left for the next save to overwrite. A file that cannot be read at all
    raises, so that nothing is saved over it."""
    try:
        text = path.read_bytes()
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        try:
            os.replace(path, path.with_name(path.name + ".unreadable"))
        except OSError:
            pass
        return {}
    return data


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Named for this process: the page and the command line may save at the same moment.
    scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(scratch, "w") as out:
        out.write(json.dumps(data, indent=1))
        # Forced to storage first: otherwise a power cut can leave the new name on an empty file.
        out.flush()
        os.fsync(out.fileno())
    os.replace(scratch, path)

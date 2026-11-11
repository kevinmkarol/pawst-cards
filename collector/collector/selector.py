# ABOUTME: The collector's link to the optional trained selector in selector/: has it score a
# ABOUTME: visit's frames and pick ones unlike photos already posted, and reads which picker to use.

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

log = logging.getLogger("collector")

HEURISTIC, MODEL = "heuristic", "model"
# Longest the model may take over one visit's frames before the built-in scoring is used.
RANK_TIMEOUT_SECONDS = 180
# The model's score for "you would call this photo good". While the model is the active
# selector, a visit whose best frame scores below this is archived but not delivered.
POST_AT_LEAST = 0.5
# Longest the model may take to compare the waiting visits with the photos already posted.
FRESH_TIMEOUT_SECONDS = 300
# How hard a photo is marked down for looking like one already posted: 1 is the selector's
# own measure, more is stricter, and 0 posts every visit however alike.
REPEAT_STRICTNESS = 1.0
# How long a posted photo goes on counting as already posted.
REPEAT_HOURS = 24.0
# The selector project in this repository; it has its own environment, with the model's
# libraries, so the collector's stays small.
SELECTOR_PROJECT = Path(__file__).resolve().parents[2] / "selector"


class ModelSelector:
    """The trained selector as the collector sees it. Until the selector's own tools have
    set up its data directory the collector behaves as if there were no selector, and until
    a model has been trained there the built-in scoring picks every photo."""

    def __init__(self, data_dir: Path, project_dir: Path = SELECTOR_PROJECT) -> None:
        self.data_dir = data_dir
        self.project_dir = project_dir

    def _state(self) -> dict:
        try:
            state = json.loads((self.data_dir / "state.json").read_text())
        except (OSError, ValueError):
            return {}
        return state if isinstance(state, dict) else {}

    def in_use(self) -> bool:
        """Whether anyone is labelling: then visits get extra candidate frames to label."""
        # The selector's tools make this directory for the frames that get rated. The data
        # directory alone proves nothing: the collector keeps the camera's views in it.
        return (self.data_dir / "frames").is_dir()

    def active(self) -> str:
        """Which selector's pick is posted. Read for every visit, so switching needs no restart."""
        return MODEL if self._state().get("active") == MODEL else HEURISTIC

    def post_at_least(self) -> float:
        """The score a visit's best frame needs for the visit to be delivered; `post_at_least`
        in the selector's state file overrides the default, and 0 delivers every visit."""
        try:
            return float(self._state().get("post_at_least", POST_AT_LEAST))
        except (TypeError, ValueError):
            return POST_AT_LEAST

    def trained(self) -> bool:
        name = self._state().get("model")
        return bool(name) and (self.data_dir / "models" / f"{name}.npz").is_file()

    def _number(self, name: str, default: float) -> float:
        try:
            return max(0.0, float(self._state().get(name, default)))
        except (TypeError, ValueError):
            return default

    def repeat_strictness(self) -> float:
        """How hard photos like ones already posted are marked down; `repeat_strictness` in
        the selector's state file overrides the default, and 0 turns the marking down off."""
        return self._number("repeat_strictness", REPEAT_STRICTNESS)

    def repeat_seconds(self) -> float:
        """How long a posted photo counts as already posted; `repeat_hours` in the selector's
        state file overrides the default."""
        return self._number("repeat_hours", REPEAT_HOURS) * 3600

    def _ask(self, arguments: Sequence[str], timeout: float, request: Optional[str] = None) -> dict:
        """The last line the selector's command prints, read as JSON."""
        command = [
            "uv", "run", "--no-sync", "--project", str(self.project_dir),
            "python", "-m", "selector", "--data-dir", str(self.data_dir), *arguments,
        ]  # fmt: skip
        done = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.project_dir,
            input=request,
        )
        return json.loads(done.stdout.strip().splitlines()[-1])

    def scores(self, frames: Sequence[Path]) -> Optional[List[float]]:
        """The model's score for each frame, or None when there is no model or it failed; the
        failure is logged and the visit carries on with the built-in scoring."""
        if not frames or not self.trained():
            return None
        try:
            scores = self._ask(["rank", *(str(frame) for frame in frames)], RANK_TIMEOUT_SECONDS)[
                "scores"
            ]
            if len(scores) != len(frames):
                raise ValueError(f"{len(scores)} scores for {len(frames)} frames")
            return [float(score) for score in scores]
        except subprocess.CalledProcessError as error:
            log.warning("selector model failed: %s", (error.stderr or "").strip()[-400:])
        except Exception as error:
            log.warning("selector model failed: %s", error)
        return None

    def fresh(
        self, visits: Sequence[Sequence[Tuple[Path, float]]], earlier: Sequence[Path]
    ) -> Optional[List[dict]]:
        """For each waiting visit, given its frames and their scores, which frame to post so
        that it does not repeat the photos posted `earlier` or the other waiting visits: a
        dict with `frame` (None when the visit only repeats them), `worth` and `alike`. None
        when the selector failed; the failure is logged."""
        request = {
            "visits": [
                [{"frame": str(frame), "score": score} for frame, score in visit]
                for visit in visits
            ],
            "earlier": [str(photo) for photo in earlier],
            "post_at_least": self.post_at_least(),
            "strictness": self.repeat_strictness(),
        }
        try:
            picks = self._ask(["fresh"], FRESH_TIMEOUT_SECONDS, json.dumps(request))["picks"]
            if len(picks) != len(visits):
                raise ValueError(f"{len(picks)} picks for {len(visits)} visits")
            return picks
        except subprocess.CalledProcessError as error:
            log.warning("selector's repeat check failed: %s", (error.stderr or "").strip()[-400:])
        except Exception as error:
            log.warning("selector's repeat check failed: %s", error)
        return None

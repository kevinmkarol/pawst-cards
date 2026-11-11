# ABOUTME: The agent's timings that can be changed while it runs, through the API: when a visit
# ABOUTME: ends, how long one may be, and how the camera goes home and sweeps. Kept across restarts.

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from petcam.controller import Controller
from petcam.files import remove, save_whole

# Name -> (lowest, highest, whether it is a whole number). Seconds, except the count.
LIMITS: Dict[str, Tuple[float, float, bool]] = {
    # Seconds without the pet that end a visit.
    "gone_after": (10.0, 600.0, False),
    # Longest visit; a pet that stays longer is posted once and then ignored until it moves.
    "max_visit": (60.0, 3600.0, False),
    # Detections within two seconds needed before a visit starts.
    "confirm_sightings": (1, 10, True),
    # Quiet seconds before the camera returns home.
    "home_after": (10.0, 3600.0, False),
    # Seconds at each stop of a sweep.
    "dwell": (5.0, 600.0, False),
}


class Tuning:
    """Reads and changes the timings on a running controller. The values the agent was
    started with are the defaults; ones set through the API replace them until the API is
    told to go back, and are kept in `state_file`."""

    def __init__(self, controller: Controller, state_file: Optional[Path] = None) -> None:
        self._controller = controller
        self._state_file = state_file
        self.default = self._read()
        # Only the timings the API set: the others keep following the flags.
        self._set = self._load()
        self._apply(self._set)

    @property
    def changed(self) -> bool:
        return bool(self._set)

    def _places(self) -> Dict[str, Tuple[Any, str]]:
        presence, patrol = self._controller.presence, self._controller.patrol
        places = {
            "gone_after": (presence, "gone_after_seconds"),
            "max_visit": (presence, "max_visit_seconds"),
            "confirm_sightings": (presence, "confirm_sightings"),
        }
        if patrol is not None:
            places["home_after"] = (patrol, "home_after_seconds")
            places["dwell"] = (patrol, "dwell_seconds")
        return places

    def _read(self) -> Dict[str, float]:
        values = {}
        for name, (holder, attribute) in self._places().items():
            value = getattr(holder, attribute)
            if value is not None:
                values[name] = value
        return values

    def _apply(self, values: Dict[str, float]) -> None:
        for name, (holder, attribute) in self._places().items():
            if name in values:
                setattr(holder, attribute, values[name])

    def status(self) -> dict:
        return {
            "values": self._read(),
            "default": self.default,
            "limits": {name: list(LIMITS[name][:2]) for name in self._read()},
            "changed": self.changed,
        }

    def set(self, body: Dict[str, Any]) -> None:
        """Change the timings named in `body`; ValueError, with nothing changed, when one of
        them is not a timing or is out of range."""
        known = self._read()
        changes: Dict[str, float] = {}
        for name, value in body.items():
            if name not in known:
                raise ValueError(f"{name} is not a timing; the timings are {', '.join(known)}")
            low, high, whole = LIMITS[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            # Written so that NaN, which compares false with everything, is refused as well.
            if not (low <= value <= high):
                raise ValueError(f"{name} must be from {low:g} to {high:g}")
            if whole and value != int(value):
                raise ValueError(f"{name} must be a whole number")
            changes[name] = int(value) if whole else float(value)
        if not changes:
            return
        self._apply(changes)
        self._set.update(changes)
        self._save(self._set)

    def reset(self) -> None:
        """Back to the timings the agent was started with."""
        self._apply(self.default)
        self._set = {}
        if self._state_file is not None:
            remove(self._state_file)

    def _load(self) -> Dict[str, float]:
        if self._state_file is None:
            return {}
        try:
            saved = json.loads(self._state_file.read_text())
        except (OSError, ValueError):
            return {}
        if not isinstance(saved, dict):
            return {}
        kept = {}
        for name, value in saved.items():
            limits = LIMITS.get(name)
            if (
                limits is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and limits[0] <= value <= limits[1]
            ):
                kept[name] = int(value) if limits[2] else float(value)
        return kept

    def _save(self, values: Dict[str, float]) -> None:
        if self._state_file is not None:
            save_whole(self._state_file, json.dumps(values))

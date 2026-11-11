# ABOUTME: Keeps the camera's aim up to date: gathers where it has looked, redraws the map of what
# ABOUTME: each direction has yielded, and has the camera rest and sweep by the plan made from it.

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from collector.camera_client import CameraClient, CameraError
from collector.files import write_whole
from collector.views import (
    CELL_DEGREES,
    HORIZONTAL_FOV_DEGREES,
    VERTICAL_FOV_DEGREES,
    Limits,
    VisitView,
    build_map,
    cell_center,
    describe,
    hour_start,
    make_plan,
    view_of,
)

log = logging.getLogger(__name__)

FIXED = "fixed"
LEARNED = "learned"
MANUAL = "manual"
# Seconds between redrawings of the map; a change of the settings redraws it at once.
REDRAW_EVERY_SECONDS = 600.0
# A plan the camera follows is replaced by a newer one no more often than this, so the camera
# watches each sweep's stops long enough to learn something about them.
REPLAN_EVERY_SECONDS = 3 * 3600.0
# Hours of the camera's looking that are kept.
KEEP_GAZE_DAYS = 60
DEFAULT_EXPLORE = 0.3
DEFAULT_STOPS = 8
# How many of the best photos are listed with the map.
PHOTOS_LISTED = 80


@dataclass(frozen=True)
class Settings:
    """What the person has asked for, from settings.json; the page writes it."""

    # FIXED: the home and stops the camera's agent was started with. LEARNED: the map's.
    # MANUAL: the ones set on the camera by hand, from the page; they are left as they are.
    mode: str = FIXED
    # From 0, sweep only where pets have turned up, to 1, sweep where the camera has not looked.
    explore: float = DEFAULT_EXPLORE
    # Nothing before this time counts; set when the camera is moved to a new spot.
    since: Optional[float] = None
    stops: int = DEFAULT_STOPS
    limits: Limits = Limits()

    @classmethod
    def read(cls, path: Path) -> "Settings":
        try:
            data = json.loads(path.read_text())
            limits = Limits(
                **{
                    name: float(data[name])
                    for name in ("yaw_low", "yaw_high", "pitch_low", "pitch_high")
                    if name in data
                }
            )
            since = data.get("since")
            return cls(
                mode=data["mode"] if data.get("mode") in (LEARNED, MANUAL) else FIXED,
                explore=min(1.0, max(0.0, float(data.get("explore", DEFAULT_EXPLORE)))),
                since=None if since is None else float(since),
                stops=min(16, max(0, int(data.get("stops", DEFAULT_STOPS)))),
                limits=limits,
            )
        except (OSError, ValueError, TypeError, AttributeError):
            return cls()

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "explore": self.explore,
            "since": self.since,
            "stops": self.stops,
        }


def _write(path: Path, value: object) -> None:
    write_whole(path, json.dumps(value))


def _pose(entry: dict) -> Tuple[float, float]:
    return (float(entry["yaw"]), float(entry["pitch"]))


def _missed(camera: dict) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """The poses the camera says it was sent to and did not reach, each with where it stopped
    instead; none from an agent that does not check."""
    found = []
    reported: Any = camera.get("missed")
    for entry in reported if isinstance(reported, list) else []:
        try:
            found.append((_pose(entry["asked"]), _pose(entry["reached"])))
        except (KeyError, TypeError, ValueError):
            continue
    return found


class Aimer:
    """`directory` holds what this keeps and what the page reads: settings.json (written by
    the page), gaze.json (the camera's looking, gathered), map.json and pose.json."""

    def __init__(
        self,
        client: CameraClient,
        archive_dir: Path,
        directory: Path,
        resolution: Tuple[int, int],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.client = client
        self.archive_dir = archive_dir
        self.directory = directory
        self.resolution = resolution
        self._clock = clock
        self._drawn_at: Optional[float] = None
        self._drawn_for: Optional[Settings] = None
        self._planned_at: Optional[float] = None
        self._planned_for: Optional[Tuple[Settings, Limits]] = None
        # Archive directory name -> (visit.json's modification time, what the map needs of it).
        self._views: Dict[str, Tuple[int, Optional[VisitView]]] = {}

    def saw(self, status: dict) -> None:
        """Note where the camera says it points right now, for the page."""
        if "pose" not in status:
            return
        try:
            _write(
                self.directory / "pose.json",
                {
                    "at": self._clock(),
                    "pose": status.get("pose"),
                    "in_visit": status.get("in_visit"),
                    "patrolling": (status.get("patrol") or {}).get("patrolling"),
                },
            )
        except OSError as error:
            log.warning("could not note the camera's pose: %s", error)

    def refresh(self) -> bool:
        """Redraw the map and bring the camera's plan up to date, if it is time to or the
        settings have changed; says whether it did. Raises what the camera client raises."""
        settings = Settings.read(self.directory / "settings.json")
        now = self._clock()
        if (
            self._drawn_at is not None
            and now - self._drawn_at < REDRAW_EVERY_SECONDS
            and settings == self._drawn_for
        ):
            return False
        try:
            camera = self.client.patrol()
        except CameraError as error:
            if "HTTP 404" not in str(error):
                raise
            self._drawn_at, self._drawn_for = now, settings
            # An agent from before it recorded where it looks; the page says so.
            _write(self.directory / "map.json", self._drawing(settings, now, None, None, [], {}))
            return True
        gaze = self._gather_gaze(now)
        visits = self._visits()
        view_map = build_map(visits, gaze, now, settings.since)
        # The limits the person set, drawn in wherever the gimbal has stopped short of a pose.
        limits = settings.limits.within_reach(_missed(camera))
        plan = make_plan(
            view_map,
            settings.explore,
            _pose(camera["plan"]["home"] if camera.get("learned") else camera["default"]["home"]),
            settings.stops,
            limits,
        )
        camera = self._steer(settings, limits, camera, plan, now)
        drawing = self._drawing(settings, now, camera, plan, visits, gaze)
        drawing["cells"] = describe(view_map)
        drawing["limits"] = limits.as_dict()
        drawing["stopped_short"] = limits != settings.limits
        # The centers of the cells some pose within the limits sees; the rest cannot be watched.
        seeable = limits.seeable()
        drawing["seeable"] = [
            list(cell_center((int(yaw), int(pitch)))) for yaw, pitch in np.argwhere(seeable)
        ]
        mean, doubt = view_map.expected()
        # What a cell nobody has watched is taken to be: every such cell is the same.
        unwatched = view_map.watched == 0
        if unwatched.any():
            drawing["unwatched"] = {
                "expected": round(float(mean[unwatched].min()), 4),
                "doubt": round(float(doubt[unwatched].max()), 4),
            }
        _write(self.directory / "map.json", drawing)
        # Noted last, so a pass the camera dropped out of is made again at the next poll.
        self._drawn_at, self._drawn_for = now, settings
        return True

    def _steer(
        self, settings: Settings, limits: Limits, camera: dict, plan: dict, now: float
    ) -> dict:
        """Have the camera follow what the settings ask for; returns what it follows after."""
        if settings.mode == FIXED:
            if camera.get("learned"):
                log.info("aim: back to the camera's own home and patrol")
                camera = self.client.set_patrol(None)
            self._planned_at = self._planned_for = None
            return camera
        if settings.mode == MANUAL:
            self._planned_at = self._planned_for = None
            return camera
        due = (
            self._planned_at is None
            or (settings, limits) != self._planned_for
            or now - self._planned_at >= REPLAN_EVERY_SECONDS
        )
        if due:
            if not camera.get("learned") or camera.get("plan") != plan:
                log.info(
                    "aim: new plan, home %s, %d stops, sweep after %.0f s idle",
                    plan["home"],
                    len(plan["stops"]),
                    plan["patrol_after"],
                )
                camera = self.client.set_patrol(plan)
            # Noted only once the camera has the plan: one it did not receive is sent again.
            self._planned_at, self._planned_for = now, (settings, limits)
        return camera

    def _gather_gaze(self, now: float) -> Dict[str, Dict[str, float]]:
        """The camera's record of where it looked, added to what was gathered before."""
        path = self.directory / "gaze.json"
        try:
            held: Dict[str, Dict[str, float]] = json.loads(path.read_text())
        except (OSError, ValueError):
            held = {}
        # The last hour held was still being added to when it was fetched.
        held.update(self.client.gaze(max(held) if held else None))
        oldest = now - KEEP_GAZE_DAYS * 86400
        held = {
            hour: looked for hour, looked in held.items() if (hour_start(hour) or 0.0) >= oldest
        }
        _write(path, held)
        return held

    def _visits(self) -> List[VisitView]:
        """Every archived visit that recorded where the camera pointed."""
        views: Dict[str, Tuple[int, Optional[VisitView]]] = {}
        for record in self.archive_dir.glob("*/visit.json"):
            name = record.parent.name
            try:
                changed = record.stat().st_mtime_ns
                held = self._views.get(name)
                if held is not None and held[0] == changed:
                    views[name] = held
                    continue
                views[name] = (
                    changed,
                    view_of(name, json.loads(record.read_text()), self.resolution),
                )
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                log.warning("aim: visit %s left out of the map: %r", name, error)
        self._views = views
        return [view for _, view in views.values() if view is not None]

    def _drawing(
        self,
        settings: Settings,
        now: float,
        camera: Optional[dict],
        plan: Optional[dict],
        visits: List[VisitView],
        gaze: Dict[str, Dict[str, float]],
    ) -> dict:
        counted = [v for v in visits if settings.since is None or v.at >= settings.since]
        best = sorted((v for v in counted if v.worth > 0), key=lambda v: -v.worth)
        return {
            "drawn_at": now,
            "cell_degrees": CELL_DEGREES,
            "fov": [HORIZONTAL_FOV_DEGREES, VERTICAL_FOV_DEGREES],
            "settings": settings.as_dict(),
            # None when the camera's agent is too old to say where it looks or take a plan.
            "camera": camera,
            "proposed": plan,
            "visits": len(counted),
            "watched_hours": round(
                sum(
                    seconds
                    for hour, looked in gaze.items()
                    if settings.since is None or (hour_start(hour) or 0.0) >= settings.since
                    for seconds in looked.values()
                )
                / 3600,
                2,
            ),
            "photos": [
                {
                    "visit": v.key,
                    "frame": v.photo.rsplit("/", 1)[-1] if v.photo else None,
                    "score": v.score,
                    "worth": round(v.worth, 3),
                    "at": v.at,
                    "yaw": round(v.photo_at[0], 1),
                    "pitch": round(v.photo_at[1], 1),
                    "found_yaw": round(v.found[0], 1),
                    "found_pitch": round(v.found[1], 1),
                }
                for v in best[:PHOTOS_LISTED]
            ],
            "cells": [],
        }

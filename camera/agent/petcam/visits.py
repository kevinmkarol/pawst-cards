# ABOUTME: Remembers each pet visit: when it started and ended, every sighting during it, and which
# ABOUTME: recorded clip files belong to it, so the collector can fetch the visit afterwards.

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from petcam.detections import Box
from petcam.files import save_whole

# The camera's direction is noted during a visit whenever it has turned this many degrees, and
# at least this often in seconds.
GAZE_STEP_DEGREES = 1.0
GAZE_EVERY_SECONDS = 10.0
# Most sightings kept across the finished visits, a visit counting for at least VISIT_WEIGHT:
# with the collector away for weeks the record must not outgrow the camera's memory.
MOST_KEPT_SIGHTINGS = 40_000
VISIT_WEIGHT = 20

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sighting:
    at: float
    box: Box


@dataclass
class Visit:
    id: str
    started_at: float
    ended_at: Optional[float] = None
    sightings: List[Sighting] = field(default_factory=list)
    # Which boot of the camera recorded the visit; the clock starts over at each one.
    boot: Optional[str] = None
    # Seconds to add to the visit's times to get the real time. Not zero when the visit was
    # recorded before the camera's clock was set; its clips carry the same uncorrected times.
    clock_offset: float = 0.0
    # Where the camera pointed during the visit: (time, yaw, pitch) each time it had turned.
    gaze: List[Tuple[float, float, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "boot": self.boot,
            "clock_offset": self.clock_offset,
            "gaze": [list(looked) for looked in self.gaze],
            "sightings": [
                {
                    "at": s.at,
                    "cx": s.box.cx,
                    "cy": s.box.cy,
                    "w": s.box.w,
                    "h": s.box.h,
                    "score": s.box.score,
                    "class_id": s.box.class_id,
                }
                for s in self.sightings
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Visit":
        return cls(
            id=data["id"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            boot=data.get("boot"),
            clock_offset=data.get("clock_offset", 0.0),
            gaze=[(at, yaw, pitch) for at, yaw, pitch in data.get("gaze", [])],
            sightings=[
                Sighting(
                    at=s["at"],
                    box=Box(
                        cx=s["cx"],
                        cy=s["cy"],
                        w=s["w"],
                        h=s["h"],
                        score=s["score"],
                        class_id=s["class_id"],
                    ),
                )
                for s in data["sightings"]
            ],
        )


@dataclass(frozen=True)
class Clip:
    path: Path
    size: int
    mtime: float


def recorded_clips(directory: Path) -> List[Clip]:
    """Every clip file in `directory`, oldest first. The recorder deletes its oldest clips when
    storage fills, so one that disappears while the directory is being read is left out."""
    clips = []
    for path in directory.glob("*.mp4"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        clips.append(Clip(path=path, size=stat.st_size, mtime=stat.st_mtime))
    return sorted(clips, key=lambda clip: clip.mtime)


class VisitLog:
    """Finished visits are written to `state_file` so they survive an agent restart. `boot_id`
    names the current boot of the camera."""

    def __init__(self, state_file: Optional[Path] = None, boot_id: Optional[str] = None) -> None:
        self.open: Optional[Visit] = None
        self.state_file = state_file
        self.boot_id = boot_id
        self._finished: List[Visit] = self._load()

    def start(self, now: float) -> Visit:
        self.open = Visit(
            id=time.strftime("%Y%m%d-%H%M%S", time.gmtime(now)) + f"-{int(now * 100) % 100:02d}",
            started_at=now,
            boot=self.boot_id,
        )
        return self.open

    def sighting(self, box: Box, now: float) -> None:
        if self.open is not None:
            self.open.sightings.append(Sighting(at=now, box=box))

    def looking(self, yaw: float, pitch: float, now: float) -> None:
        """Note which way the camera points, if a visit is open and it has turned since the
        last note or that note is old."""
        if self.open is None:
            return
        gaze = self.open.gaze
        if gaze:
            at, last_yaw, last_pitch = gaze[-1]
            if (
                abs(yaw - last_yaw) < GAZE_STEP_DEGREES
                and abs(pitch - last_pitch) < GAZE_STEP_DEGREES
                and now - at < GAZE_EVERY_SECONDS
            ):
                return
        gaze.append((round(now, 2), round(yaw, 1), round(pitch, 1)))

    def end(self, now: float) -> Visit:
        visit = self.open
        if visit is None:
            raise RuntimeError("no visit in progress")
        visit.ended_at = now
        self._finished.append(visit)
        self.open = None
        # Beyond the most kept, the oldest go: the recorder has likely deleted their clips.
        weight = 0
        newest = len(self._finished) - 1
        for index in range(newest, -1, -1):
            weight += max(len(self._finished[index].sightings), VISIT_WEIGHT)
            if weight > MOST_KEPT_SIGHTINGS and index < newest:
                log.warning("too many visits wait to be collected; the oldest %d go", index + 1)
                del self._finished[: index + 1]
                break
        self._save()
        return visit

    def finished(self) -> List[Visit]:
        return list(self._finished)

    def get(self, visit_id: str) -> Optional[Visit]:
        return next((v for v in self._finished if v.id == visit_id), None)

    def collected(self, *visit_ids: str) -> None:
        self._finished = [v for v in self._finished if v.id not in visit_ids]
        self._save()

    def clock_stepped(self, seconds: float) -> None:
        """The camera's clock was moved by `seconds`. Every visit finished since boot was timed
        by the clock as it was before, so its real time is that much later."""
        if self.boot_id is None:
            return
        for visit in self._finished:
            if visit.boot == self.boot_id:
                visit.clock_offset += seconds
        self._save()

    def _load(self) -> List[Visit]:
        if self.state_file is None or not self.state_file.exists():
            return []
        try:
            return [Visit.from_dict(v) for v in json.loads(self.state_file.read_text())["visits"]]
        except (OSError, ValueError, KeyError, TypeError) as error:
            # Kept beside the state file for a look later; the next save would overwrite it.
            aside = self.state_file.with_name(self.state_file.name + ".unreadable")
            log.error(
                "visit state %s is unreadable (%s); moved to %s", self.state_file, error, aside
            )
            try:
                os.replace(self.state_file, aside)
            except OSError:
                pass
            return []

    def _save(self) -> None:
        # If this fails the visits are still held in memory and are written at the next change.
        if self.state_file is not None:
            save_whole(
                self.state_file, json.dumps({"visits": [v.as_dict() for v in self._finished]})
            )

    def clips_among(self, visit: Visit, clips: List[Clip], slice_seconds: int) -> List[Clip]:
        """The clips out of `clips` written while the visit was being recorded, in their order.

        The recorder closes a slice up to one slice length after the visit ends, so the window
        extends past ended_at by that much plus a little slack.
        """
        earliest = visit.started_at - 1
        latest = (visit.ended_at or time.time()) + slice_seconds + 2
        # Checks every clip, to keep this simple. `clips` is ordered by mtime, so bisecting for the
        # window's two ends would find the same run without visiting the rest.
        return [clip for clip in clips if earliest <= clip.mtime <= latest]

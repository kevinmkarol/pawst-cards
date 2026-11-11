# ABOUTME: Remembers which way the camera was looking while it waited for a pet, as seconds per
# ABOUTME: direction for each hour, so the collector can tell watched places from unwatched ones.

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

from petcam.patrol import Pose
from petcam.files import save_whole

log = logging.getLogger(__name__)

# Directions are rounded to this many degrees.
BIN_DEGREES = 5
# Hours of looking that are kept; the collector fetches them long before they are dropped.
KEEP_HOURS = 14 * 24


def hour_of(now: float) -> str:
    """The name of the hour `now` falls in, UTC, which sorts in time order."""
    return time.strftime("%Y%m%d%H", time.gmtime(now))


def direction(pose: Pose) -> str:
    yaw = int(round(pose.yaw / BIN_DEGREES)) * BIN_DEGREES % 360
    pitch = int(round(pose.pitch / BIN_DEGREES)) * BIN_DEGREES
    return f"{yaw},{pitch}"


class GazeLog:
    """Seconds spent looking each way, per hour. Kept in `state_file`, when there is one, by
    `save`."""

    def __init__(self, state_file: Optional[Path] = None) -> None:
        self.state_file = state_file
        # Hour -> direction -> seconds.
        self._hours: Dict[str, Dict[str, float]] = self._load()

    def add(self, pose: Pose, seconds: float, now: float) -> None:
        hour = self._hours.setdefault(hour_of(now), {})
        where = direction(pose)
        hour[where] = hour.get(where, 0.0) + seconds
        if len(self._hours) > KEEP_HOURS:
            for old in sorted(self._hours)[:-KEEP_HOURS]:
                del self._hours[old]

    def since(self, hour: Optional[str] = None) -> dict:
        """The hours from `hour` on, or all of them."""
        return {
            "bin_degrees": BIN_DEGREES,
            "hours": {
                name: {where: round(seconds, 1) for where, seconds in looked.items()}
                for name, looked in sorted(self._hours.items())
                if hour is None or name >= hour
            },
        }

    def save(self) -> None:
        if self.state_file is not None:
            save_whole(self.state_file, json.dumps(self.since()["hours"]))

    def _load(self) -> Dict[str, Dict[str, float]]:
        if self.state_file is None or not self.state_file.exists():
            return {}
        try:
            hours = json.loads(self.state_file.read_text())
            return {
                str(hour): {str(where): float(seconds) for where, seconds in looked.items()}
                for hour, looked in hours.items()
            }
        except (OSError, ValueError, AttributeError, TypeError) as error:
            log.error("gaze log %s is unreadable (%s); starting again", self.state_file, error)
            return {}

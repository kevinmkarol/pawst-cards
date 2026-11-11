# ABOUTME: The agent's sense of time on a camera with no battery clock: timestamps that never
# ABOUTME: jump mid-visit, a report when the system clock is stepped, and a saved time for boot.

import logging
import time
from pathlib import Path
from typing import Callable
from petcam.files import save_whole

log = logging.getLogger(__name__)

# The system clock moving this far against the monotonic clock is a step (NTP setting the time
# after boot), not drift.
STEP_SECONDS = 5.0


class Clock:
    """Wall-clock time that advances with the monotonic clock between calls to `resync`.

    The camera starts at 1970 on every boot and is set to the real time whenever NTP first
    answers, which can be seconds or days later. Reading the system clock directly would put
    that step into the middle of whatever was being timed.
    """

    def __init__(
        self,
        wall: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._wall = wall
        self._monotonic = monotonic
        self._offset = wall() - monotonic()

    def now(self) -> float:
        return self._monotonic() + self._offset

    def resync(self) -> float:
        """Follow the system clock again. Returns how far it was stepped, in seconds, or 0.0
        when it only drifted."""
        offset = self._wall() - self._monotonic()
        moved = offset - self._offset
        self._offset = offset
        return moved if abs(moved) >= STEP_SECONDS else 0.0


def restore_saved_time(
    path: Path,
    wall: Callable[[], float] = time.time,
    set_wall: Callable[[float], None] = lambda t: time.clock_settime(time.CLOCK_REALTIME, t),
) -> bool:
    """Set the system clock to the time saved in `path` when the clock is behind it, which it is
    after a boot until NTP answers. The result is late by however long the camera was off, but
    visits get a date near the real one and stay in order. Returns whether the clock was set."""
    try:
        saved = float(path.read_text())
    except (OSError, ValueError):
        return False
    if wall() >= saved:
        return False
    try:
        set_wall(saved)
    except OSError as error:
        log.warning("could not set the clock to the saved time: %s", error)
        return False
    return True


def save_time(path: Path, now: float) -> None:
    """Remember `now` for the next boot; a power cut leaves either the old time or the new."""
    save_whole(path, f"{now:.0f}\n")

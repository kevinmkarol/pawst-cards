# ABOUTME: Reads from a visit's record how much was moving at any moment: how far the camera
# ABOUTME: turned and how fast the pet's box travelled, so photos are taken while both were still.

from bisect import bisect_left, bisect_right
from math import hypot
from typing import List, Optional, Sequence, Tuple

from collector.scoring import Sighting

# Movement is looked for this long either side of a moment, in seconds.
WINDOW_SECONDS = 0.5
# A moment is worth half as much when the camera turned this many degrees around it. The camera
# notes its direction once it has turned a degree, so this is one noted turn.
TURN_HALVES_AT_DEGREES = 1.0
# And half as much when the pet's box crossed this share of the picture's width in a second.
SPEED_HALVES_AT_WIDTHS = 0.06
# What a moment is worth, of 1, when nothing says how fast the pet was moving: as if it moved
# at the halving speed.
UNKNOWN_STILLNESS = 0.5


def _eased(value: float, halves_at: float) -> float:
    """1 for no movement, a half at `halves_at`, falling towards 0 beyond it."""
    return 1.0 / (1.0 + value / halves_at)


class VisitMotion:
    """What a visit's record says was moving. `sightings` are the pet's boxes in a detection
    picture of `resolution`; `gaze` is [time, yaw, pitch] each time the camera had turned, empty
    from a camera that does not record it."""

    def __init__(
        self,
        sightings: Sequence[Sighting],
        gaze: Sequence[Sequence[float]],
        resolution: Tuple[int, int],
    ) -> None:
        self._sightings: List[Sighting] = sorted(sightings, key=lambda sighting: sighting.at)
        self._seen_at = [sighting.at for sighting in self._sightings]
        self._gaze = sorted((float(at), float(yaw), float(pitch)) for at, yaw, pitch in gaze)
        self._looked_at = [at for at, _, _ in self._gaze]
        self._width = float(resolution[0])

    def gimbal_turn(self, at: float, window: float = WINDOW_SECONDS) -> Optional[float]:
        """Degrees the camera turned within `window` seconds either side of `at`; None when
        the visit has no record of where it pointed."""
        if not self._gaze:
            return None
        first = bisect_right(self._looked_at, at - window)
        last = bisect_right(self._looked_at, at + window)
        turned = 0.0
        # Each note is the direction from then until the next one, so a turn happened at the
        # later note of a pair.
        for index in range(max(first, 1), last):
            _, yaw, pitch = self._gaze[index]
            _, yaw_before, pitch_before = self._gaze[index - 1]
            turned += hypot(yaw - yaw_before, pitch - pitch_before)
        return turned

    def box_speed(self, at: float, window: float = WINDOW_SECONDS) -> Optional[float]:
        """How fast the pet's box crossed the picture around `at`, in picture widths a second,
        whether the pet moved or the camera did; None when the pet was not seen at least twice
        within `window` seconds either side."""
        first = bisect_left(self._seen_at, at - window)
        last = bisect_right(self._seen_at, at + window) - 1
        if last <= first:
            return None
        start, end = self._sightings[first], self._sightings[last]
        if end.at <= start.at:
            return None
        # From the first sighting to the last, not along every step between them: the detector's
        # boxes jitter by a pixel or two each time, which would add up to a speed of its own.
        travelled = hypot(end.cx - start.cx, end.cy - start.cy)
        return travelled / self._width / (end.at - start.at)

    def stillness(self, at: float) -> float:
        """How still things were around `at`: 1 when neither the camera nor the pet's box
        moved, towards 0 the more they did."""
        turn, speed = self.gimbal_turn(at), self.box_speed(at)
        # A camera that keeps no record of its direction says the same of every moment.
        camera = 1.0 if turn is None else _eased(turn, TURN_HALVES_AT_DEGREES)
        pet = UNKNOWN_STILLNESS if speed is None else _eased(speed, SPEED_HALVES_AT_WIDTHS)
        return camera * pet

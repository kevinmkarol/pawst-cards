# ABOUTME: Notices when detection frames stop arriving and says what to try next: ask for the
# ABOUTME: nodes again, or restart the camera's capture service, with growing gaps between restarts.

from enum import Enum
from typing import Optional


class Remedy(Enum):
    # Send the node create requests again; harmless when the nodes exist.
    CREATE = "create"
    # Restart sscma-node and build the whole pipeline again.
    RESTART = "restart"


class CaptureWatchdog:
    """The model node sends a frame several times a second whether or not there is a pet, so a
    gap means capture has stopped. Times are monotonic seconds."""

    def __init__(
        self,
        now: float,
        create_every_seconds: float = 15.0,
        restart_after_seconds: float = 90.0,
        first_restart_gap_seconds: float = 300.0,
        longest_restart_gap_seconds: float = 3600.0,
    ) -> None:
        self.create_every_seconds = create_every_seconds
        self.restart_after_seconds = restart_after_seconds
        self.first_restart_gap_seconds = first_restart_gap_seconds
        self.longest_restart_gap_seconds = longest_restart_gap_seconds
        self.restarts = 0
        self._last_frame_at: Optional[float] = None
        # When the current wait for a frame began: agent start, the last frame, or a restart.
        self._waiting_since = now
        self._last_create_at = now
        self._restart_gap = first_restart_gap_seconds
        self._next_restart_at = now + restart_after_seconds
        self._flowing = False
        self._replaced = False
        self._last_restart_at: Optional[float] = None

    def frame(self, now: float) -> bool:
        """Record a frame. Returns True when it is the first after a gap or since the start."""
        resumed = not self._flowing
        self._flowing = True
        self._last_frame_at = now
        self._waiting_since = now
        self._restart_gap = self.first_restart_gap_seconds
        self._next_restart_at = now + self.restart_after_seconds
        return resumed

    def camera_replaced(self) -> None:
        """Something other than the agent has created the camera node, so it is no longer the
        one the pipeline asked for. The next check asks for a restart, unless there was one
        within the last `restart_after_seconds`."""
        self._replaced = True

    def check(self, now: float) -> Optional[Remedy]:
        """What to do about a gap, if anything. Call about once a second."""
        if self._replaced:
            self._replaced = False
            if self._last_restart_at is None or (
                now - self._last_restart_at >= self.restart_after_seconds
            ):
                return self._restart(now)
        if now - self._waiting_since < self.create_every_seconds:
            return None
        self._flowing = False
        if now >= self._next_restart_at:
            return self._restart(now)
        if now - self._last_create_at >= self.create_every_seconds:
            self._last_create_at = now
            return Remedy.CREATE
        return None

    def _restart(self, now: float) -> Remedy:
        self.restarts += 1
        # The restart may put back a recorder that is not recording, so the next frame counts
        # as the first again and the recorder is told what to do.
        self._flowing = False
        self._last_restart_at = now
        self._waiting_since = now
        self._next_restart_at = now + self._restart_gap
        self._restart_gap = min(self._restart_gap * 2, self.longest_restart_gap_seconds)
        self._last_create_at = now
        return Remedy.RESTART

    def status(self, now: float) -> dict:
        return {
            "flowing": self._flowing,
            "seconds_since_frame": (
                None if self._last_frame_at is None else round(now - self._last_frame_at, 1)
            ),
            "restarts": self.restarts,
        }

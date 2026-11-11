# ABOUTME: Tracks whether a pet visit is in progress: one starts once several sightings arrive close
# ABOUTME: together and ends after a quiet period or at a cap; after a cap the pet must move first.

from enum import Enum
from typing import List, Optional


class VisitEvent(Enum):
    STARTED = "started"
    ENDED = "ended"


class Presence:
    def __init__(
        self,
        gone_after_seconds: float,
        confirm_sightings: int = 1,
        confirm_window_seconds: float = 2.0,
        max_visit_seconds: Optional[float] = None,
    ) -> None:
        self.gone_after_seconds = gone_after_seconds
        self.confirm_sightings = confirm_sightings
        self.confirm_window_seconds = confirm_window_seconds
        self.max_visit_seconds = max_visit_seconds
        self.in_visit = False
        # A pet that stayed for a whole capped visit; sightings of it do not start another
        # visit until it moves or goes away.
        self.settled = False
        self.started_at: Optional[float] = None
        self.last_seen_at: Optional[float] = None
        self._recent_sightings: List[float] = []

    def update(self, pet_seen: bool, now: float, moving: bool = False) -> Optional[VisitEvent]:
        """Feed one observation; returns an event when a visit starts or ends.

        `moving` says the pet changed position recently; only a moving pet can start a visit
        while the presence is settled.
        """
        if pet_seen:
            self.last_seen_at = now
            if self.in_visit:
                if self._visit_reached_cap(now):
                    self.in_visit = False
                    self.settled = True
                    return VisitEvent.ENDED
                return None
            if self.settled and not moving:
                return None
            self.settled = False
            self._recent_sightings = [
                t for t in self._recent_sightings if now - t <= self.confirm_window_seconds
            ]
            self._recent_sightings.append(now)
            if len(self._recent_sightings) < self.confirm_sightings:
                return None
            self.in_visit = True
            self.started_at = self._recent_sightings[0]
            self._recent_sightings = []
            return VisitEvent.STARTED
        if self.last_seen_at is not None and now - self.last_seen_at >= self.gone_after_seconds:
            self.settled = False
            if self.in_visit:
                self.in_visit = False
                return VisitEvent.ENDED
        return None

    def reset(self) -> None:
        """Forget every time remembered so far, for when the clock they came from is moved. A
        visit in progress must be ended by the caller first."""
        self.in_visit = False
        self.started_at = None
        self.last_seen_at = None
        self._recent_sightings = []

    def _visit_reached_cap(self, now: float) -> bool:
        if self.max_visit_seconds is None or self.started_at is None:
            return False
        return now - self.started_at >= self.max_visit_seconds

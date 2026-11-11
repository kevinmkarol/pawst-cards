# ABOUTME: Decides where the gimbal should rest when no pet is around: back to a home pose after a
# ABOUTME: quiet spell, and a slow sweep through patrol stops after a long one.

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from petcam.files import remove, save_whole

log = logging.getLogger(__name__)

# What the motors accept, in degrees.
YAW_RANGE = (0.0, 360.0)
PITCH_RANGE = (0.0, 180.0)
# A plan sent to the camera has at most this many stops, and sweeps no more often than this.
MAX_STOPS = 24
SHORTEST_PATROL_AFTER_SECONDS = 60.0
# A pose the gimbal was sent to counts as reached when the motors report one within this many
# degrees of it. They stop silently at their limits, and at anything in their way.
REACH_TOLERANCE_DEGREES = 5.0


@dataclass(frozen=True)
class Pose:
    yaw: float
    pitch: float

    def as_dict(self) -> dict:
        return {"yaw": self.yaw, "pitch": self.pitch}

    @classmethod
    def from_dict(cls, data: Any) -> "Pose":
        """The pose in `data`; ValueError unless it is a yaw and pitch the motors accept."""
        try:
            yaw, pitch = float(data["yaw"]), float(data["pitch"])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError("a pose needs a yaw and a pitch in degrees")
        # Written so that NaN, which compares false with everything, is refused as well.
        if not (YAW_RANGE[0] <= yaw <= YAW_RANGE[1] and PITCH_RANGE[0] <= pitch <= PITCH_RANGE[1]):
            raise ValueError(
                f"yaw must be within {YAW_RANGE[0]:g}..{YAW_RANGE[1]:g} "
                f"and pitch within {PITCH_RANGE[0]:g}..{PITCH_RANGE[1]:g} degrees"
            )
        return cls(yaw=yaw, pitch=pitch)

    def near(self, other: "Pose") -> bool:
        around = abs((self.yaw - other.yaw + 180) % 360 - 180)
        return (
            around <= REACH_TOLERANCE_DEGREES
            and abs(self.pitch - other.pitch) <= REACH_TOLERANCE_DEGREES
        )


@dataclass(frozen=True)
class Plan:
    """Where the camera rests, the stops it sweeps, and how long it rests before a sweep."""

    home: Pose
    stops: Tuple[Pose, ...]
    patrol_after_seconds: float

    def as_dict(self) -> dict:
        return {
            "home": self.home.as_dict(),
            "stops": [stop.as_dict() for stop in self.stops],
            "patrol_after": self.patrol_after_seconds,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Plan":
        """The plan in `data`; ValueError when it is not one the camera should follow."""
        if not isinstance(data, dict):
            raise ValueError("a plan is a JSON object")
        stops = data.get("stops", [])
        if not isinstance(stops, list) or len(stops) > MAX_STOPS:
            raise ValueError(f"stops must be a list of at most {MAX_STOPS} poses")
        try:
            patrol_after = float(data["patrol_after"])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError("patrol_after must be a number of seconds")
        if not (SHORTEST_PATROL_AFTER_SECONDS <= patrol_after < math.inf):
            raise ValueError(
                f"patrol_after must be at least {SHORTEST_PATROL_AFTER_SECONDS:g} seconds"
            )
        return cls(
            home=Pose.from_dict(data.get("home")),
            stops=tuple(Pose.from_dict(stop) for stop in stops),
            patrol_after_seconds=patrol_after,
        )


class Patrol:
    def __init__(
        self,
        home: Pose,
        stops: Sequence[Pose],
        home_after_seconds: float,
        patrol_after_seconds: float,
        dwell_seconds: float,
    ) -> None:
        self.home = home
        self.stops: List[Pose] = list(stops)
        self.home_after_seconds = home_after_seconds
        self.patrol_after_seconds = patrol_after_seconds
        self.dwell_seconds = dwell_seconds
        # Where the gimbal was last sent; None when something else has moved it.
        self._at: Optional[Pose] = None
        self._stop_index: Optional[int] = None
        self._arrived_at = 0.0
        self._home_since = 0.0
        # Poses the gimbal was sent to and, after waiting there, was not at: pose asked for ->
        # where the motors said it was instead. They outlast a change of plan, since what the
        # gimbal can reach does not depend on the plan.
        self.missed: Dict[Pose, Pose] = {}

    def plan(self) -> Plan:
        return Plan(self.home, tuple(self.stops), self.patrol_after_seconds)

    def follow(self, plan: Plan) -> None:
        """Rest and sweep by `plan` from now on. The gimbal goes to the new home the next time
        it is quiet."""
        self.home = plan.home
        self.stops = list(plan.stops)
        self.patrol_after_seconds = plan.patrol_after_seconds
        self.displaced()

    def displaced(self) -> None:
        """Something else (tracking, a manual move, a new plan) moved the gimbal or will."""
        self._at = None
        self._stop_index = None

    def clock_stepped(self, seconds: float) -> None:
        """The clock the caller's times come from was moved by `seconds`."""
        self._arrived_at += seconds
        self._home_since += seconds

    def next_pose(
        self, now: float, last_activity_at: float, pointing: Optional[Pose] = None
    ) -> Optional[Pose]:
        """The pose to move to now, or None to stay put. `pointing` is where the motors say the
        camera points; with it, a pose the gimbal has rested at is checked on leaving it."""
        if now - last_activity_at < self.home_after_seconds:
            self._stop_index = None
            return None
        if self._at != self.home and self._stop_index is None:
            return self._go_home(now)
        if self._stop_index is None:
            idle_since = max(self._home_since, last_activity_at)
            if self.stops and now - idle_since >= self.patrol_after_seconds:
                self._check_reached(pointing)
                return self._go_to_stop(0, now)
            return None
        if now - self._arrived_at < self.dwell_seconds:
            return None
        self._check_reached(pointing)
        if self._stop_index + 1 >= len(self.stops):
            return self._go_home(now)
        return self._go_to_stop(self._stop_index + 1, now)

    def status(self) -> dict:
        return {
            "at": None if self._at is None else self._at.as_dict(),
            "patrolling": self._stop_index is not None,
        }

    def _check_reached(self, pointing: Optional[Pose]) -> None:
        """Note whether the gimbal is at the pose it was last sent to, now that it has had a
        rest's worth of time to get there."""
        asked = self._at
        if asked is None or pointing is None:
            return
        if pointing.near(asked):
            self.missed.pop(asked, None)
        else:
            reached = Pose(round(pointing.yaw, 1), round(pointing.pitch, 1))
            if self.missed.get(asked) != reached:
                log.warning(
                    "sent to %s, the gimbal stopped at %s", asked.as_dict(), reached.as_dict()
                )
            self.missed[asked] = reached

    def _go_home(self, now: float) -> Pose:
        self._at = self.home
        self._stop_index = None
        self._home_since = now
        return self.home

    def _go_to_stop(self, index: int, now: float) -> Pose:
        self._stop_index = index
        self._arrived_at = now
        self._at = self.stops[index]
        return self._at


class Plans:
    """The plan the patrol follows: the one the agent was started with until another is set,
    and a set one is kept in `state_file` so it is still followed after a restart."""

    def __init__(self, patrol: Patrol, state_file: Optional[Path] = None) -> None:
        self.patrol = patrol
        self.default = patrol.plan()
        self.state_file = state_file
        self.learned = False
        saved = self._load()
        if saved is not None:
            self.patrol.follow(saved)
            self.learned = True

    def status(self) -> dict:
        return {
            "plan": self.patrol.plan().as_dict(),
            "default": self.default.as_dict(),
            "learned": self.learned,
            "missed": [
                {"asked": asked.as_dict(), "reached": reached.as_dict()}
                for asked, reached in self.patrol.missed.items()
            ],
        }

    def forget_missed(self) -> None:
        """Take every pose to be within the gimbal's reach again, until it shows otherwise."""
        self.patrol.missed.clear()

    def set(self, plan: Plan) -> None:
        self.patrol.follow(plan)
        self.learned = True
        if self.state_file is not None:
            save_whole(self.state_file, json.dumps(plan.as_dict()))

    def reset(self) -> None:
        """Back to the plan the agent was started with."""
        self.patrol.follow(self.default)
        self.learned = False
        if self.state_file is not None:
            remove(self.state_file)

    def _load(self) -> Optional[Plan]:
        if self.state_file is None or not self.state_file.exists():
            return None
        try:
            return Plan.from_dict(json.loads(self.state_file.read_text()))
        except (OSError, ValueError) as error:
            log.error("saved plan %s is unusable (%s); using the default", self.state_file, error)
            return None

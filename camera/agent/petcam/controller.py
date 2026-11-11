# ABOUTME: Decides what to do with each detection frame: start or stop recording a pet visit and
# ABOUTME: nudge the gimbal toward the pet. Reaches the gimbal and recorder through two interfaces.

import math
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Protocol, Sequence, Tuple

from petcam.detections import (
    DEFAULT_PETS,
    Box,
    Frame,
    class_ids,
    class_names,
    select_target,
)
from petcam.gaze import GazeLog
from petcam.motion import Motion
from petcam.patrol import Patrol, Pose
from petcam.presence import Presence, VisitEvent
from petcam.tracking import (
    HORIZONTAL_FOV_DEGREES,
    VERTICAL_FOV_DEGREES,
    Tracker,
    degrees_to_pixels,
)
from petcam.visits import VisitLog

# A pet counts as moving when it has travelled this far (in model pixels) within the window.
MOTION_WINDOW_SECONDS = 2.0
MOTION_THRESHOLD_PX = 60

# Speed for going home and between patrol stops, in motor units.
POSE_MOVE_SPEED = 90
# A reading of the motors' angles older than this no longer says where the camera points.
POSE_FRESH_SECONDS = 2.0
# Readings kept from just before a visit starts, so the visit's record begins where it began.
RECENT_POSES = 30


class GimbalControl(Protocol):
    def move(
        self, yaw_degrees: float, yaw_speed: int, pitch_degrees: float, pitch_speed: int
    ) -> None: ...

    def move_to(self, yaw_degrees: float, pitch_degrees: float, speed: int) -> None: ...


class RecorderControl(Protocol):
    def set_recording(self, enabled: bool) -> None: ...


class Controller:
    def __init__(
        self,
        gimbal: GimbalControl,
        recorder: RecorderControl,
        gone_after_seconds: float,
        move_interval_seconds: float,
        pets: Sequence[int] = class_ids(DEFAULT_PETS),
        state_file: Optional[Path] = None,
        confirm_sightings: int = 1,
        max_visit_seconds: Optional[float] = None,
        patrol: Optional[Patrol] = None,
        boot_id: Optional[str] = None,
        gaze: Optional[GazeLog] = None,
    ) -> None:
        self.gimbal = gimbal
        self.patrol = patrol
        self.gaze = gaze
        # Where the motors last said the camera points, and when they said it.
        self.pose: Optional[Pose] = None
        self._pose_at: Optional[float] = None
        self._recent_poses: Deque[Tuple[float, Pose]] = deque(maxlen=RECENT_POSES)
        self.recorder = recorder
        # Class numbers of the animals that start a visit and are followed.
        self.pets = tuple(pets)
        self.presence = Presence(
            gone_after_seconds=gone_after_seconds,
            confirm_sightings=confirm_sightings,
            max_visit_seconds=max_visit_seconds,
        )
        self.tracker = Tracker()
        self.motion = Motion(window_seconds=MOTION_WINDOW_SECONDS, threshold_px=MOTION_THRESHOLD_PX)
        # How far the gimbal has panned and tilted, in pixels, so pet positions can be compared
        # across frames even though the camera follows the pet.
        self._pan_px = 0.0
        self._tilt_px = 0.0
        # Last time a pet was seen, or when the agent started; the patrol counts quiet from here.
        self._last_activity_at: Optional[float] = None
        self.visits = VisitLog(state_file=state_file, boot_id=boot_id)
        self.move_interval_seconds = move_interval_seconds
        self.tracking_enabled = True
        self.frames_seen = 0
        self.last_pet: Optional[Box] = None
        self._last_move_at: Optional[float] = None

    def observe(self, frame: Frame, now: float) -> None:
        """Handle one detection frame from the model node."""
        self.frames_seen += 1
        pet = select_target(frame.boxes, self.pets)
        moving = False
        if pet is not None:
            self._last_activity_at = now
            self.motion.add(pet.cx + self._pan_px, pet.cy + self._tilt_px, now)
            moving = self.motion.moving(now)
        self._apply_presence(
            self.presence.update(pet_seen=pet is not None, now=now, moving=moving), now
        )
        if pet is None:
            self._idle(now)
            return
        self.last_pet = pet
        self.visits.sighting(pet, now)
        if self.tracking_enabled:
            self._follow(pet, frame, now)

    def pointing(self, pose: Optional[Pose], now: float) -> None:
        """The motors were just asked where the camera points; None when they did not answer.
        During a visit the answer goes into the visit's record; between visits the time since
        the last answer is added to the log of where the camera has looked."""
        last_at = self._pose_at
        self.pose = pose
        self._pose_at = now if pose is not None else None
        if pose is None:
            return
        self._recent_poses.append((now, pose))
        if self.presence.in_visit:
            self.visits.looking(pose.yaw, pose.pitch, now)
        elif self.gaze is not None and last_at is not None:
            if 0 < now - last_at <= POSE_FRESH_SECONDS:
                self.gaze.add(pose, now - last_at, now)

    def tick(self, now: float) -> None:
        """Advance time without a frame, so a visit still ends if detections stop arriving."""
        self._apply_presence(self.presence.update(pet_seen=False, now=now), now)
        self._idle(now)

    def clock_stepped(self, seconds: float, now: float) -> None:
        """The clock that `now` values come from is about to move by `seconds`; `now` is its
        last reading before the move.

        A visit in progress ends here, timed entirely by the clock as it was, which is also the
        clock its clips were stamped with. The pet, if still there, starts a new visit on the
        new clock.
        """
        if self.presence.in_visit:
            # Marked over first, as in shutdown: if stopping the recording fails, shutdown
            # must not end the visit a second time.
            self.presence.in_visit = False
            self._apply_presence(VisitEvent.ENDED, now)
        self.presence.reset()
        self.motion.clear()
        self.visits.clock_stepped(seconds)
        if self._last_activity_at is not None:
            self._last_activity_at += seconds
        if self._last_move_at is not None:
            self._last_move_at += seconds
        if self._pose_at is not None:
            self._pose_at += seconds
        self._recent_poses.clear()
        if self.patrol is not None:
            self.patrol.clock_stepped(seconds)

    def shutdown(self, now: Optional[float] = None) -> None:
        """Stop any recording in progress so the save node is not left running without the
        agent. `now` is the time by the clock the visit was timed with."""
        if self.presence.in_visit:
            self.presence.in_visit = False
            self._apply_presence(VisitEvent.ENDED, time.time() if now is None else now)

    def moved_by_hand(self) -> None:
        """The gimbal was turned from outside, so it is no longer where the patrol put it."""
        if self.patrol is not None:
            self.patrol.displaced()

    def status(self) -> dict:
        status = {
            "pets": class_names(self.pets),
            "in_visit": self.presence.in_visit,
            "settled": self.presence.settled,
            "visit_started_at": self.presence.started_at,
            "last_seen_at": self.presence.last_seen_at,
            "tracking_enabled": self.tracking_enabled,
            "frames_seen": self.frames_seen,
            "last_pet": (
                None
                if self.last_pet is None
                else {
                    "cx": self.last_pet.cx,
                    "cy": self.last_pet.cy,
                    "w": self.last_pet.w,
                    "h": self.last_pet.h,
                    "score": self.last_pet.score,
                }
            ),
        }
        status["pose"] = None if self.pose is None else self.pose.as_dict()
        if self.patrol is not None:
            status["patrol"] = self.patrol.status()
        return status

    def _idle(self, now: float) -> None:
        """With no pet in sight, let the patrol decide whether the gimbal should move."""
        if self._last_activity_at is None:
            self._last_activity_at = now
        if self.patrol is None or not self.tracking_enabled:
            return
        pose = self.patrol.next_pose(now, self._last_activity_at, self.pose)
        if pose is not None:
            self.gimbal.move_to(pose.yaw, pose.pitch, POSE_MOVE_SPEED)

    def _apply_presence(self, event: Optional[VisitEvent], now: float) -> None:
        if event == VisitEvent.STARTED:
            visit = self.visits.start(
                self.presence.started_at if self.presence.started_at is not None else now
            )
            # The sightings that opened the visit came a moment before it was sure to be one.
            for at, pose in self._recent_poses:
                if at >= visit.started_at - 1:
                    self.visits.looking(pose.yaw, pose.pitch, at)
            self.recorder.set_recording(True)
        elif event == VisitEvent.ENDED:
            # The visit is saved first: a recorder that cannot be reached must not lose it.
            self.visits.end(now)
            self.recorder.set_recording(False)

    def _follow(self, pet: Box, frame: Frame, now: float) -> None:
        if self._last_move_at is not None and now - self._last_move_at < self.move_interval_seconds:
            return
        move = self.tracker.move_for(pet, frame.resolution)
        if not move.is_motion:
            return
        self._last_move_at = now
        self.gimbal.move(move.yaw_degrees, move.yaw_speed, move.pitch_degrees, move.pitch_speed)
        if self.patrol is not None:
            self.patrol.displaced()
        # The move is always toward the pet, so the picture shifts in the direction of its offset.
        width, height = frame.resolution
        self._pan_px += math.copysign(
            degrees_to_pixels(abs(move.yaw_degrees), HORIZONTAL_FOV_DEGREES, width),
            pet.cx - width / 2,
        )
        self._tilt_px += math.copysign(
            degrees_to_pixels(abs(move.pitch_degrees), VERTICAL_FOV_DEGREES, height),
            pet.cy - height / 2,
        )

# ABOUTME: Tests for the controller that turns detection frames into recorder and gimbal actions.
# ABOUTME: Gimbal and recorder are recording fakes; the decision logic under test is real.

from typing import List, Tuple

import pytest

from petcam.controller import Controller
from petcam.detections import Box, Frame
from petcam.patrol import Patrol, Pose

RESOLUTION = (640, 640)


class FakeGimbal:
    def __init__(self) -> None:
        self.moves: List[Tuple[float, int, float, int]] = []
        self.poses: List[Tuple[float, float, int]] = []

    def move_to(self, yaw_degrees: float, pitch_degrees: float, speed: int) -> None:
        self.poses.append((yaw_degrees, pitch_degrees, speed))

    def move(
        self, yaw_degrees: float, yaw_speed: int, pitch_degrees: float, pitch_speed: int
    ) -> None:
        self.moves.append((yaw_degrees, yaw_speed, pitch_degrees, pitch_speed))


class FakeRecorder:
    def __init__(self) -> None:
        self.calls: List[bool] = []

    def set_recording(self, enabled: bool) -> None:
        self.calls.append(enabled)


def frame_with(boxes: List[Box], timestamp_ms: int = 0) -> Frame:
    return Frame(boxes=boxes, resolution=RESOLUTION, timestamp_ms=timestamp_ms)


def pet_at(cx: float, cy: float) -> Box:
    return Box(cx=cx, cy=cy, w=80, h=80, score=0.8, class_id=15)


def person_at(cx: float, cy: float) -> Box:
    return Box(cx=cx, cy=cy, w=80, h=80, score=0.8, class_id=0)


def make() -> Tuple[Controller, FakeGimbal, FakeRecorder]:
    gimbal, recorder = FakeGimbal(), FakeRecorder()
    controller = Controller(
        gimbal=gimbal,
        recorder=recorder,
        gone_after_seconds=60,
        move_interval_seconds=0.3,
    )
    return controller, gimbal, recorder


def test_pet_arrival_starts_recording_and_departure_stops_it() -> None:
    controller, _, recorder = make()
    controller.observe(frame_with([pet_at(320, 320)]), now=100.0)
    assert recorder.calls == [True]
    controller.observe(frame_with([]), now=130.0)
    assert recorder.calls == [True]
    controller.observe(frame_with([]), now=160.0)
    assert recorder.calls == [True, False]


def test_person_alone_is_not_a_visit() -> None:
    controller, gimbal, recorder = make()
    controller.observe(frame_with([person_at(500, 320)]), now=100.0)
    assert recorder.calls == []
    assert gimbal.moves == []


def test_off_center_pet_moves_gimbal_when_tracking_enabled() -> None:
    controller, gimbal, _ = make()
    controller.observe(frame_with([pet_at(500, 320)]), now=100.0)
    assert len(gimbal.moves) == 1
    yaw, _, pitch, _ = gimbal.moves[0]
    assert yaw > 0 and pitch == 0.0


def test_centered_pet_does_not_move_gimbal() -> None:
    controller, gimbal, _ = make()
    controller.observe(frame_with([pet_at(320, 320)]), now=100.0)
    assert gimbal.moves == []


def test_moves_are_throttled() -> None:
    controller, gimbal, _ = make()
    controller.observe(frame_with([pet_at(500, 320)]), now=100.0)
    controller.observe(frame_with([pet_at(500, 320)]), now=100.1)
    controller.observe(frame_with([pet_at(500, 320)]), now=100.2)
    assert len(gimbal.moves) == 1
    controller.observe(frame_with([pet_at(500, 320)]), now=100.4)
    assert len(gimbal.moves) == 2


def test_tracking_can_be_disabled_without_affecting_recording() -> None:
    controller, gimbal, recorder = make()
    controller.tracking_enabled = False
    controller.observe(frame_with([pet_at(500, 320)]), now=100.0)
    assert gimbal.moves == []
    assert recorder.calls == [True]


def test_status_reports_visit_and_last_target() -> None:
    controller, _, _ = make()
    assert controller.status()["in_visit"] is False
    controller.observe(frame_with([pet_at(500, 320)], timestamp_ms=1234), now=100.0)
    status = controller.status()
    assert status["in_visit"] is True
    assert status["tracking_enabled"] is True
    assert status["visit_started_at"] == 100.0
    assert status["last_pet"] == {"cx": 500, "cy": 320, "w": 80, "h": 80, "score": 0.8}
    assert status["frames_seen"] == 1


def test_idle_tick_without_frames_still_ends_visit() -> None:
    controller, _, recorder = make()
    controller.observe(frame_with([pet_at(320, 320)]), now=100.0)
    controller.tick(now=161.0)
    assert recorder.calls == [True, False]


def test_cats_and_dogs_are_followed_unless_told_otherwise() -> None:
    controller, _, recorder = make()
    dog = Box(cx=320, cy=320, w=80, h=80, score=0.8, class_id=16)
    controller.observe(frame_with([person_at(320, 320)]), now=100.0)
    assert recorder.calls == []
    controller.observe(frame_with([dog]), now=101.0)
    assert recorder.calls == [True]


def test_the_animals_followed_are_configurable() -> None:
    gimbal, recorder = FakeGimbal(), FakeRecorder()
    controller = Controller(
        gimbal=gimbal,
        recorder=recorder,
        gone_after_seconds=60,
        move_interval_seconds=0.3,
        pets=[0],
    )
    controller.observe(frame_with([person_at(320, 320)]), now=100.0)
    assert recorder.calls == [True]
    controller.observe(frame_with([pet_at(320, 320)]), now=200.0)
    assert recorder.calls == [True, False]


def test_shutdown_stops_an_active_recording() -> None:
    controller, _, recorder = make()
    controller.observe(frame_with([pet_at(320, 320)]), now=100.0)
    controller.shutdown()
    assert recorder.calls == [True, False]
    assert not controller.presence.in_visit


def test_shutdown_without_visit_leaves_recorder_alone() -> None:
    controller, _, recorder = make()
    controller.shutdown()
    assert recorder.calls == []


def test_visits_are_logged_with_sightings() -> None:
    controller, _, _ = make()
    controller.observe(frame_with([pet_at(320, 320)]), now=100.0)
    controller.observe(frame_with([pet_at(330, 320)]), now=101.0)
    controller.observe(frame_with([]), now=170.0)
    visits = controller.visits.finished()
    assert len(visits) == 1
    assert visits[0].started_at == 100.0
    assert visits[0].ended_at == 170.0
    assert [s.at for s in visits[0].sightings] == [100.0, 101.0]


def test_controller_can_require_confirmed_sightings() -> None:
    gimbal, recorder = FakeGimbal(), FakeRecorder()
    controller = Controller(
        gimbal=gimbal,
        recorder=recorder,
        gone_after_seconds=60,
        move_interval_seconds=0.3,
        confirm_sightings=3,
    )
    controller.observe(frame_with([pet_at(320, 320)]), now=100.0)
    controller.observe(frame_with([pet_at(320, 320)]), now=100.1)
    assert recorder.calls == []
    controller.observe(frame_with([pet_at(320, 320)]), now=100.2)
    assert recorder.calls == [True]
    assert controller.visits.open is not None and controller.visits.open.started_at == 100.0


def make_capped() -> Tuple[Controller, FakeGimbal, FakeRecorder]:
    gimbal, recorder = FakeGimbal(), FakeRecorder()
    controller = Controller(
        gimbal=gimbal,
        recorder=recorder,
        gone_after_seconds=60,
        move_interval_seconds=0.3,
        max_visit_seconds=600,
    )
    return controller, gimbal, recorder


def test_capped_visit_stops_recording_while_pet_is_still_there() -> None:
    controller, _, recorder = make_capped()
    controller.observe(frame_with([pet_at(320, 320)]), now=100.0)
    controller.observe(frame_with([pet_at(320, 320)]), now=700.0)
    assert recorder.calls == [True, False]
    assert len(controller.visits.finished()) == 1
    assert controller.status()["settled"] is True


def test_settled_pet_that_walks_starts_a_new_visit() -> None:
    controller, _, recorder = make_capped()
    controller.observe(frame_with([pet_at(320, 320)]), now=100.0)
    controller.observe(frame_with([pet_at(320, 320)]), now=700.0)
    # Jitter of a few pixels per frame is not walking.
    for i in range(20):
        controller.observe(frame_with([pet_at(320 + (i % 2) * 3, 320)]), now=701.0 + i * 0.1)
    assert recorder.calls == [True, False]
    # Fifty pixels a frame for a second is.
    for i in range(10):
        controller.observe(frame_with([pet_at(100 + i * 50, 320)]), now=710.0 + i * 0.1)
    assert recorder.calls == [True, False, True]
    assert controller.status()["settled"] is False


def test_tracked_pet_motion_counts_the_gimbal_move() -> None:
    # The gimbal re-centers a walking pet, so its pixel position barely changes; the commanded
    # move stands in for the shift the camera cancelled.
    controller, gimbal, recorder = make_capped()
    controller.observe(frame_with([pet_at(320, 320)]), now=100.0)
    controller.observe(frame_with([pet_at(320, 320)]), now=700.0)
    for i in range(10):
        controller.observe(frame_with([pet_at(420, 320)]), now=710.0 + i * 0.4)
    assert len(gimbal.moves) >= 3
    assert recorder.calls == [True, False, True]


HOME = Pose(yaw=225, pitch=120)
STOP = Pose(yaw=15, pitch=85)


def make_with_patrol() -> Tuple[Controller, FakeGimbal, FakeRecorder]:
    gimbal, recorder = FakeGimbal(), FakeRecorder()
    controller = Controller(
        gimbal=gimbal,
        recorder=recorder,
        gone_after_seconds=60,
        move_interval_seconds=0.3,
        patrol=Patrol(
            home=HOME,
            stops=[STOP],
            home_after_seconds=90,
            patrol_after_seconds=600,
            dwell_seconds=25,
        ),
    )
    return controller, gimbal, recorder


def test_idle_camera_goes_home_then_patrols() -> None:
    controller, gimbal, _ = make_with_patrol()
    controller.tick(now=100.0)
    controller.tick(now=189.0)
    assert gimbal.poses == []
    controller.tick(now=190.0)
    assert gimbal.poses == [(225, 120, 90)]
    controller.tick(now=790.0)
    assert gimbal.poses[-1] == (15, 85, 90)
    controller.tick(now=815.0)
    assert gimbal.poses[-1] == (225, 120, 90)


def test_tracking_a_pet_displaces_the_camera_and_it_returns_home_after_the_visit() -> None:
    controller, gimbal, _ = make_with_patrol()
    controller.tick(now=100.0)
    controller.tick(now=190.0)
    controller.observe(frame_with([pet_at(500, 320)]), now=200.0)
    assert len(gimbal.moves) == 1
    controller.observe(frame_with([]), now=260.0)
    controller.tick(now=289.0)
    assert gimbal.poses == [(225, 120, 90)]
    controller.tick(now=290.0)
    assert gimbal.poses == [(225, 120, 90), (225, 120, 90)]


def test_patrol_waits_while_tracking_is_disabled() -> None:
    controller, gimbal, _ = make_with_patrol()
    controller.tracking_enabled = False
    controller.tick(now=100.0)
    controller.tick(now=5000.0)
    assert gimbal.poses == []
    controller.tracking_enabled = True
    controller.tick(now=5001.0)
    assert gimbal.poses == [(225, 120, 90)]


def test_status_reports_the_patrol() -> None:
    controller, _, _ = make_with_patrol()
    controller.tick(now=100.0)
    controller.tick(now=190.0)
    assert controller.status()["patrol"] == {"at": {"yaw": 225, "pitch": 120}, "patrolling": False}


def test_without_a_patrol_the_camera_stays_where_it_is() -> None:
    controller, gimbal, _ = make()
    controller.tick(now=100.0)
    controller.tick(now=5000.0)
    assert gimbal.poses == []
    assert "patrol" not in controller.status()


def test_a_clock_step_ends_the_visit_on_the_old_clock_and_a_new_one_starts_on_the_new(
    tmp_path,
) -> None:
    gimbal, recorder = FakeGimbal(), FakeRecorder()
    controller = Controller(
        gimbal=gimbal,
        recorder=recorder,
        gone_after_seconds=60,
        move_interval_seconds=0.3,
        state_file=tmp_path / "visits.json",
        boot_id="this-boot",
    )
    # The camera booted without a clock: it is 1970 while the pet visits.
    controller.observe(frame_with([pet_at(320, 320)]), now=500.0)
    controller.observe(frame_with([pet_at(320, 320)]), now=510.0)
    assert controller.presence.in_visit
    # NTP answers; the system clock moves 56 years.
    step = 1_800_000_000.0
    controller.clock_stepped(step, now=511.0)
    assert not controller.presence.in_visit
    (visit,) = controller.visits.finished()
    assert (visit.started_at, visit.ended_at, visit.clock_offset) == (500.0, 511.0, step)
    # The pet is still there and is seen on the new clock.
    controller.observe(frame_with([pet_at(320, 320)]), now=step + 512.0)
    assert controller.presence.in_visit
    controller.tick(now=step + 600.0)
    first, second = controller.visits.finished()
    assert (second.started_at, second.ended_at, second.clock_offset) == (
        step + 512.0,
        step + 600.0,
        0.0,
    )


def test_a_clock_step_with_no_visit_does_not_start_or_end_one() -> None:
    controller, _, recorder = make()
    controller.tick(now=100.0)
    controller.clock_stepped(1_800_000_000.0, now=101.0)
    controller.tick(now=1_800_000_102.0)
    assert not controller.presence.in_visit
    assert controller.visits.finished() == []


def test_a_visit_is_saved_even_when_the_recorder_cannot_be_told_to_stop() -> None:
    controller, _, recorder = make()
    controller.observe(frame_with([pet_at(320, 320)]), now=100.0)

    def broken(enabled: bool) -> None:
        raise BrokenPipeError("broker went away")

    recorder.set_recording = broken
    with pytest.raises(BrokenPipeError):
        controller.shutdown()
    assert len(controller.visits.finished()) == 1


def test_a_visit_ended_by_a_clock_step_is_not_ended_again_at_shutdown() -> None:
    controller, _, recorder = make()
    controller.observe(frame_with([pet_at(320, 320)]), now=100.0)

    def broken(enabled: bool) -> None:
        raise BrokenPipeError("broker went away")

    recorder.set_recording = broken
    with pytest.raises(BrokenPipeError):
        controller.clock_stepped(3600.0, now=101.0)
    controller.shutdown()
    assert len(controller.visits.finished()) == 1


def test_a_gimbal_moved_by_hand_is_sent_home_again_once_it_is_quiet() -> None:
    controller, gimbal, _ = make_with_patrol()
    controller.tick(now=100.0)
    controller.tick(now=190.0)
    assert gimbal.poses == [(225, 120, 90)]
    controller.moved_by_hand()
    controller.tick(now=400.0)
    assert gimbal.poses == [(225, 120, 90), (225, 120, 90)]

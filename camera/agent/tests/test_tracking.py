# ABOUTME: Tests for turning a pet's position in the frame into gimbal yaw/pitch moves.
# ABOUTME: Checks dead zone, sign conventions, clamping, and the speed curve.

import pytest

from petcam.detections import Box
from petcam.tracking import degrees_to_pixels, Move, Tracker, pixel_offset_to_degrees

RESOLUTION = (640, 640)


def centered(cx: float, cy: float) -> Box:
    return Box(cx=cx, cy=cy, w=50, h=50, score=0.9, class_id=15)


def test_offset_inside_dead_zone_gives_no_movement() -> None:
    assert pixel_offset_to_degrees(3, fov_degrees=60, resolution_px=640) == 0.0
    assert pixel_offset_to_degrees(-5, fov_degrees=60, resolution_px=640) == 0.0


def test_offset_grows_with_distance_and_keeps_sign() -> None:
    near = pixel_offset_to_degrees(40, fov_degrees=60, resolution_px=640)
    far = pixel_offset_to_degrees(200, fov_degrees=60, resolution_px=640)
    assert 0 < near < far
    assert pixel_offset_to_degrees(-200, fov_degrees=60, resolution_px=640) == pytest.approx(-far)


def test_offset_is_clamped_to_max_angle() -> None:
    tracker = Tracker(max_angle_degrees=15)
    move = tracker.move_for(centered(639, 320), RESOLUTION)
    assert abs(move.yaw_degrees) <= 15


def test_pet_right_of_center_turns_yaw_positive_and_pitch_zero() -> None:
    # Verified on the gimbal: a positive yaw command brings a target on the right of the
    # (mirrored) frame toward the center.
    move = Tracker().move_for(centered(480, 320), RESOLUTION)
    assert move.yaw_degrees > 0
    assert move.pitch_degrees == 0.0


def test_pet_below_center_tilts_pitch_positive() -> None:
    move = Tracker().move_for(centered(320, 480), RESOLUTION)
    assert move.yaw_degrees == 0.0
    assert move.pitch_degrees > 0


def test_pet_at_center_is_no_move() -> None:
    move = Tracker().move_for(centered(320, 320), RESOLUTION)
    assert move == Move(yaw_degrees=0.0, pitch_degrees=0.0, yaw_speed=10, pitch_speed=10)
    assert not move.is_motion


def test_speed_rises_with_offset_within_limits() -> None:
    small = Tracker().move_for(centered(340, 320), RESOLUTION)
    large = Tracker().move_for(centered(600, 320), RESOLUTION)
    assert 10 <= small.yaw_speed < large.yaw_speed <= 2000


def test_degrees_to_pixels_is_the_picture_shift_for_a_turn() -> None:
    # Turning by half the field of view shifts the picture by half the frame.
    assert degrees_to_pixels(30, fov_degrees=60, resolution_px=640) == pytest.approx(320)
    assert degrees_to_pixels(0.0, fov_degrees=60, resolution_px=640) == 0.0

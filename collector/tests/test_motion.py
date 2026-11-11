# ABOUTME: Tests what a visit's record says was moving: how far the camera turned around a moment
# ABOUTME: and how fast the pet's box crossed the picture, and the stillness made of the two.

import pytest

from collector.motion import VisitMotion
from collector.scoring import Sighting

RESOLUTION = (640, 640)


def seen(at: float, cx: float, cy: float = 320.0) -> Sighting:
    return Sighting(at=at, cx=cx, cy=cy, w=100, h=100, score=0.9)


def test_gimbal_turn_adds_up_the_turns_noted_around_a_moment() -> None:
    gaze = [[100.0, 90.0, 10.0], [103.0, 93.0, 14.0], [103.2, 96.0, 14.0], [110.0, 96.0, 14.0]]
    motion = VisitMotion([], gaze, RESOLUTION)
    # Looking one way from 100 to 103, then a turn of 5 degrees and one of 3.
    assert motion.gimbal_turn(101.0) == 0.0
    assert motion.gimbal_turn(103.1) == pytest.approx(8.0)
    assert motion.gimbal_turn(103.6) == pytest.approx(3.0)
    # The note every ten seconds says the camera had not turned.
    assert motion.gimbal_turn(110.0) == 0.0


def test_gimbal_turn_is_unknown_without_a_record_of_where_the_camera_pointed() -> None:
    assert VisitMotion([], [], RESOLUTION).gimbal_turn(100.0) is None


def test_box_speed_is_the_boxes_travel_in_picture_widths_a_second() -> None:
    walking = [seen(100.0 + step / 10, cx=100 + 32 * step) for step in range(11)]
    motion = VisitMotion(walking, [], RESOLUTION)
    # 32 pixels every tenth of a second is half the 640 pixel picture a second.
    assert motion.box_speed(100.5) == pytest.approx(0.5)
    sitting = [seen(200.0 + step / 10, cx=320 + (step % 2)) for step in range(11)]
    assert VisitMotion(sitting, [], RESOLUTION).box_speed(200.5) == pytest.approx(0.0, abs=0.01)


def test_box_speed_is_unknown_for_a_pet_seen_only_once_around_the_moment() -> None:
    motion = VisitMotion([seen(100.0, 320), seen(105.0, 400)], [], RESOLUTION)
    assert motion.box_speed(100.0) is None
    assert motion.box_speed(50.0) is None


def test_stillness_is_whole_when_nothing_moved_and_falls_with_either_movement() -> None:
    sitting = [seen(100.0 + step / 10, cx=320) for step in range(21)]
    walking = [seen(200.0 + step / 10, cx=100 + 20 * step) for step in range(21)]
    gaze = [[99.0, 90.0, 10.0], [150.0, 90.0, 10.0], [150.2, 95.0, 10.0], [199.0, 95.0, 10.0]]
    motion = VisitMotion(sitting + walking, gaze, RESOLUTION)
    assert motion.stillness(101.0) == 1.0
    assert motion.stillness(201.0) < 0.5
    turning = VisitMotion([seen(150.0 + step / 10, cx=320) for step in range(5)], gaze, RESOLUTION)
    assert turning.stillness(150.2) < 0.5


def test_a_pet_whose_speed_is_unknown_is_worth_less_than_one_known_to_be_still() -> None:
    sitting = [seen(100.0 + step / 10, cx=320) for step in range(11)]
    motion = VisitMotion(sitting + [seen(300.0, 320)], [], RESOLUTION)
    assert motion.stillness(300.0) < motion.stillness(100.5)

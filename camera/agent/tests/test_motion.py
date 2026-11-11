# ABOUTME: Tests for the motion window that decides whether a tracked pet has moved recently.
# ABOUTME: Positions are fed in pixels with explicit times so the tests are deterministic.

from petcam.motion import Motion


def test_nothing_added_is_not_moving() -> None:
    assert not Motion(window_seconds=2.0, threshold_px=60).moving(now=10.0)


def test_jitter_around_one_spot_is_not_moving() -> None:
    motion = Motion(window_seconds=2.0, threshold_px=60)
    for i in range(20):
        motion.add(320 + (i % 2) * 4, 320 - (i % 3) * 3, now=10.0 + i * 0.1)
    assert not motion.moving(now=12.0)


def test_steady_drift_inside_the_window_is_moving() -> None:
    motion = Motion(window_seconds=2.0, threshold_px=60)
    for i in range(10):
        motion.add(320 + i * 8, 320, now=10.0 + i * 0.1)
    assert motion.moving(now=11.0)


def test_only_positions_inside_the_window_count() -> None:
    motion = Motion(window_seconds=2.0, threshold_px=60)
    motion.add(100, 320, now=10.0)
    motion.add(300, 320, now=11.0)
    assert motion.moving(now=11.9)
    assert not motion.moving(now=12.5)

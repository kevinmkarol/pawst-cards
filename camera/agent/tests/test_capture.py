# ABOUTME: Tests the capture watchdog: nothing while frames arrive, node requests repeated during
# ABOUTME: a gap, a restart after a long gap, and restarts spaced further apart each time.

from petcam.capture import CaptureWatchdog, Remedy


def make(now: float = 0.0) -> CaptureWatchdog:
    return CaptureWatchdog(
        now=now,
        create_every_seconds=15,
        restart_after_seconds=90,
        first_restart_gap_seconds=300,
        longest_restart_gap_seconds=1000,
    )


def remedies(watchdog: CaptureWatchdog, start: int, end: int) -> list:
    """(second, remedy) for every second in the range at which the watchdog asks for one."""
    found = []
    for second in range(start, end):
        remedy = watchdog.check(float(second))
        if remedy is not None:
            found.append((second, remedy))
    return found


def test_nothing_to_do_while_frames_arrive() -> None:
    watchdog = make()
    for second in range(0, 600):
        watchdog.frame(float(second))
        assert watchdog.check(second + 0.5) is None
    assert watchdog.status(600.0) == {"flowing": True, "seconds_since_frame": 1.0, "restarts": 0}


def test_node_requests_are_repeated_until_the_first_frame() -> None:
    watchdog = make()
    assert remedies(watchdog, 0, 50) == [
        (15, Remedy.CREATE),
        (30, Remedy.CREATE),
        (45, Remedy.CREATE),
    ]
    assert watchdog.frame(50.0) is True
    assert watchdog.frame(50.1) is False
    assert remedies(watchdog, 51, 60) == []


def test_a_long_gap_restarts_capture_and_restarts_get_further_apart() -> None:
    watchdog = make()
    watchdog.frame(0.0)
    restarts = [s for s, r in remedies(watchdog, 1, 2500) if r == Remedy.RESTART]
    assert restarts == [90, 390, 990, 1990]
    assert watchdog.status(2500.0) == {
        "flowing": False,
        "seconds_since_frame": 2500.0,
        "restarts": 4,
    }


def test_frames_after_a_restart_start_the_count_over() -> None:
    watchdog = make()
    watchdog.frame(0.0)
    assert [s for s, r in remedies(watchdog, 1, 100) if r == Remedy.RESTART] == [90]
    assert watchdog.frame(100.0) is True
    assert [s for s, r in remedies(watchdog, 101, 300) if r == Remedy.RESTART] == [190]


def test_status_before_any_frame() -> None:
    assert make().status(5.0) == {"flowing": False, "seconds_since_frame": None, "restarts": 0}


def test_a_replaced_camera_node_restarts_capture_at_once_though_frames_arrive() -> None:
    watchdog = make()
    watchdog.frame(0.0)
    watchdog.frame(200.0)
    watchdog.camera_replaced()
    assert watchdog.check(201.0) == Remedy.RESTART
    assert watchdog.check(202.0) is None
    # The restart may have put back a recorder that is off, so the next frame is a first one.
    assert watchdog.frame(203.0) is True


def test_a_camera_node_replaced_right_after_a_restart_does_not_restart_again() -> None:
    watchdog = make()
    watchdog.frame(0.0)
    watchdog.camera_replaced()
    assert watchdog.check(2.0) == Remedy.RESTART
    watchdog.frame(20.0)
    watchdog.camera_replaced()
    assert watchdog.check(22.0) is None
    watchdog.frame(100.0)
    watchdog.camera_replaced()
    assert watchdog.check(101.0) == Remedy.RESTART

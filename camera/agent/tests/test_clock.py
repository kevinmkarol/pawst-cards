# ABOUTME: Tests the agent's clock: steady time between resyncs, reporting a stepped system
# ABOUTME: clock, and saving and restoring the time across a boot with real files.

from pathlib import Path
from typing import List

from petcam.clock import Clock, restore_saved_time, save_time


class Clocks:
    """A system clock and a monotonic clock that the test moves by hand."""

    def __init__(self, wall: float, monotonic: float) -> None:
        self.wall_time = wall
        self.monotonic_time = monotonic

    def wall(self) -> float:
        return self.wall_time

    def monotonic(self) -> float:
        return self.monotonic_time

    def pass_seconds(self, seconds: float) -> None:
        self.wall_time += seconds
        self.monotonic_time += seconds


def test_time_advances_with_the_monotonic_clock() -> None:
    clocks = Clocks(wall=120.0, monotonic=50.0)
    clock = Clock(wall=clocks.wall, monotonic=clocks.monotonic)
    assert clock.now() == 120.0
    clocks.pass_seconds(10)
    assert clock.now() == 130.0


def test_a_stepped_system_clock_is_not_seen_until_resync_which_reports_it() -> None:
    clocks = Clocks(wall=120.0, monotonic=50.0)
    clock = Clock(wall=clocks.wall, monotonic=clocks.monotonic)
    clocks.pass_seconds(10)
    clocks.wall_time = 1_800_000_000.0  # NTP answers
    assert clock.now() == 130.0
    assert clock.resync() == 1_800_000_000.0 - 130.0
    assert clock.now() == 1_800_000_000.0
    assert clock.resync() == 0.0


def test_drift_is_followed_without_being_reported_as_a_step() -> None:
    clocks = Clocks(wall=1000.0, monotonic=50.0)
    clock = Clock(wall=clocks.wall, monotonic=clocks.monotonic)
    clocks.wall_time += 0.4
    assert clock.resync() == 0.0
    assert clock.now() == 1000.4


def test_saved_time_sets_a_clock_that_is_behind_it(tmp_path: Path) -> None:
    saved = tmp_path / "last_time"
    save_time(saved, 1_800_000_000.4)
    was_set: List[float] = []
    assert restore_saved_time(saved, wall=lambda: 95.0, set_wall=was_set.append) is True
    assert was_set == [1_800_000_000.0]


def test_saved_time_leaves_a_clock_that_is_ahead_of_it(tmp_path: Path) -> None:
    saved = tmp_path / "last_time"
    save_time(saved, 1_800_000_000.0)
    was_set: List[float] = []
    assert restore_saved_time(saved, wall=lambda: 1_800_000_500.0, set_wall=was_set.append) is False
    assert was_set == []


def test_no_saved_time_or_an_unreadable_one_leaves_the_clock(tmp_path: Path) -> None:
    was_set: List[float] = []
    assert (
        restore_saved_time(tmp_path / "missing", wall=lambda: 95.0, set_wall=was_set.append)
        is False
    )
    (tmp_path / "garbled").write_text("not a time")
    assert (
        restore_saved_time(tmp_path / "garbled", wall=lambda: 95.0, set_wall=was_set.append)
        is False
    )
    assert was_set == []


def test_a_clock_that_may_not_be_set_is_reported_not_raised(tmp_path: Path) -> None:
    saved = tmp_path / "last_time"
    save_time(saved, 1_800_000_000.0)

    def refuse(_: float) -> None:
        raise PermissionError("not root")

    assert restore_saved_time(saved, wall=lambda: 95.0, set_wall=refuse) is False

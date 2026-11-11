# ABOUTME: Tests how the agent saves its state files: whole or not at all, and a disk that cannot
# ABOUTME: be written to costs the saved copy without stopping the change or the agent.

import argparse
from pathlib import Path

import pytest

from petcam.agent import pose
from petcam.files import save_whole
from petcam.gaze import direction
from petcam.patrol import Patrol, Plan, Plans, Pose


def test_a_saved_file_replaces_the_old_one_and_leaves_nothing_beside_it(tmp_path: Path) -> None:
    path = tmp_path / "state" / "plan.json"
    assert save_whole(path, "one") and save_whole(path, "two")
    assert path.read_text() == "two"
    assert [entry.name for entry in path.parent.iterdir()] == ["plan.json"]


def test_a_plan_that_cannot_be_saved_is_still_followed(tmp_path: Path) -> None:
    (tmp_path / "full").write_text("a file where the directory should be")
    path = tmp_path / "full" / "plan.json"
    assert not save_whole(path, "one")
    plans = Plans(Patrol(Pose(180, 100), [Pose(90, 100)], 90, 600, 25), path)
    new = Plan(Pose(60, 100), (), 300)
    plans.set(new)
    assert plans.learned and plans.patrol.plan() == new


def test_a_pose_on_the_command_line_must_be_one_the_gimbal_can_be_sent_to() -> None:
    assert pose("345,85") == Pose(345, 85)
    for bad in ("nan,100", "400,100", "180", "left,up"):
        with pytest.raises(argparse.ArgumentTypeError):
            pose(bad)


def test_directions_just_short_of_a_full_turn_count_as_yaw_zero() -> None:
    assert direction(Pose(359, 120)) == direction(Pose(1, 120)) == "0,120"

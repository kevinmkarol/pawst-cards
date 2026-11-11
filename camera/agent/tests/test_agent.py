# ABOUTME: Tests for the agent's command line options.
# ABOUTME: Only argument parsing is covered here; the hardware loop is exercised on the camera.

import pytest

from petcam.agent import parse_args
from petcam.patrol import Pose


def test_visit_cap_defaults_to_ten_minutes() -> None:
    assert parse_args([]).max_visit == 600.0


def test_visit_cap_can_be_set() -> None:
    assert parse_args(["--max-visit", "120"]).max_visit == 120.0


def test_home_and_patrol_default_to_poses_around_the_middle_of_the_turn() -> None:
    args = parse_args([])
    assert args.home == Pose(yaw=180, pitch=100)
    assert args.patrol == [
        Pose(yaw=90, pitch=100),
        Pose(yaw=135, pitch=100),
        Pose(yaw=225, pitch=100),
        Pose(yaw=270, pitch=100),
    ]


def test_home_and_patrol_can_be_given_as_the_boot_script_passes_them() -> None:
    args = parse_args("--home 225,120 --patrol 345,85 155,120".split())
    assert args.home == Pose(yaw=225, pitch=120)
    assert args.patrol == [Pose(yaw=345, pitch=85), Pose(yaw=155, pitch=120)]


def test_home_and_patrol_can_be_given_as_yaw_comma_pitch() -> None:
    args = parse_args(["--home", "200,110", "--patrol", "10,80", "350,80", "--dwell", "5"])
    assert args.home == Pose(yaw=200, pitch=110)
    assert args.patrol == [Pose(yaw=10, pitch=80), Pose(yaw=350, pitch=80)]
    assert args.dwell == 5.0


def test_patrol_can_be_switched_off_with_no_stops() -> None:
    assert parse_args(["--patrol"]).patrol == []


def test_api_listens_on_every_interface_unless_told_otherwise() -> None:
    assert parse_args([]).api_host == "0.0.0.0"
    assert parse_args(["--api-host", "127.0.0.1"]).api_host == "127.0.0.1"


def test_pets_default_to_cats_and_dogs_and_take_names_or_numbers() -> None:
    assert parse_args([]).pets == (15, 16)
    assert parse_args("--pets bird 17".split()).pets == (14, 17)


def test_an_unknown_pet_stops_the_agent_with_the_names_it_knows(capsys) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--pets", "rabbit"])
    assert "it knows: person" in capsys.readouterr().err

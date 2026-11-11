# ABOUTME: Tests the timings that can be changed on a running agent: they take effect on the
# ABOUTME: controller at once, survive a restart, are refused when out of range, and can be undone.

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Tuple

import pytest

from petcam.api import ControlApi
from petcam.controller import Controller
from petcam.patrol import Patrol, Pose
from petcam.tuning import Tuning


class Idle:
    def move(self, *args: object) -> None:
        pass

    def move_to(self, *args: object) -> None:
        pass

    def set_recording(self, enabled: bool) -> None:
        pass


def controller(patrol: bool = True) -> Controller:
    idle = Idle()
    return Controller(
        gimbal=idle,
        recorder=idle,
        gone_after_seconds=60,
        move_interval_seconds=0.3,
        confirm_sightings=3,
        max_visit_seconds=600,
        patrol=Patrol(Pose(180, 100), [Pose(90, 100)], 90, 600, 25) if patrol else None,
    )


def test_timings_start_as_the_agent_was_started_and_change_on_the_controller() -> None:
    running = controller()
    tuning = Tuning(running)
    assert tuning.status()["values"] == {
        "gone_after": 60,
        "max_visit": 600,
        "confirm_sightings": 3,
        "home_after": 90,
        "dwell": 25,
    }
    assert tuning.status()["changed"] is False
    tuning.set({"gone_after": 30, "dwell": 40, "confirm_sightings": 2})
    assert running.presence.gone_after_seconds == 30.0
    assert running.presence.confirm_sightings == 2
    assert running.patrol.dwell_seconds == 40.0
    assert running.presence.max_visit_seconds == 600
    assert tuning.status()["changed"] is True
    assert tuning.status()["default"]["gone_after"] == 60


def test_a_bad_timing_changes_nothing(tmp_path: Path) -> None:
    running = controller()
    tuning = Tuning(running, tmp_path / "tuning.json")
    for bad in (
        {"gone_after": 30, "dwell": 1},
        {"gone_after": 5},
        {"gone_after": float("nan")},
        {"gone_after": "soon"},
        {"confirm_sightings": 2.5},
        {"confirm_sightings": True},
        {"speed": 3},
    ):
        with pytest.raises(ValueError):
            tuning.set(bad)
    assert running.presence.gone_after_seconds == 60
    assert not (tmp_path / "tuning.json").exists()


def test_changed_timings_survive_a_restart_and_can_be_undone(tmp_path: Path) -> None:
    path = tmp_path / "petcam" / "tuning.json"
    Tuning(controller(), path).set({"gone_after": 45, "home_after": 120})
    restarted = controller()
    tuning = Tuning(restarted, path)
    assert restarted.presence.gone_after_seconds == 45.0
    assert restarted.patrol.home_after_seconds == 120.0
    assert tuning.changed and tuning.default["gone_after"] == 60
    tuning.reset()
    assert restarted.presence.gone_after_seconds == 60
    assert not path.exists()
    path.write_text(json.dumps({"gone_after": 99999, "dwell": "long", "max_visit": 900}))
    again = controller()
    Tuning(again, path)
    assert again.presence.gone_after_seconds == 60
    assert again.presence.max_visit_seconds == 900.0


def test_without_a_patrol_only_the_visit_timings_are_offered() -> None:
    tuning = Tuning(controller(patrol=False))
    assert set(tuning.status()["values"]) == {"gone_after", "max_visit", "confirm_sightings"}
    with pytest.raises(ValueError):
        tuning.set({"dwell": 30})


def call(url: str, body: object = None) -> Tuple[int, dict]:
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_api_reads_changes_and_resets_the_timings() -> None:
    running = controller()
    idle = Idle()
    api = ControlApi(running, idle, "127.0.0.1", 0, tuning=Tuning(running))
    api.start()
    try:
        url = f"http://127.0.0.1:{api.port}/tuning"
        assert call(url)[1]["limits"]["gone_after"] == [10.0, 600.0]
        status, body = call(url, {"max_visit": 900})
        assert status == 200 and body["values"]["max_visit"] == 900.0
        status, body = call(url, {"max_visit": 5})
        assert status == 400 and "60 to 3600" in body["error"]
        status, body = call(url, {"default": True})
        assert body["values"]["max_visit"] == 600 and body["changed"] is False
    finally:
        api.stop()


def test_an_agent_without_tuning_has_no_such_address() -> None:
    running = controller()
    api = ControlApi(running, Idle(), "127.0.0.1", 0)
    api.start()
    try:
        assert call(f"http://127.0.0.1:{api.port}/tuning")[0] == 404
    finally:
        api.stop()


def test_changing_no_timing_leaves_them_as_the_flags_have_them(tmp_path: Path) -> None:
    tuning = Tuning(controller(), tmp_path / "tuning.json")
    tuning.set({})
    assert not tuning.changed and not (tmp_path / "tuning.json").exists()


def test_only_the_timings_that_were_set_are_kept_so_the_rest_follow_the_flags(
    tmp_path: Path,
) -> None:
    state = tmp_path / "tuning.json"
    Tuning(controller(), state).set({"dwell": 40})
    assert json.loads(state.read_text()) == {"dwell": 40.0}
    restarted = controller()
    restarted.presence.gone_after_seconds = 120
    tuning = Tuning(restarted, state)
    assert tuning.status()["values"]["gone_after"] == 120
    assert tuning.status()["values"]["dwell"] == 40.0
    assert tuning.changed is True

# ABOUTME: Tests for the record of where the camera points: the hourly log of idle looking, the
# ABOUTME: track kept inside each visit, and plans set while the agent runs.

import json
import urllib.request
from pathlib import Path

import pytest

from petcam.api import ControlApi
from petcam.controller import Controller
from petcam.detections import Box, Frame
from petcam.gaze import GazeLog, hour_of
from petcam.patrol import Patrol, Plan, Plans, Pose
from petcam.visits import Visit

HOME = Pose(yaw=225, pitch=120)
STOPS = [Pose(yaw=345, pitch=85), Pose(yaw=105, pitch=120)]
NOON = 1794744000.0  # 2026-11-15 12:00 UTC


class FakeGimbal:
    def move(self, *args: object) -> None:
        pass

    def move_to(self, *args: object) -> None:
        pass


class FakeRecorder:
    def set_recording(self, enabled: bool) -> None:
        pass


def patrol() -> Patrol:
    return Patrol(
        home=HOME, stops=STOPS, home_after_seconds=90, patrol_after_seconds=600, dwell_seconds=25
    )


def controller(gaze: GazeLog = None) -> Controller:
    return Controller(
        gimbal=FakeGimbal(),
        recorder=FakeRecorder(),
        gone_after_seconds=60,
        move_interval_seconds=0.3,
        patrol=patrol(),
        gaze=gaze,
    )


def pet() -> Frame:
    return Frame(
        boxes=[Box(cx=320, cy=320, w=80, h=80, score=0.8, class_id=15)],
        resolution=(640, 640),
        timestamp_ms=0,
    )


def test_log_adds_up_seconds_by_rounded_direction_and_hour() -> None:
    log = GazeLog()
    log.add(Pose(224.2, 119.0), 0.2, NOON)
    log.add(Pose(226.9, 121.4), 0.3, NOON + 1)
    log.add(Pose(105, 85), 1.0, NOON + 3600)
    hours = log.since()["hours"]
    assert hours[hour_of(NOON)] == {"225,120": 0.5}
    assert hours[hour_of(NOON + 3600)] == {"105,85": 1.0}
    assert list(log.since(hour_of(NOON + 3600))["hours"]) == [hour_of(NOON + 3600)]


def test_log_survives_a_restart(tmp_path: Path) -> None:
    log = GazeLog(tmp_path / "gaze.json")
    log.add(HOME, 2.0, NOON)
    log.save()
    assert GazeLog(tmp_path / "gaze.json").since()["hours"] == {hour_of(NOON): {"225,120": 2.0}}


def test_unreadable_log_starts_again(tmp_path: Path) -> None:
    (tmp_path / "gaze.json").write_text("[1, 2")
    assert GazeLog(tmp_path / "gaze.json").since()["hours"] == {}


def test_idle_looking_goes_to_the_log_and_not_into_a_visit() -> None:
    log = GazeLog()
    c = controller(log)
    c.pointing(HOME, NOON)
    c.pointing(HOME, NOON + 0.2)
    c.pointing(HOME, NOON + 0.4)
    assert log.since()["hours"][hour_of(NOON)] == {"225,120": pytest.approx(0.4, abs=0.05)}
    assert c.status()["pose"] == {"yaw": 225, "pitch": 120}


def test_a_gap_in_the_motors_answers_is_not_counted_as_looking() -> None:
    log = GazeLog()
    c = controller(log)
    c.pointing(HOME, NOON)
    c.pointing(None, NOON + 0.2)
    c.pointing(HOME, NOON + 30)
    c.pointing(HOME, NOON + 100)
    assert log.since()["hours"] == {}
    assert c.status()["pose"] == {"yaw": 225, "pitch": 120}


def test_visit_records_where_the_camera_pointed_and_the_log_pauses() -> None:
    log = GazeLog()
    c = controller(log)
    c.pointing(HOME, NOON - 0.2)
    c.observe(pet(), NOON)
    c.pointing(Pose(225, 120), NOON + 0.2)
    c.pointing(Pose(225.3, 120), NOON + 0.4)  # has not turned a degree
    c.pointing(Pose(231, 118), NOON + 0.6)
    c.pointing(Pose(231, 118), NOON + 11)  # still, but the last note is old
    c.tick(NOON + 100)
    visit = c.visits.finished()[0]
    assert visit.gaze == [
        (NOON - 0.2, 225, 120),
        (NOON + 0.6, 231, 118),
        (NOON + 11, 231, 118),
    ]
    assert log.since()["hours"] == {}
    assert Visit.from_dict(json.loads(json.dumps(visit.as_dict()))).gaze == visit.gaze


def test_visit_saved_before_gaze_was_recorded_still_loads() -> None:
    data = Visit(id="v", started_at=1.0, ended_at=2.0).as_dict()
    del data["gaze"]
    assert Visit.from_dict(data).gaze == []


def test_plan_refuses_what_the_camera_should_not_follow() -> None:
    good = {"home": {"yaw": 100, "pitch": 90}, "stops": [], "patrol_after": 300}
    assert Plan.from_dict(good) == Plan(Pose(100, 90), (), 300)
    for bad in (
        {**good, "home": {"yaw": 400, "pitch": 90}},
        {**good, "home": {"yaw": float("nan"), "pitch": 90}},
        {**good, "home": None},
        {**good, "stops": [{"yaw": 10}]},
        {**good, "stops": [{"yaw": 10, "pitch": 90}] * 25},
        {**good, "patrol_after": 5},
        {**good, "patrol_after": "soon"},
        [good],
    ):
        with pytest.raises(ValueError):
            Plan.from_dict(bad)


def test_set_plan_is_followed_and_kept_across_a_restart(tmp_path: Path) -> None:
    p = patrol()
    plans = Plans(p, tmp_path / "plan.json")
    assert not plans.learned
    # Long idle and not known to be anywhere: the first move is to the default home.
    assert p.next_pose(now=1000.0, last_activity_at=0.0) == HOME
    new = Plan(Pose(60, 100), (Pose(30, 100),), 120)
    plans.set(new)
    assert p.next_pose(now=1001.0, last_activity_at=0.0) == Pose(60, 100)
    assert p.next_pose(now=1100.0, last_activity_at=0.0) is None
    assert p.next_pose(now=1125.0, last_activity_at=0.0) == Pose(30, 100)

    again = Plans(patrol(), tmp_path / "plan.json")
    assert again.learned
    assert again.patrol.plan() == new
    assert again.default == Plan(HOME, tuple(STOPS), 600)

    again.reset()
    assert again.patrol.plan() == again.default
    assert not Plans(patrol(), tmp_path / "plan.json").learned


def test_unusable_saved_plan_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "plan.json").write_text('{"home": {"yaw": 999, "pitch": 0}}')
    plans = Plans(patrol(), tmp_path / "plan.json")
    assert not plans.learned
    assert plans.patrol.plan().home == HOME


def test_api_serves_gaze_and_takes_plans(tmp_path: Path) -> None:
    log = GazeLog()
    c = controller(log)
    c.pointing(HOME, NOON)
    c.pointing(HOME, NOON + 1)
    api = ControlApi(
        controller=c,
        gimbal=FakeGimbal(),
        host="127.0.0.1",
        port=0,
        plans=Plans(c.patrol, tmp_path / "plan.json"),
    )
    api.start()
    base = f"http://127.0.0.1:{api.port}"

    def call(path: str, body: dict = None) -> dict:
        request = urllib.request.Request(
            base + path,
            data=None if body is None else json.dumps(body).encode(),
            method="GET" if body is None else "POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    try:
        assert call("/gaze")["hours"] == {hour_of(NOON): {"225,120": 1.0}}
        assert call("/gaze?since=2999010100")["hours"] == {}
        assert call("/patrol")["learned"] is False
        plan = {"home": {"yaw": 60.0, "pitch": 100.0}, "stops": [], "patrol_after": 300.0}
        reply = call("/patrol", plan)
        assert reply["learned"] is True and reply["plan"] == plan
        with pytest.raises(urllib.error.HTTPError) as refused:
            call("/patrol", {"home": {"yaw": 999, "pitch": 0}, "stops": [], "patrol_after": 300})
        assert refused.value.code == 400
        assert call("/patrol")["plan"] == plan
        assert call("/patrol", {"default": True})["plan"]["home"] == {"yaw": 225, "pitch": 120}
        # A pose the gimbal stopped short of is reported until the agent is told to forget it.
        assert call("/patrol")["missed"] == []
        c.patrol.missed[Pose(300, 100)] = Pose(280.5, 100)
        assert call("/patrol")["missed"] == [
            {"asked": {"yaw": 300, "pitch": 100}, "reached": {"yaw": 280.5, "pitch": 100}}
        ]
        reply = call("/patrol", {"forget_missed": True})
        assert reply["missed"] == [] and reply["learned"] is False
    finally:
        api.stop()

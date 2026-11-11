# ABOUTME: Tests for keeping the camera's aim up to date: gathering its record of where it looked,
# ABOUTME: writing the map for the page, and sending plans only when the settings ask for them.

import json
from pathlib import Path
from typing import Dict, List, Optional

from collector.aiming import REDRAW_EVERY_SECONDS, REPLAN_EVERY_SECONDS, Aimer, Settings
from collector.camera_client import CameraError, CameraUnreachable
from collector.selector import ModelSelector

NOW = 1794744000.0  # 2026-11-15 12:00 UTC
DEFAULT = {
    "home": {"yaw": 225.0, "pitch": 120.0},
    "stops": [{"yaw": 105.0, "pitch": 120.0}],
    "patrol_after": 600.0,
}


class FakeCamera:
    def __init__(self) -> None:
        self.plan: Optional[dict] = None
        self.sent: List[Optional[dict]] = []
        self.hours: Dict[str, Dict[str, float]] = {"2026111511": {"225,120": 3000.0}}
        self.asked_since: List[Optional[str]] = []
        self.too_old = False
        self.missed: List[dict] = []

    def patrol(self) -> dict:
        if self.too_old:
            raise CameraError("GET /patrol: HTTP 404 b'{}'")
        return {
            "plan": self.plan or DEFAULT,
            "default": DEFAULT,
            "learned": self.plan is not None,
            "missed": self.missed,
        }

    def set_patrol(self, plan: Optional[dict]) -> dict:
        self.sent.append(plan)
        self.plan = plan
        return self.patrol()

    def gaze(self, since: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        self.asked_since.append(since)
        return {hour: looked for hour, looked in self.hours.items() if not since or hour >= since}


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


def make(tmp_path: Path):
    camera, clock = FakeCamera(), Clock()
    archive = tmp_path / "visits"
    archive.mkdir()
    aimer = Aimer(camera, archive, tmp_path / "views", (640, 640), clock=clock)
    return aimer, camera, clock, tmp_path / "views"


def settings(views: Path, **values: object) -> None:
    views.mkdir(exist_ok=True)
    (views / "settings.json").write_text(json.dumps(values))


def add_visit(archive: Path, name: str, yaw: float, score: float) -> None:
    (archive / name).mkdir()
    record = {
        "visit": {
            "started_at": NOW - 3600,
            "clock_offset": 0.0,
            "gaze": [[NOW - 3600, yaw, 100.0]],
            "sightings": [{"at": NOW - 3600, "cx": 320, "cy": 320}],
        },
        "result": {
            "best_frame": f"/archive/{name}/frames/f.jpg",
            "best_frame_at": NOW - 3600,
            "posted": True,
            "candidates": [{"model_score": score}],
        },
    }
    (archive / name / "visit.json").write_text(json.dumps(record))


def test_settings_default_to_the_cameras_own_plan_and_survive_nonsense(tmp_path: Path) -> None:
    assert Settings.read(tmp_path / "none.json").mode == "fixed"
    (tmp_path / "s.json").write_text('{"mode": "learned", "explore": 7, "since": 5, "stops": 99}')
    read = Settings.read(tmp_path / "s.json")
    assert (read.mode, read.explore, read.since, read.stops) == ("learned", 1.0, 5.0, 16)
    (tmp_path / "s.json").write_text('{"explore": "lots"}')
    assert Settings.read(tmp_path / "s.json") == Settings()


def test_fixed_mode_draws_the_map_and_leaves_the_camera_alone(tmp_path: Path) -> None:
    aimer, camera, _, views = make(tmp_path)
    add_visit(tmp_path / "visits", "20261115-110000-00", 60.0, 0.9)
    add_visit(tmp_path / "visits", "old-visit-without-gaze", 60.0, 0.9)
    record = tmp_path / "visits" / "old-visit-without-gaze" / "visit.json"
    old = json.loads(record.read_text())
    del old["visit"]["gaze"]
    record.write_text(json.dumps(old))

    assert aimer.refresh()
    assert camera.sent == []
    drawn = json.loads((views / "map.json").read_text())
    assert drawn["visits"] == 1
    assert drawn["watched_hours"] == 0.83
    assert drawn["camera"]["learned"] is False
    assert drawn["proposed"]["home"] == DEFAULT["home"]
    assert drawn["photos"][0]["visit"] == "20261115-110000-00"
    assert drawn["photos"][0]["yaw"] == 60.0
    assert any(cell["found"] > 0 for cell in drawn["cells"])
    assert drawn["unwatched"]["doubt"] > 0
    assert json.loads((views / "gaze.json").read_text()) == camera.hours


def test_map_is_redrawn_on_a_timer_or_when_the_settings_change(tmp_path: Path) -> None:
    aimer, camera, clock, views = make(tmp_path)
    assert aimer.refresh()
    clock.now += 60
    assert not aimer.refresh()
    settings(views, mode="fixed", explore=0.9)
    assert aimer.refresh()
    clock.now += REDRAW_EVERY_SECONDS
    camera.hours["2026111512"] = {"225,120": 500.0}
    assert aimer.refresh()
    # Only the hours from the last one held are asked for again.
    assert camera.asked_since[-1] == "2026111511"
    assert set(json.loads((views / "gaze.json").read_text())) == {"2026111511", "2026111512"}


def test_learned_mode_sends_a_plan_and_then_holds_it_for_hours(tmp_path: Path) -> None:
    aimer, camera, clock, views = make(tmp_path)
    settings(views, mode="learned", explore=0.5)
    aimer.refresh()
    assert len(camera.sent) == 1
    first = camera.sent[0]
    assert first["home"] == DEFAULT["home"] and len(first["stops"]) == 8

    # More looking changes what is worth exploring, but the plan is kept for now.
    camera.hours["2026111511"] = {"225,120": 3000.0, "15,80": 3000.0}
    clock.now += REDRAW_EVERY_SECONDS
    aimer.refresh()
    assert len(camera.sent) == 1
    assert json.loads((views / "map.json").read_text())["proposed"] != first

    clock.now += REPLAN_EVERY_SECONDS
    aimer.refresh()
    assert len(camera.sent) == 2 and camera.sent[1] != first

    # Changing the slider replans at once.
    settings(views, mode="learned", explore=0.0)
    aimer.refresh()
    assert len(camera.sent) == 3
    assert camera.sent[2]["patrol_after"] == 1200


def test_switching_back_to_fixed_returns_the_camera_to_its_own_plan(tmp_path: Path) -> None:
    aimer, camera, _, views = make(tmp_path)
    settings(views, mode="learned")
    aimer.refresh()
    settings(views, mode="fixed")
    aimer.refresh()
    assert camera.sent[-1] is None and camera.plan is None
    assert json.loads((views / "map.json").read_text())["camera"]["learned"] is False


def test_an_agent_too_old_to_take_plans_is_reported_not_fatal(tmp_path: Path) -> None:
    aimer, camera, _, views = make(tmp_path)
    camera.too_old = True
    settings(views, mode="learned")
    assert aimer.refresh()
    drawn = json.loads((views / "map.json").read_text())
    assert drawn["camera"] is None and drawn["cells"] == []
    assert camera.sent == []


def test_current_pose_is_noted_for_the_page(tmp_path: Path) -> None:
    aimer, _, _, views = make(tmp_path)
    aimer.saw({"in_visit": False})  # an agent that does not report it
    assert not (views / "pose.json").exists()
    aimer.saw({"pose": {"yaw": 225.0, "pitch": 120.0}, "in_visit": True, "patrol": {}})
    noted = json.loads((views / "pose.json").read_text())
    assert noted["pose"] == {"yaw": 225.0, "pitch": 120.0} and noted["in_visit"] is True


def test_a_plan_the_camera_did_not_receive_is_sent_again_at_the_next_look(tmp_path: Path) -> None:
    aimer, camera, clock, views = make(tmp_path)
    settings(views, mode="learned", explore=0.5)
    answer = camera.set_patrol

    def out_of_reach(plan: Optional[dict]) -> dict:
        raise CameraUnreachable("POST /patrol: timed out")

    camera.set_patrol = out_of_reach
    try:
        aimer.refresh()
    except CameraUnreachable:
        pass
    camera.set_patrol = answer
    clock.now += 15
    aimer.refresh()
    assert len(camera.sent) == 1


def test_the_views_alone_do_not_make_the_selector_look_in_use(tmp_path: Path) -> None:
    # The collector keeps the camera's pose under the selector's directory; that must not
    # count as someone having started labelling.
    aimer, _, _, _ = make(tmp_path)
    aimer = Aimer(aimer.client, aimer.archive_dir, tmp_path / "selector" / "views", (640, 640))
    aimer.saw({"pose": {"yaw": 1.0, "pitch": 2.0}})
    assert (tmp_path / "selector" / "views" / "pose.json").is_file()
    assert not ModelSelector(tmp_path / "selector").in_use()


def test_a_plan_set_by_hand_is_left_on_the_camera(tmp_path: Path) -> None:
    aimer, camera, clock, views = make(tmp_path)
    by_hand = {"home": {"yaw": 90.0, "pitch": 100.0}, "stops": [], "patrol_after": 600.0}
    camera.plan = by_hand
    settings(views, mode="manual")
    add_visit(tmp_path / "visits", "20261115-110000-00", 60.0, 0.9)
    assert aimer.refresh()
    clock.now += REDRAW_EVERY_SECONDS
    assert aimer.refresh()
    assert camera.sent == []
    drawn = json.loads((views / "map.json").read_text())
    assert drawn["camera"]["plan"] == by_hand
    assert drawn["settings"]["mode"] == "manual"


def test_a_stop_the_gimbal_could_not_reach_narrows_the_plan_and_the_map_at_once(
    tmp_path: Path,
) -> None:
    aimer, camera, clock, views = make(tmp_path)
    settings(views, mode="learned", explore=1.0)
    aimer.refresh()
    drawing = json.loads((views / "map.json").read_text())
    assert not drawing["stopped_short"] and [45.0, 105.0] in drawing["seeable"]
    assert any(stop["yaw"] < 100 for stop in camera.sent[-1]["stops"])

    camera.missed = [
        {"asked": {"yaw": 0.0, "pitch": 100.0}, "reached": {"yaw": 100.0, "pitch": 100.0}},
        {"asked": "nonsense"},
    ]
    # The map's next redrawing sends a new plan, though the last is not yet hours old.
    clock.now += 601
    aimer.refresh()
    assert len(camera.sent) == 2
    assert all(stop["yaw"] >= 100 for stop in camera.sent[-1]["stops"])
    drawing = json.loads((views / "map.json").read_text())
    assert drawing["stopped_short"] and drawing["limits"]["yaw_low"] == 100.0
    assert [45.0, 105.0] not in drawing["seeable"] and [205.0, 105.0] in drawing["seeable"]

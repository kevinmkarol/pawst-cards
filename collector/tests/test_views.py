# ABOUTME: Tests for the map of the camera's surroundings and the plan made from it, including a
# ABOUTME: simulated room in which the camera has to find where the pet turns up.

import time

import numpy as np
import pytest

from collector.views import (
    Limits,
    ViewMap,
    VisitView,
    build_map,
    cell_of,
    describe,
    direction,
    in_view,
    make_plan,
    patrol_after,
    pose_at,
    view_of,
    worth_of,
)

RESOLUTION = (640, 640)
NOW = 1794744000.0  # 2026-11-15 12:00 UTC
HOME = (225.0, 120.0)


def hour(at: float) -> str:
    return time.strftime("%Y%m%d%H", time.gmtime(at))


def visit(found, worth=0.8, at=NOW - 3600, photo_at=None, key="v") -> VisitView:
    return VisitView(
        key=key,
        at=at,
        worth=worth,
        found=found,
        photo_at=photo_at or found,
        photo=f"/archive/{key}/frames/best.jpg",
        score=0.5 + worth / 2,
    )


def test_pose_at_reads_between_the_notes_and_holds_at_the_ends() -> None:
    gaze = [[10.0, 200.0, 100.0], [20.0, 220.0, 110.0]]
    assert pose_at(gaze, 5.0) == (200.0, 100.0)
    assert pose_at(gaze, 15.0) == (210.0, 105.0)
    assert pose_at(gaze, 99.0) == (220.0, 110.0)
    assert pose_at([], 15.0) is None


def test_direction_of_a_pet_follows_the_trackers_signs() -> None:
    assert direction((225, 120), 320, 320, RESOLUTION) == (225, 120)
    yaw, pitch = direction((225, 120), 640, 640, RESOLUTION)
    assert yaw == pytest.approx(255) and pitch == pytest.approx(140)
    yaw, pitch = direction((350, 120), 640, 0, RESOLUTION)
    assert yaw == pytest.approx(20) and pitch == pytest.approx(100)


def test_worth_is_how_far_the_best_photo_was_above_the_posting_bar() -> None:
    scored = {"candidates": [{"model_score": 0.2}, {"model_score": 0.9}], "posted": False}
    assert worth_of(scored) == (pytest.approx(0.8), 0.9)
    assert worth_of({"candidates": [{"model_score": 0.3}], "posted": False}) == (0.0, 0.3)
    assert worth_of({"candidates": [{"model_score": None}], "posted": True}) == (0.5, None)
    assert worth_of({"posted": False}) == (0.0, None)


def record(gaze, best_at=105.0) -> dict:
    return {
        "visit": {
            "started_at": 100.0,
            "clock_offset": 50.0,
            "gaze": gaze,
            "sightings": [
                {"at": 100.0, "cx": 320, "cy": 320},
                {"at": 101.0, "cx": 320, "cy": 320},
                {"at": 105.2, "cx": 640, "cy": 320},
            ],
        },
        "result": {
            "best_frame": "/a/v/frames/f.jpg",
            "best_frame_at": best_at,
            "chosen_by": "model",
            "posted": True,
            "candidates": [{"model_score": 0.75}],
        },
    }


def test_visit_view_places_the_pet_where_it_was_found_and_photographed() -> None:
    view = view_of("v", record([[100.0, 225.0, 120.0], [105.0, 100.0, 90.0]]), RESOLUTION)
    assert view.at == 150.0
    assert view.worth == pytest.approx(0.5)
    assert view.found == (pytest.approx(200.0), pytest.approx(114.0))
    assert view.photo_at == (pytest.approx(130.0), pytest.approx(90.0))


def test_photo_with_no_sighting_near_it_is_placed_where_the_camera_pointed() -> None:
    view = view_of("v", record([[100.0, 225.0, 120.0]], best_at=130.0), RESOLUTION)
    assert view.photo_at == (225.0, 120.0)


def test_visit_without_a_record_of_where_the_camera_pointed_is_left_out() -> None:
    assert view_of("v", record([]), RESOLUTION) is None


def test_view_covers_the_field_of_view_and_wraps_around() -> None:
    seen = in_view((225, 120))
    assert seen[cell_of((225, 120))] and seen[cell_of((245, 135))]
    assert not seen[cell_of((260, 120))] and not seen[cell_of((225, 145))]
    assert in_view((350, 90))[cell_of((15, 90))]


def test_map_counts_watching_finding_and_photos_with_recent_days_weighing_more() -> None:
    gaze = {
        hour(NOW - 3600): {"225,120": 3600.0},
        hour(NOW - 7 * 86400): {"60,100": 3600.0},
        "garbage": {"0,0": 9.0},
    }
    recent = visit((230, 118), worth=1.0, photo_at=(100, 95), key="recent")
    old = visit((60, 100), worth=1.0, at=NOW - 7 * 86400, key="old")
    view_map = build_map([recent, old], gaze, NOW)
    assert view_map.visits == 2
    assert view_map.watched[cell_of((225, 120))] == pytest.approx(1.0, abs=0.01)
    assert view_map.watched[cell_of((60, 100))] == pytest.approx(0.5, abs=0.01)
    assert view_map.watched[cell_of((140, 100))] == 0
    assert view_map.found.sum() == pytest.approx(1.5, abs=0.01)
    assert view_map.found[cell_of((230, 118))] == pytest.approx(1.0, abs=0.01)
    assert view_map.photos[cell_of((100, 95))] == pytest.approx(1.0, abs=0.01)
    assert view_map.best[cell_of((100, 95))].key == "recent"

    cells = {(c["yaw"], c["pitch"]): c for c in describe(view_map)}
    assert cells[(105.0, 95.0)]["best"] == {
        "visit": "recent",
        "frame": "best.jpg",
        "score": 1.0,
        "at": NOW - 3600,
    }


def test_map_forgets_what_came_before_the_camera_was_moved() -> None:
    gaze = {hour(NOW - 86400): {"225,120": 3600.0}, hour(NOW - 3600): {"60,100": 3600.0}}
    view_map = build_map(
        [visit((225, 120), at=NOW - 86400), visit((60, 100))], gaze, NOW, since=NOW - 7200
    )
    assert view_map.visits == 1
    assert view_map.watched[cell_of((225, 120))] == 0
    assert view_map.found[cell_of((225, 120))] == 0
    assert view_map.found[cell_of((60, 100))] > 0


def test_empty_map_keeps_the_home_and_spreads_the_stops() -> None:
    plan = make_plan(ViewMap(), explore=0.5, home_now=HOME, stops=8)
    assert plan["home"] == {"yaw": 225.0, "pitch": 120.0}
    assert len(plan["stops"]) == 8
    yaws = [stop["yaw"] for stop in plan["stops"]]
    assert yaws == sorted(yaws)
    seen = in_view(HOME)
    for stop in plan["stops"]:
        view = in_view((stop["yaw"], stop["pitch"]))
        assert (view & ~seen).sum() >= view.sum() / 2
        seen |= view


def test_a_few_visits_do_not_move_the_home() -> None:
    gaze = {hour(NOW - 3600): {"60,100": 600.0, "225,120": 3000.0}}
    view_map = build_map([visit((60, 100)), visit((60, 100))], gaze, NOW)
    assert make_plan(view_map, 0.3, HOME, 8)["home"] == {"yaw": 225.0, "pitch": 120.0}


def test_stops_respect_the_limits_and_patrols_come_sooner_with_more_exploration() -> None:
    limits = Limits(yaw_low=90, yaw_high=180, pitch_low=90, pitch_high=110)
    plan = make_plan(ViewMap(), 1.0, HOME, 8, limits)
    assert plan["stops"]
    for stop in plan["stops"]:
        assert 90 <= stop["yaw"] <= 180 and 90 <= stop["pitch"] <= 110
    assert patrol_after(0.0) == 1200 and patrol_after(1.0) == 120
    assert patrol_after(0.3) == pytest.approx(600, abs=10)
    assert make_plan(ViewMap(), 0.0, HOME, 0)["stops"] == []


def room(explore: float, days: float, seed: int = 0):
    """A camera resting at HOME in a room where the pet turns up around (65, 105), far outside
    the home's view, twice an hour when the camera is looking there, and a quarter as often
    in the home's view with poorer photos. Returns the plans made every three hours."""
    rng = np.random.default_rng(seed)
    spots = [((65.0, 105.0), 2.0, 0.8), (HOME, 0.5, 0.2)]
    gaze, visits, plans = {}, [], []
    plan = {
        "home": {"yaw": HOME[0], "pitch": HOME[1]},
        "stops": [{"yaw": 345.0, "pitch": 85.0}, {"yaw": 255.0, "pitch": 120.0}],
        "patrol_after": 600,
    }
    start = NOW - days * 86400
    for step in range(int(days * 24)):
        at = start + step * 3600
        if step % 3 == 0 and step:
            home = (plan["home"]["yaw"], plan["home"]["pitch"])
            plan = make_plan(build_map(visits, gaze, at), explore, home, 8)
            plans.append(plan)
        home = (plan["home"]["yaw"], plan["home"]["pitch"])
        stops = [(s["yaw"], s["pitch"]) for s in plan["stops"]]
        sweeps = 3600 / (plan["patrol_after"] + 25 * len(stops))
        looked = {home: 3600 - sweeps * 25 * len(stops)}
        for stop in stops:
            looked[stop] = looked.get(stop, 0) + sweeps * 25
        gaze[hour(at)] = {f"{yaw:g},{pitch:g}": seconds for (yaw, pitch), seconds in looked.items()}
        for spot, per_hour, worth in spots:
            in_sight = sum(s for pose, s in looked.items() if in_view(pose)[cell_of(spot)])
            for _ in range(rng.poisson(per_hour * in_sight / 3600)):
                where = (spot[0] + rng.normal(0, 4), spot[1] + rng.normal(0, 3))
                visits.append(visit(where, worth=worth, at=at + 1800, key=f"v{len(visits)}"))
    return plans


def near(pose, spot) -> bool:
    return abs(pose[0] - spot[0]) <= 15 and abs(pose[1] - spot[1]) <= 10


@pytest.mark.parametrize("explore", [0.3, 0.7])
def test_camera_finds_the_better_spot_and_makes_it_home(explore: float) -> None:
    plans = room(explore, days=6)
    home = (plans[-1]["home"]["yaw"], plans[-1]["home"]["pitch"])
    assert near(home, (65, 105))
    # And having moved, it stays: the last day's plans all rest pointing at the spot.
    for plan in plans[-8:]:
        assert near((plan["home"]["yaw"], plan["home"]["pitch"]), (65, 105))


def test_limits_are_drawn_in_to_where_the_gimbal_stopped_short() -> None:
    limits = Limits(yaw_low=0, yaw_high=345, pitch_low=70, pitch_high=140)
    # Sent to yaw 300 it stopped at 281.6; sent to pitch 70 it stopped at 84.2.
    reach = limits.within_reach([((300.0, 100.0), (281.6, 100.0)), ((150.0, 70.0), (151.0, 84.2))])
    assert reach == Limits(yaw_low=0, yaw_high=281, pitch_low=85, pitch_high=140)
    assert limits.within_reach([]) == limits
    # A miss that would leave nothing to choose from is not believed.
    assert limits.within_reach([((300.0, 100.0), (-20.0, 100.0))]) == limits
    plan = make_plan(ViewMap(), 1.0, HOME, 8, reach)
    assert plan["stops"] and all(s["yaw"] <= 281 and s["pitch"] >= 85 for s in plan["stops"])


def test_only_what_a_pose_within_the_limits_sees_is_seeable() -> None:
    seeable = Limits(yaw_low=135, yaw_high=345, pitch_low=70, pitch_high=140).seeable()
    assert seeable[cell_of((200, 100))] and seeable[cell_of((5, 100))]
    assert not seeable[cell_of((60, 100))] and not seeable[cell_of((200, 45))]
    assert int(seeable.sum()) == 26 * 11


def test_a_home_outside_the_limits_is_brought_within_them() -> None:
    limits = Limits(yaw_low=90, yaw_high=180, pitch_low=90, pitch_high=110)
    assert make_plan(ViewMap(), 0.3, HOME, 0, limits)["home"] == {"yaw": 180.0, "pitch": 110.0}

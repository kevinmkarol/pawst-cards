# ABOUTME: Tests for the idle behaviour: return to the home pose after a quiet spell and sweep the
# ABOUTME: patrol stops after a longer one. Time and activity are passed in explicitly.

from petcam.patrol import Patrol, Pose

HOME = Pose(yaw=225, pitch=120)
STOPS = [Pose(yaw=345, pitch=85), Pose(yaw=15, pitch=85), Pose(yaw=105, pitch=120)]


def make(stops=STOPS) -> Patrol:
    return Patrol(
        home=HOME,
        stops=stops,
        home_after_seconds=90,
        patrol_after_seconds=600,
        dwell_seconds=25,
    )


def test_nothing_to_do_while_there_was_recent_activity() -> None:
    patrol = make()
    assert patrol.next_pose(now=100.0, last_activity_at=50.0) is None


def test_goes_home_once_after_the_quiet_spell() -> None:
    patrol = make()
    assert patrol.next_pose(now=189.0, last_activity_at=100.0) is None
    assert patrol.next_pose(now=190.0, last_activity_at=100.0) == HOME
    assert patrol.next_pose(now=191.0, last_activity_at=100.0) is None


def test_goes_home_again_after_being_displaced() -> None:
    patrol = make()
    assert patrol.next_pose(now=190.0, last_activity_at=100.0) == HOME
    patrol.displaced()
    assert patrol.next_pose(now=191.0, last_activity_at=100.0) == HOME


def test_sweeps_the_stops_after_a_long_idle_and_returns_home() -> None:
    patrol = make()
    assert patrol.next_pose(now=190.0, last_activity_at=100.0) == HOME
    assert patrol.next_pose(now=789.0, last_activity_at=100.0) is None
    assert patrol.next_pose(now=790.0, last_activity_at=100.0) == STOPS[0]
    assert patrol.next_pose(now=814.0, last_activity_at=100.0) is None
    assert patrol.next_pose(now=815.0, last_activity_at=100.0) == STOPS[1]
    assert patrol.next_pose(now=840.0, last_activity_at=100.0) == STOPS[2]
    assert patrol.next_pose(now=865.0, last_activity_at=100.0) == HOME
    assert patrol.next_pose(now=866.0, last_activity_at=100.0) is None
    assert patrol.next_pose(now=1464.0, last_activity_at=100.0) is None
    assert patrol.next_pose(now=1465.0, last_activity_at=100.0) == STOPS[0]


def test_activity_during_a_sweep_abandons_it_and_home_follows_the_quiet_spell() -> None:
    patrol = make()
    patrol.next_pose(now=190.0, last_activity_at=100.0)
    assert patrol.next_pose(now=790.0, last_activity_at=100.0) == STOPS[0]
    assert patrol.next_pose(now=800.0, last_activity_at=795.0) is None
    patrol.displaced()
    assert patrol.next_pose(now=884.0, last_activity_at=795.0) is None
    assert patrol.next_pose(now=885.0, last_activity_at=795.0) == HOME
    assert patrol.next_pose(now=1484.0, last_activity_at=795.0) is None
    assert patrol.next_pose(now=1485.0, last_activity_at=795.0) == STOPS[0]


def test_without_stops_it_only_goes_home() -> None:
    patrol = make(stops=[])
    assert patrol.next_pose(now=190.0, last_activity_at=100.0) == HOME
    assert patrol.next_pose(now=5000.0, last_activity_at=100.0) is None


def test_status_names_what_it_is_doing() -> None:
    patrol = make()
    assert patrol.status() == {"at": None, "patrolling": False}
    patrol.next_pose(now=190.0, last_activity_at=100.0)
    assert patrol.status() == {"at": {"yaw": 225, "pitch": 120}, "patrolling": False}
    patrol.next_pose(now=790.0, last_activity_at=100.0)
    assert patrol.status() == {"at": {"yaw": 345, "pitch": 85}, "patrolling": True}


def test_a_stop_the_gimbal_did_not_reach_is_noted_and_forgiven_once_it_is_reached() -> None:
    patrol = make()
    patrol.next_pose(now=190.0, last_activity_at=100.0)
    # Leaving home for the sweep, the gimbal is at home; leaving the first stop, it is not there.
    assert patrol.next_pose(now=790.0, last_activity_at=100.0, pointing=HOME) == STOPS[0]
    assert patrol.next_pose(now=800.0, last_activity_at=100.0, pointing=Pose(300, 85)) is None
    assert patrol.missed == {}
    patrol.next_pose(now=815.0, last_activity_at=100.0, pointing=Pose(300.04, 85))
    assert patrol.missed == {STOPS[0]: Pose(300.0, 85)}
    # A pose within a few degrees, also across the turn from 360 to 0, counts as reached.
    patrol.next_pose(now=840.0, last_activity_at=100.0, pointing=Pose(11, 88))
    patrol.next_pose(now=865.0, last_activity_at=100.0, pointing=None)
    assert patrol.missed == {STOPS[0]: Pose(300.0, 85)}
    assert Pose(358, 85).near(Pose(2, 85)) and not Pose(350, 85).near(Pose(2, 85))
    # It outlasts a new plan, and the next sweep that gets there clears it.
    patrol.follow(patrol.plan())
    assert patrol.missed
    patrol.next_pose(now=2000.0, last_activity_at=100.0)
    patrol.next_pose(now=2600.0, last_activity_at=100.0, pointing=HOME)
    patrol.next_pose(now=2625.0, last_activity_at=100.0, pointing=Pose(344, 86))
    assert patrol.missed == {}


def test_a_home_the_gimbal_did_not_reach_is_noted_when_the_sweep_sets_off() -> None:
    patrol = make()
    patrol.next_pose(now=190.0, last_activity_at=100.0)
    patrol.next_pose(now=790.0, last_activity_at=100.0, pointing=Pose(225, 140))
    assert patrol.missed == {HOME: Pose(225, 140)}

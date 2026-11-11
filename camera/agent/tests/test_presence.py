# ABOUTME: Tests for the visit state machine: when a pet counts as arrived and as gone.
# ABOUTME: Time is passed in explicitly so the tests are deterministic.

from petcam.presence import Presence, VisitEvent


def test_first_sighting_starts_a_visit() -> None:
    presence = Presence(gone_after_seconds=60)
    assert presence.update(pet_seen=True, now=100.0) == VisitEvent.STARTED
    assert presence.in_visit


def test_repeated_sightings_do_not_restart() -> None:
    presence = Presence(gone_after_seconds=60)
    presence.update(pet_seen=True, now=100.0)
    assert presence.update(pet_seen=True, now=101.0) is None
    assert presence.update(pet_seen=False, now=102.0) is None


def test_visit_ends_only_after_quiet_period() -> None:
    presence = Presence(gone_after_seconds=60)
    presence.update(pet_seen=True, now=100.0)
    assert presence.update(pet_seen=False, now=159.9) is None
    assert presence.in_visit
    assert presence.update(pet_seen=False, now=160.0) == VisitEvent.ENDED
    assert not presence.in_visit


def test_sighting_during_quiet_period_extends_visit() -> None:
    presence = Presence(gone_after_seconds=60)
    presence.update(pet_seen=True, now=100.0)
    presence.update(pet_seen=False, now=150.0)
    presence.update(pet_seen=True, now=155.0)
    assert presence.update(pet_seen=False, now=214.0) is None
    assert presence.update(pet_seen=False, now=215.0) == VisitEvent.ENDED


def test_no_events_while_nothing_seen() -> None:
    presence = Presence(gone_after_seconds=60)
    assert presence.update(pet_seen=False, now=1.0) is None
    assert presence.update(pet_seen=False, now=1000.0) is None


def test_visit_records_start_and_last_seen_times() -> None:
    presence = Presence(gone_after_seconds=60)
    presence.update(pet_seen=True, now=100.0)
    presence.update(pet_seen=True, now=130.0)
    assert presence.started_at == 100.0
    assert presence.last_seen_at == 130.0


def test_a_visit_needs_several_sightings_close_together() -> None:
    # A single stray detection (the ceiling lamp read as a pet) must not start a visit.
    presence = Presence(gone_after_seconds=60, confirm_sightings=3, confirm_window_seconds=2.0)
    assert presence.update(pet_seen=True, now=100.0) is None
    assert not presence.in_visit
    assert presence.update(pet_seen=True, now=100.2) is None
    assert presence.update(pet_seen=True, now=100.4) == VisitEvent.STARTED
    assert presence.started_at == 100.0


def test_sightings_too_far_apart_do_not_confirm() -> None:
    presence = Presence(gone_after_seconds=60, confirm_sightings=3, confirm_window_seconds=2.0)
    presence.update(pet_seen=True, now=100.0)
    presence.update(pet_seen=True, now=105.0)
    assert presence.update(pet_seen=True, now=110.0) is None
    assert not presence.in_visit
    presence.update(pet_seen=True, now=110.5)
    assert presence.update(pet_seen=True, now=111.0) == VisitEvent.STARTED
    assert presence.started_at == 110.0


def test_visit_is_capped_while_pet_stays() -> None:
    presence = Presence(gone_after_seconds=60, max_visit_seconds=600)
    assert presence.update(pet_seen=True, now=100.0) == VisitEvent.STARTED
    assert presence.update(pet_seen=True, now=699.0) is None
    assert presence.update(pet_seen=True, now=700.0) == VisitEvent.ENDED
    assert not presence.in_visit
    assert presence.settled


def test_settled_pet_does_not_start_a_visit_until_it_moves() -> None:
    presence = Presence(gone_after_seconds=60, max_visit_seconds=600)
    presence.update(pet_seen=True, now=100.0)
    presence.update(pet_seen=True, now=700.0)
    assert presence.update(pet_seen=True, now=701.0) is None
    assert presence.update(pet_seen=True, now=900.0) is None
    assert presence.update(pet_seen=True, now=901.0, moving=True) == VisitEvent.STARTED
    assert not presence.settled


def test_settled_clears_after_quiet_period() -> None:
    presence = Presence(gone_after_seconds=60, max_visit_seconds=600)
    presence.update(pet_seen=True, now=100.0)
    presence.update(pet_seen=True, now=700.0)
    assert presence.update(pet_seen=False, now=759.0) is None
    assert presence.settled
    assert presence.update(pet_seen=False, now=760.0) is None
    assert not presence.settled
    assert presence.update(pet_seen=True, now=800.0) == VisitEvent.STARTED


def test_without_a_cap_a_long_visit_continues() -> None:
    presence = Presence(gone_after_seconds=60)
    presence.update(pet_seen=True, now=100.0)
    assert presence.update(pet_seen=True, now=100000.0) is None
    assert presence.in_visit

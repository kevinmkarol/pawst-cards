# ABOUTME: Tests for the visit log: sightings per visit, finished visits, and which clip files
# ABOUTME: in the recording directory belong to a visit. Uses real files in a temp directory.

import os
from pathlib import Path

from petcam.detections import Box
from petcam.visits import MOST_KEPT_SIGHTINGS, VISIT_WEIGHT, VisitLog, recorded_clips


def pet() -> Box:
    return Box(cx=300, cy=310, w=90, h=80, score=0.77, class_id=15)


def touch(path: Path, mtime: float) -> None:
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))


def test_sightings_are_recorded_for_the_open_visit() -> None:
    log = VisitLog()
    log.start(now=100.0)
    log.sighting(pet(), now=101.5)
    log.sighting(pet(), now=102.0)
    visit = log.end(now=170.0)
    assert visit.started_at == 100.0
    assert visit.ended_at == 170.0
    assert [s.at for s in visit.sightings] == [101.5, 102.0]
    assert visit.sightings[0].box == pet()


def test_the_oldest_visits_are_dropped_when_none_are_collected_for_a_long_time() -> None:
    log = VisitLog()
    for number in range(MOST_KEPT_SIGHTINGS // VISIT_WEIGHT + 50):
        log.start(now=1000.0 * number)
        log.end(now=1000.0 * number + 60)
    kept = log.finished()
    assert len(kept) == MOST_KEPT_SIGHTINGS // VISIT_WEIGHT
    assert kept[-1].started_at == 1000.0 * (MOST_KEPT_SIGHTINGS // VISIT_WEIGHT + 49)


def test_finished_visits_are_listed_until_collected() -> None:
    log = VisitLog()
    log.start(now=100.0)
    log.end(now=170.0)
    log.start(now=300.0)
    log.end(now=400.0)
    assert [v.started_at for v in log.finished()] == [100.0, 300.0]
    log.collected(log.finished()[0].id)
    assert [v.started_at for v in log.finished()] == [300.0]


def test_visit_ids_are_unique_and_filename_safe() -> None:
    log = VisitLog()
    log.start(now=1700000000.25)
    first = log.end(now=1700000060.0)
    log.start(now=1700000100.0)
    second = log.end(now=1700000160.0)
    assert first.id != second.id
    assert all(ch.isalnum() or ch in "-_" for ch in first.id)


def test_clips_for_visit_are_files_modified_during_the_visit(tmp_path: Path) -> None:
    log = VisitLog()
    log.start(now=1000.0)
    visit = log.end(now=1060.0)
    touch(tmp_path / "before.mp4", 990.0)
    touch(tmp_path / "first.mp4", 1012.0)
    touch(tmp_path / "last.mp4", 1068.0)
    touch(tmp_path / "later.mp4", 1200.0)
    touch(tmp_path / "notes.txt", 1030.0)
    clips = log.clips_among(visit, recorded_clips(tmp_path), slice_seconds=10)
    assert [c.path.name for c in clips] == ["first.mp4", "last.mp4"]


def test_as_dict_is_json_friendly() -> None:
    log = VisitLog()
    log.start(now=100.0)
    log.sighting(pet(), now=101.0)
    visit = log.end(now=170.0)
    data = visit.as_dict()
    assert data["id"] == visit.id
    assert data["started_at"] == 100.0
    assert data["ended_at"] == 170.0
    assert data["sightings"] == [
        {"at": 101.0, "cx": 300, "cy": 310, "w": 90, "h": 80, "score": 0.77, "class_id": 15}
    ]


def test_finished_visits_survive_a_restart(tmp_path: Path) -> None:
    state = tmp_path / "visits.json"
    log = VisitLog(state_file=state)
    log.start(now=100.0)
    log.sighting(pet(), now=101.0)
    log.end(now=170.0)
    log.start(now=300.0)
    log.end(now=400.0)

    reloaded = VisitLog(state_file=state)
    assert [v.started_at for v in reloaded.finished()] == [100.0, 300.0]
    assert reloaded.finished()[0].sightings[0].box == pet()
    assert reloaded.finished()[0].id == log.finished()[0].id

    reloaded.collected(reloaded.finished()[0].id)
    assert [v.started_at for v in VisitLog(state_file=state).finished()] == [300.0]


def test_open_visit_is_not_persisted_until_it_ends(tmp_path: Path) -> None:
    state = tmp_path / "visits.json"
    log = VisitLog(state_file=state)
    log.start(now=100.0)
    assert VisitLog(state_file=state).finished() == []


def test_missing_or_corrupt_state_file_starts_empty(tmp_path: Path) -> None:
    assert VisitLog(state_file=tmp_path / "none.json").finished() == []
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{not json")
    assert VisitLog(state_file=corrupt).finished() == []


def test_clips_for_visit_skips_a_clip_that_vanished(tmp_path: Path) -> None:
    log = VisitLog()
    log.start(now=1000.0)
    visit = log.end(now=1060.0)
    touch(tmp_path / "first.mp4", 1012.0)
    (tmp_path / "gone.mp4").symlink_to(tmp_path / "deleted-by-recorder.mp4")
    clips = log.clips_among(visit, recorded_clips(tmp_path), slice_seconds=10)
    assert [c.path.name for c in clips] == ["first.mp4"]


def test_state_survives_a_restart_with_boot_and_clock_offset(tmp_path: Path) -> None:
    state = tmp_path / "visits.json"
    log = VisitLog(state_file=state, boot_id="boot-a")
    log.start(now=100.0)
    log.sighting(pet(), now=101.0)
    log.end(now=170.0)
    log.clock_stepped(5000.0)
    reloaded = VisitLog(state_file=state, boot_id="boot-b").finished()
    assert [(v.started_at, v.boot, v.clock_offset) for v in reloaded] == [(100.0, "boot-a", 5000.0)]
    assert not (tmp_path / "visits.json.new").exists()


def test_a_clock_step_only_corrects_visits_from_this_boot(tmp_path: Path) -> None:
    state = tmp_path / "visits.json"
    earlier = VisitLog(state_file=state, boot_id="boot-a")
    earlier.start(now=100.0)
    earlier.end(now=170.0)
    log = VisitLog(state_file=state, boot_id="boot-b")
    log.start(now=50.0)
    log.end(now=90.0)
    log.clock_stepped(1_800_000_000.0)
    assert [v.clock_offset for v in log.finished()] == [0.0, 1_800_000_000.0]


def test_state_written_before_boots_were_recorded_still_loads(tmp_path: Path) -> None:
    state = tmp_path / "visits.json"
    state.write_text(
        '{"visits": [{"id": "v1", "started_at": 1.0, "ended_at": 2.0, "sightings": []}]}'
    )
    (visit,) = VisitLog(state_file=state, boot_id="boot-a").finished()
    assert (visit.boot, visit.clock_offset) == (None, 0.0)


def test_unreadable_state_is_set_aside_not_overwritten(tmp_path: Path) -> None:
    state = tmp_path / "visits.json"
    state.write_text('{"visits": [{"id": "v1", "started_at"')
    log = VisitLog(state_file=state)
    assert log.finished() == []
    assert (
        tmp_path / "visits.json.unreadable"
    ).read_text() == '{"visits": [{"id": "v1", "started_at"'
    log.start(now=100.0)
    log.end(now=170.0)
    assert len(VisitLog(state_file=state).finished()) == 1

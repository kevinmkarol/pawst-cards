# ABOUTME: Tests pruning of the collector's archive with real directories: old, finished visits
# ABOUTME: are deleted whole, and unfinished or recent visits and stray files are left.

import json
import os
from pathlib import Path

from collector.archive import prune_visits
from collector.service import parse_args

DAY = 86400.0
NOW = 1_800_000_000.0


def archived(
    archive: Path, name: str, age_days: float, delivered: bool = True, source: bool = True
) -> Path:
    """A visit directory as the collector leaves it, collected `age_days` before NOW."""
    visit_dir = archive / name
    (visit_dir / "clips").mkdir(parents=True)
    (visit_dir / "frames").mkdir()
    (visit_dir / "clips" / "a.mp4").write_bytes(b"v" * 3000)
    (visit_dir / "clips" / "b.mp4").write_bytes(b"v" * 2000)
    (visit_dir / "frames" / "a_0001.jpg").write_bytes(b"frame")
    (visit_dir / "pet_photo.jpg").write_bytes(b"photo")
    (visit_dir / "visit.json").write_text(json.dumps({"visit": {}, "result": {}}))
    collected_at = NOW - age_days * DAY
    os.utime(visit_dir / "visit.json", (collected_at, collected_at))
    if source:
        (visit_dir / "source.json").write_text("{}")
    if delivered:
        (visit_dir / "delivered").write_text("then\n")
    return visit_dir


def test_visits_past_the_keep_time_are_deleted_whole_and_recent_ones_kept(tmp_path) -> None:
    old = archived(tmp_path, "old", age_days=3.5)
    recent = archived(tmp_path, "recent", age_days=2.5)
    # 5000 bytes of clips, 5 of frame, 5 of photo, the record, the source and the marker.
    size = sum(path.stat().st_size for path in old.rglob("*") if path.is_file())
    assert prune_visits(tmp_path, keep_days=3, now=NOW) == (1, size)
    assert size > 5000
    assert not old.exists()
    assert len(list((recent / "clips").iterdir())) == 2
    assert (recent / "visit.json").exists()
    # A second pass finds nothing more to do.
    assert prune_visits(tmp_path, keep_days=3, now=NOW) == (0, 0)


def test_a_visit_still_waiting_to_be_delivered_is_kept(tmp_path) -> None:
    waiting = archived(tmp_path, "waiting", age_days=30, delivered=False)
    assert prune_visits(tmp_path, keep_days=3, now=NOW) == (0, 0)
    assert (waiting / "clips" / "a.mp4").exists()


def test_a_visit_archived_before_delivery_markers_existed_is_deleted(tmp_path) -> None:
    legacy = archived(tmp_path, "legacy", age_days=30, delivered=False, source=False)
    assert prune_visits(tmp_path, keep_days=3, now=NOW)[0] == 1
    assert not legacy.exists()


def test_an_old_visit_that_has_already_lost_its_clips_is_deleted(tmp_path) -> None:
    visit_dir = archived(tmp_path, "no-clips", age_days=30)
    for clip in (visit_dir / "clips").iterdir():
        clip.unlink()
    (visit_dir / "clips").rmdir()
    assert prune_visits(tmp_path, keep_days=3, now=NOW)[0] == 1
    assert not visit_dir.exists()


def test_a_download_in_progress_without_a_record_is_left_alone(tmp_path) -> None:
    (tmp_path / "partial" / "clips").mkdir(parents=True)
    (tmp_path / "partial" / "clips" / "a.mp4.part").write_bytes(b"v" * 10)
    (tmp_path / "stray-file").write_text("not a visit")
    assert prune_visits(tmp_path, keep_days=0, now=NOW) == (0, 0)
    assert (tmp_path / "partial" / "clips" / "a.mp4.part").exists()
    assert (tmp_path / "stray-file").exists()


def test_an_archive_that_does_not_exist_yet_is_nothing_to_prune(tmp_path) -> None:
    assert prune_visits(tmp_path / "missing", keep_days=3, now=NOW) == (0, 0)


def test_keep_time_comes_from_the_environment_and_defaults_to_for_ever(monkeypatch) -> None:
    monkeypatch.setenv("PETCAM_CAMERA_AGENT_URL", "http://camera:8765")
    monkeypatch.delenv("PETCAM_KEEP_VISITS_DAYS", raising=False)
    assert parse_args([]).keep_visits_days is None
    for for_ever in ("", "0"):
        monkeypatch.setenv("PETCAM_KEEP_VISITS_DAYS", for_ever)
        assert parse_args([]).keep_visits_days is None
    monkeypatch.setenv("PETCAM_KEEP_VISITS_DAYS", "3")
    assert parse_args([]).keep_visits_days == 3.0
    assert parse_args(["--keep-visits-days", "0.5"]).keep_visits_days == 0.5

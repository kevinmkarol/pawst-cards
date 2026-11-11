# ABOUTME: Tests the settings the collector reads again while it runs: what the settings file may
# ABOUTME: change, what the page is told back, test postcards, and one pass of the service by them.

import json
import sys
from pathlib import Path
from typing import List

import pytest

from collector import service
from collector.deliver import Delivery
from collector.outbox import Outbox
from collector.service import live_from, parse_args, sender_for
import logging

from collector.settings import Live, RecentLog, read_live, send_test, write_status

from tests.test_outbox import visit as waiting
from tests.test_service import camera, visit  # noqa: F401  (camera is a fixture)

START = Live(
    deliver="command",
    deliver_url=None,
    deliver_clips=False,
    deliver_every_minutes=None,
    subject="pet",
    timezone="UTC",
    keep_visits_days=None,
)


NOTHING_ASKED = {"test_at": None, "send_now_at": None}


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_without_a_settings_file_the_starting_values_stand(tmp_path: Path) -> None:
    assert read_live(tmp_path / "settings.json", START) == (START, NOTHING_ASKED)


def test_the_settings_file_replaces_only_what_it_names(tmp_path: Path) -> None:
    path = write(
        tmp_path / "settings.json",
        {
            "deliver": "webhook",
            "deliver_url": "https://hooks.example/pawst",
            "deliver_every_minutes": 30,
            "subject": " Biscuit ",
            "timezone": "Europe/Paris",
            "test_at": 12.5,
        },
    )
    live, asked = read_live(path, START)
    assert live.deliver == "webhook"
    assert live.deliver_url == "https://hooks.example/pawst"
    assert live.deliver_every_minutes == 30.0
    assert live.subject == "Biscuit"
    assert live.timezone == "Europe/Paris"
    assert live.deliver_clips is False
    assert live.keep_visits_days is None
    assert asked == {"test_at": 12.5, "send_now_at": None}


def test_null_or_zero_removes_an_interval_or_keep_time(tmp_path: Path) -> None:
    start = Live(**{**START.__dict__, "deliver_every_minutes": 60.0, "keep_visits_days": 30.0})
    path = write(tmp_path / "settings.json", {"deliver_every_minutes": None, "keep_visits_days": 0})
    live, _ = read_live(path, start)
    assert live.deliver_every_minutes is None
    assert live.keep_visits_days is None


def test_values_that_make_no_sense_are_left_as_they_started(tmp_path: Path) -> None:
    path = write(
        tmp_path / "settings.json",
        {
            "deliver": "carrier pigeon",
            "deliver_url": "file:///etc/passwd",
            "deliver_clips": "yes",
            "deliver_every_minutes": -5,
            "subject": "",
            "timezone": "Mars/Olympus",
            "keep_visits_days": True,
            "paused": "yes",
            "test_at": "soon",
        },
    )
    assert read_live(path, START) == (START, NOTHING_ASKED)
    (tmp_path / "settings.json").write_text("[1, 2")
    assert read_live(tmp_path / "settings.json", START) == (START, NOTHING_ASKED)


def test_the_status_names_the_webhook_host_and_the_command_program_only(tmp_path: Path) -> None:
    live = Live(**{**START.__dict__, "deliver_url": "https://hooks.example/T0/secret-token"})
    write_status(
        tmp_path / "collector" / "status.json",
        live,
        "/opt/bin/openclaw message send --token hunter2 {photo}",
        dry_run=False,
        problem=None,
        test=None,
    )
    text = (tmp_path / "collector" / "status.json").read_text()
    status = json.loads(text)
    assert status["settings"]["deliver_url_host"] == "hooks.example"
    assert status["command_program"] == "openclaw"
    assert "secret-token" not in text and "hunter2" not in text


def test_a_test_postcard_is_the_photo_delivered_last(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    for name, marker in (("a", "2026\n"), ("b", "2026 not posted\n")):
        (archive / name).mkdir(parents=True)
        (archive / name / "visit.json").write_text("{}")
        (archive / name / f"pet_{name}.jpg").write_bytes(b"jpeg")
        (archive / name / "delivered").write_text(marker)
    sent: List[Delivery] = []
    outcome = send_test(sent.append, archive, tmp_path / "test", requested_at=5.0)
    assert outcome["ok"] is True and outcome["requested_at"] == 5.0
    assert sent[0].photo == archive / "a" / "pet_a.jpg"
    assert "Test postcard" in sent[0].caption


def test_a_test_postcard_is_a_plain_card_while_the_archive_is_empty(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    send_test(sent.append, tmp_path / "archive", tmp_path / "test", requested_at=1.0)
    assert sent[0].photo.is_file() and sent[0].visit_json.is_file()


def test_a_test_postcard_that_fails_says_why(tmp_path: Path) -> None:
    args = parse_args(
        ["--camera", "http://cam.example", "--deliver", "command", "--deliver-command"]
        + [f"{sys.executable} -c \"import sys; sys.exit('channel not found')\" {{photo}}"]
    )
    outcome = send_test(sender_for(args, live_from(args)), tmp_path, tmp_path / "t", 1.0)
    assert outcome["ok"] is False
    assert "channel not found" in outcome["error"]


def test_a_webhook_without_an_address_is_refused(tmp_path: Path) -> None:
    args = parse_args(["--camera", "http://cam.example", "--deliver", "command"])
    live = Live(**{**START.__dict__, "deliver": "webhook"})
    with pytest.raises(ValueError):
        sender_for(args, live)


def test_removing_the_interval_sends_off_what_was_waiting(tmp_path: Path) -> None:
    first: List[Delivery] = []
    second: List[Delivery] = []
    outbox = Outbox(tmp_path, first.append, 3600)
    outbox.add(waiting(tmp_path, 1))
    assert outbox.flush() == 1
    outbox.add(waiting(tmp_path, 2))
    assert outbox.flush() == 0
    outbox.retarget(second.append, 0)
    assert outbox.flush() == 1
    assert len(first) == 1 and second[0].caption == "Pet visit 2"


def test_one_pass_delivers_by_the_settings_file_and_reports_what_it_uses(
    camera, tmp_path: Path, monkeypatch  # noqa: F811
) -> None:
    client, controller, clip_dir = camera
    visit(controller, clip_dir, start=1000.0, with_clip=True)
    log_file = tmp_path / "sent.txt"
    selector_dir = tmp_path / "selector"
    (selector_dir / "collector").mkdir(parents=True)
    write(
        selector_dir / "collector" / "settings.json",
        {"subject": "Biscuit", "timezone": "Europe/Paris", "deliver_every_minutes": 0},
    )
    code = "import sys, pathlib; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])"
    assert (
        service.main(
            [
                "--camera",
                client.base_url,
                "--deliver",
                "command",
                "--deliver-command",
                f"{sys.executable} -c '{code}' {log_file} {{caption}}",
                "--deliver-every-minutes",
                "60",
                "--selector-dir",
                str(selector_dir),
                "--archive-dir",
                str(tmp_path / "archive"),
                "--every",
                "1",
                "--once",
            ]
        )
        == 0
    )
    assert log_file.read_text().startswith("Biscuit visit")
    status = json.loads((selector_dir / "collector" / "status.json").read_text())
    assert status["settings"]["subject"] == "Biscuit"
    assert status["settings"]["timezone"] == "Europe/Paris"
    assert status["settings"]["deliver_every_minutes"] is None
    assert status["problem"] is None


def test_waiting_visits_leave_at_once_when_asked(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    outbox = Outbox(tmp_path, sent.append, 3600)
    outbox.add(waiting(tmp_path, 1))
    assert outbox.flush() == 1
    outbox.add(waiting(tmp_path, 2))
    assert outbox.flush() == 0
    assert outbox.next_due_at() is not None
    assert outbox.flush(at_once=True) == 1
    assert outbox.next_due_at() is None


def test_the_latest_log_lines_are_written_for_the_page(tmp_path: Path) -> None:
    recent = RecentLog(tmp_path / "collector" / "log.json")
    logger = logging.getLogger("test-recent")
    logger.setLevel(logging.INFO)
    logger.addHandler(recent)
    for number in range(70):
        logger.info("visit %d collected", number)
    logger.warning("camera unreachable")
    lines = json.loads((tmp_path / "collector" / "log.json").read_text())
    assert len(lines) == 60
    assert lines[-1]["level"] == "WARNING" and lines[-1]["text"] == "camera unreachable"
    assert lines[0]["text"] == "visit 11 collected"


def test_while_paused_visits_are_collected_and_wait(camera, tmp_path: Path) -> None:  # noqa: F811
    client, controller, clip_dir = camera
    visit(controller, clip_dir, start=1000.0, with_clip=True)
    log_file = tmp_path / "sent.txt"
    selector_dir = tmp_path / "selector"
    (selector_dir / "collector").mkdir(parents=True)
    settings = selector_dir / "collector" / "settings.json"
    code = "import sys, pathlib; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])"
    arguments = [
        "--camera",
        client.base_url,
        "--deliver",
        "command",
        "--deliver-command",
        f"{sys.executable} -c '{code}' {log_file} {{caption}}",
        "--selector-dir",
        str(selector_dir),
        "--archive-dir",
        str(tmp_path / "archive"),
        "--every",
        "1",
        "--once",
    ]
    write(settings, {"paused": True})
    assert service.main(arguments) == 0
    assert not log_file.exists()
    assert client.finished_visits() == []
    status = json.loads((selector_dir / "collector" / "status.json").read_text())
    assert status["waiting"] == 1 and status["settings"]["paused"] is True
    assert status["next_delivery_at"] is None

    write(settings, {"paused": False})
    assert service.main(arguments) == 0
    assert log_file.read_text().startswith("Pet visit")
    status = json.loads((selector_dir / "collector" / "status.json").read_text())
    assert status["waiting"] == 0

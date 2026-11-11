# ABOUTME: Tests one polling pass of the collector service against a real in-process agent API:
# ABOUTME: every finished visit is collected and delivered, and a failing one does not block others.

import fcntl
import json
import os
import shutil
from pathlib import Path
from typing import Iterator, List, Tuple
from zoneinfo import ZoneInfo

import pytest
from petcam.api import ControlApi
from petcam.controller import Controller
from petcam.detections import Box, Frame

from collector.camera_client import CameraClient, CameraUnreachable
from collector.deliver import Delivery
from collector.service import (
    LOCK_FILE,
    CaptureHealth,
    Reachability,
    Retries,
    live_from,
    main,
    parse_args,
    run_once,
    sender_for,
)

FIXTURE = Path(__file__).parent / "fixtures" / "visit_clip.mp4"

EASTERN = ZoneInfo("America/New_York")


class Idle:
    def move(
        self, yaw_degrees: float, yaw_speed: int, pitch_degrees: float, pitch_speed: int
    ) -> None:
        pass

    def set_recording(self, enabled: bool) -> None:
        pass


@pytest.fixture
def camera(tmp_path: Path) -> Iterator[Tuple[CameraClient, Controller, Path]]:
    idle = Idle()
    controller = Controller(
        gimbal=idle, recorder=idle, gone_after_seconds=60, move_interval_seconds=0.3
    )
    clip_dir = tmp_path / "camera_clips"
    clip_dir.mkdir()
    api = ControlApi(
        controller=controller,
        gimbal=idle,
        host="127.0.0.1",
        port=0,
        clip_dir=clip_dir,
        slice_seconds=10,
    )
    api.start()
    try:
        yield CameraClient(f"http://127.0.0.1:{api.port}"), controller, clip_dir
    finally:
        api.stop()


def visit(controller: Controller, clip_dir: Path, start: float, with_clip: bool) -> None:
    controller.observe(
        Frame(
            boxes=[Box(cx=320, cy=320, w=200, h=200, score=0.9, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=1,
        ),
        now=start,
    )
    if with_clip:
        name = f"clip_{int(start)}.mp4"
        shutil.copy(FIXTURE, clip_dir / name)
        os.utime(clip_dir / name, (start + 3, start + 3))
    controller.tick(now=start + 61)


def test_run_once_collects_every_finished_visit(camera, tmp_path: Path) -> None:
    client, controller, clip_dir = camera
    visit(controller, clip_dir, start=1000.0, with_clip=True)
    visit(controller, clip_dir, start=2000.0, with_clip=True)
    sent: List[Delivery] = []
    collected = run_once(
        client,
        archive_dir=tmp_path / "archive",
        every_seconds=1,
        send=sent.append,
        zone=EASTERN,
        subject="pet",
    )
    assert collected == 2
    assert len(sent) == 2
    assert client.finished_visits() == []


def test_run_once_collects_the_rest_when_one_visit_has_no_clips(camera, tmp_path: Path) -> None:
    client, controller, clip_dir = camera
    visit(controller, clip_dir, start=1000.0, with_clip=False)
    visit(controller, clip_dir, start=2000.0, with_clip=True)
    sent: List[Delivery] = []
    collected = run_once(
        client,
        archive_dir=tmp_path / "archive",
        every_seconds=1,
        send=sent.append,
        zone=EASTERN,
        subject="pet",
    )
    assert collected == 1
    assert len(sent) == 1
    # The camera drops a visit that has no clips, so there is nothing left to retry.
    assert client.finished_visits() == []


def test_run_once_with_nothing_to_do(camera, tmp_path: Path) -> None:
    client, _, _ = camera
    assert (
        run_once(
            client,
            archive_dir=tmp_path,
            every_seconds=5,
            send=lambda delivery: None,
            zone=EASTERN,
            subject="pet",
        )
        == 0
    )


def test_settings_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("PETCAM_CAMERA_AGENT_URL", "http://cam.example:8765")
    monkeypatch.setenv("PETCAM_TIMEZONE", "Europe/Paris")
    monkeypatch.setenv("PETCAM_SUBJECT", "dog")
    monkeypatch.setenv("PETCAM_DELIVER", "command")
    monkeypatch.setenv("PETCAM_DELIVER_COMMAND", "post {photo} {caption}")
    monkeypatch.setenv("PETCAM_STAGE_DIR", "/tmp/stage")
    monkeypatch.setenv("PETCAM_DELIVER_URL", "https://hooks.example/pawst")
    monkeypatch.setenv("PETCAM_DELIVER_CLIPS", "true")
    monkeypatch.setenv("PETCAM_DELIVER_EVERY_MINUTES", "60")
    args = parse_args([])
    assert args.deliver_every_minutes == 60.0
    assert args.camera == "http://cam.example:8765"
    assert args.timezone == ZoneInfo("Europe/Paris")
    assert args.subject == "dog"
    assert args.deliver == "command"
    assert args.deliver_command == "post {photo} {caption}"
    assert args.stage_dir == Path("/tmp/stage")
    assert args.deliver_url == "https://hooks.example/pawst"
    assert args.deliver_clips is True
    assert args.every == 5.0
    assert args.poll == 15.0
    assert args.dry_run is False
    assert args.mirror is True


def test_camera_is_required_without_the_environment(monkeypatch, capsys) -> None:
    monkeypatch.delenv("PETCAM_CAMERA_AGENT_URL", raising=False)
    with pytest.raises(SystemExit):
        parse_args([])
    assert "--camera" in capsys.readouterr().err


def test_defaults_without_the_environment(monkeypatch) -> None:
    for name in [
        "PETCAM_TIMEZONE",
        "PETCAM_SUBJECT",
        "PETCAM_DELIVER",
        "PETCAM_DELIVER_COMMAND",
        "PETCAM_STAGE_DIR",
        "PETCAM_DELIVER_URL",
        "PETCAM_DELIVER_CLIPS",
        "PETCAM_DELIVER_EVERY_MINUTES",
    ]:
        monkeypatch.delenv(name, raising=False)
    args = parse_args(["--camera", "http://cam.example:8765"])
    assert args.timezone == ZoneInfo("UTC")
    assert args.subject == "pet"
    assert args.deliver is None
    assert args.deliver_every_minutes is None
    assert args.stage_dir is None
    assert args.deliver_clips is False


def test_mirroring_can_be_switched_off() -> None:
    args = parse_args(["--camera", "http://cam.example:8765", "--no-mirror"])
    assert args.mirror is False


def test_the_sender_runs_the_command_template(tmp_path: Path) -> None:
    record = tmp_path / "record.txt"
    args = parse_args(
        [
            "--camera",
            "http://cam.example:8765",
            "--deliver",
            "command",
            "--deliver-command",
            f"sh -c 'echo \"$1\" > {record}' sh {{caption}}",
        ]
    )
    photo = tmp_path / "pet.jpg"
    photo.write_bytes(b"jpeg")
    send = sender_for(args, live_from(args))
    send(Delivery(caption="Pet visit", photo=photo, visit_dir=tmp_path, clips=[]))
    assert record.read_text().strip() == "Pet visit"


def test_a_dry_run_sender_logs_instead_of_delivering(tmp_path: Path, caplog) -> None:
    args = parse_args(["--camera", "http://cam.example:8765", "--dry-run"])
    photo = tmp_path / "pet.jpg"
    send = sender_for(args, live_from(args))
    with caplog.at_level("INFO"):
        send(Delivery(caption="Pet visit", photo=photo, visit_dir=tmp_path, clips=[]))
    assert "dry run" in caplog.text and "Pet visit" in caplog.text


def test_the_sender_needs_a_delivery_method_and_its_settings(monkeypatch) -> None:
    for name in ("PETCAM_DELIVER", "PETCAM_DELIVER_URL", "PETCAM_DELIVER_COMMAND"):
        monkeypatch.delenv(name, raising=False)
    for extra, missing in (
        ([], "--deliver is required"),
        (["--deliver", "webhook"], "--deliver-url"),
        (["--deliver", "command"], "--deliver-command"),
    ):
        args = parse_args(["--camera", "http://cam.example:8765", *extra])
        with pytest.raises(ValueError, match=missing):
            sender_for(args, live_from(args))
    monkeypatch.setenv("PETCAM_DELIVER", "Webhook")
    args = parse_args(["--camera", "http://cam.example:8765"])
    with pytest.raises(ValueError, match="must be command or webhook"):
        sender_for(args, live_from(args))


def test_settings_the_page_saved_that_cannot_be_used_do_not_stop_the_start(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("PETCAM_DELIVER_URL", raising=False)
    selector_dir = tmp_path / "selector"
    (selector_dir / "collector").mkdir(parents=True)
    (selector_dir / "collector" / "settings.json").write_text('{"deliver": "webhook"}')
    argv = ["--camera", "http://127.0.0.1:1", "--once", "--deliver", "command"]
    argv += ["--selector-dir", str(selector_dir), "--archive-dir", str(tmp_path / "visits")]
    assert main([*argv, "--deliver-command", "true {photo}"]) == 0
    status = json.loads((selector_dir / "collector" / "status.json").read_text())
    assert status["settings"]["deliver"] == "command" and "--deliver-url" in status["problem"]
    # Starting values that cannot be used are still a usage error.
    with pytest.raises(SystemExit):
        main(argv)


def test_a_second_collector_on_the_same_archive_does_not_start(tmp_path: Path) -> None:
    archive = tmp_path / "visits"
    archive.mkdir()
    argv = ["--camera", "http://127.0.0.1:1", "--once", "--dry-run"]
    argv += ["--selector-dir", str(tmp_path / "selector"), "--archive-dir", str(archive)]
    with open(archive / LOCK_FILE, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert main(argv) == 1
    assert main(argv) == 0


def test_an_empty_camera_address_is_a_usage_error(monkeypatch) -> None:
    monkeypatch.setenv("PETCAM_CAMERA_AGENT_URL", "")
    with pytest.raises(SystemExit):
        parse_args([])


def test_run_once_survives_a_clip_ffmpeg_cannot_read(camera, tmp_path: Path) -> None:
    client, controller, clip_dir = camera
    controller.observe(
        Frame(
            boxes=[Box(cx=320, cy=320, w=200, h=200, score=0.9, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=1,
        ),
        now=1000.0,
    )
    (clip_dir / "broken.mp4").write_bytes(b"not a video" * 100)
    os.utime(clip_dir / "broken.mp4", (1003.0, 1003.0))
    controller.tick(now=1061.0)
    sent: List[Delivery] = []
    collected = run_once(
        client,
        archive_dir=tmp_path / "archive",
        every_seconds=1,
        send=sent.append,
        zone=EASTERN,
        subject="pet",
    )
    assert collected == 0
    assert sent == []
    assert len(client.finished_visits()) == 1


def test_delivery_failure_is_logged_with_its_output_and_visit_is_kept(
    camera, tmp_path: Path, caplog
) -> None:
    import subprocess

    client, controller, clip_dir = camera
    visit(controller, clip_dir, start=1000.0, with_clip=True)

    def failing_send(delivery: Delivery) -> None:
        raise subprocess.CalledProcessError(
            1, ["openclaw"], output="", stderr="gateway refused: boom"
        )

    with caplog.at_level("ERROR"):
        collected = run_once(
            client,
            archive_dir=tmp_path / "archive",
            every_seconds=1,
            send=failing_send,
            zone=EASTERN,
            subject="pet",
        )
    assert collected == 0
    assert "gateway refused: boom" in caplog.text
    assert len(client.finished_visits()) == 1


class HandClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def unreadable_visit(controller: Controller, clip_dir: Path, start: float) -> None:
    controller.observe(
        Frame(
            boxes=[Box(cx=320, cy=320, w=200, h=200, score=0.9, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=1,
        ),
        now=start,
    )
    (clip_dir / f"broken_{int(start)}.mp4").write_bytes(b"not a video" * 100)
    os.utime(clip_dir / f"broken_{int(start)}.mp4", (start + 3, start + 3))
    controller.tick(now=start + 61)


def collect(client: CameraClient, tmp_path: Path, send, retries: Retries) -> int:
    return run_once(
        client,
        archive_dir=tmp_path / "archive",
        every_seconds=1,
        send=send,
        zone=EASTERN,
        subject="pet",
        retries=retries,
    )


def test_a_backlog_larger_than_one_request_is_collected_in_one_pass(camera, tmp_path) -> None:
    client, controller, clip_dir = camera
    for number in range(12):
        visit(controller, clip_dir, start=1000.0 * (number + 1), with_clip=True)
    sent: List[Delivery] = []
    assert collect(client, tmp_path, sent.append, Retries()) == 12
    assert len(sent) == 12
    assert client.finished_visits() == []


def test_a_limited_pass_stops_after_that_many_visits(camera, tmp_path) -> None:
    client, controller, clip_dir = camera
    for number in range(3):
        visit(controller, clip_dir, start=1000.0 * (number + 1), with_clip=True)
    sent: List[Delivery] = []
    collected = run_once(
        client,
        archive_dir=tmp_path / "archive",
        every_seconds=1,
        send=sent.append,
        zone=EASTERN,
        subject="pet",
        limit=1,
    )
    assert collected == 1 and len(sent) == 1
    assert len(client.finished_visits()) == 2


def test_visits_that_keep_failing_do_not_hide_the_ones_behind_them(camera, tmp_path) -> None:
    client, controller, clip_dir = camera
    # More failing visits than one request asks for, all older than the good one.
    for number in range(7):
        unreadable_visit(controller, clip_dir, start=1000.0 * (number + 1))
    visit(controller, clip_dir, start=9000.0, with_clip=True)
    sent: List[Delivery] = []
    assert collect(client, tmp_path, sent.append, Retries()) == 1
    assert len(sent) == 1
    assert len(client.finished_visits()) == 7


def test_a_failed_visit_is_left_alone_until_its_retry_is_due(camera, tmp_path) -> None:
    client, controller, clip_dir = camera
    visit(controller, clip_dir, start=1000.0, with_clip=True)
    clock = HandClock()
    retries = Retries(first_seconds=30, longest_seconds=100, clock=clock)
    attempts: List[Delivery] = []

    def target_is_down(delivery: Delivery) -> None:
        attempts.append(delivery)
        raise OSError("delivery target is down")

    assert collect(client, tmp_path, target_is_down, retries) == 0
    clock.now = 29.0
    assert collect(client, tmp_path, target_is_down, retries) == 0
    assert len(attempts) == 1
    clock.now = 30.0
    assert collect(client, tmp_path, target_is_down, retries) == 0
    assert len(attempts) == 2
    # The second wait is twice the first, and waits stop growing at the longest.
    clock.now = 89.0
    assert collect(client, tmp_path, target_is_down, retries) == 0
    assert len(attempts) == 2
    clock.now = 90.0
    assert collect(client, tmp_path, target_is_down, retries) == 0
    assert retries.failed("another-visit") == 30
    assert [retries.failed("another-visit") for _ in range(3)] == [60, 100, 100]

    delivered: List[Delivery] = []
    clock.now = 1000.0
    assert collect(client, tmp_path, delivered.append, retries) == 1
    # The visit succeeded, and "another-visit" is not on the camera, so it is forgotten.
    assert retries.waiting() == 0


def test_a_camera_that_stops_answering_ends_the_pass(camera, tmp_path) -> None:
    client, controller, clip_dir = camera
    visit(controller, clip_dir, start=1000.0, with_clip=True)
    visit(controller, clip_dir, start=2000.0, with_clip=True)

    class DropsOut(CameraClient):
        def download_clip(self, name, dest, expected_size=None) -> None:
            raise CameraUnreachable(f"clip {name}: timed out")

    retries = Retries()
    with pytest.raises(CameraUnreachable):
        collect(DropsOut(client.base_url), tmp_path, lambda delivery: None, retries)
    # Being out of reach is not held against the visits.
    assert retries.waiting() == 0
    assert len(client.finished_visits()) == 2


def test_an_unreachable_camera_is_logged_once_then_hourly_and_on_return(caplog) -> None:
    clock = HandClock()
    reachability = Reachability(clock=clock)
    with caplog.at_level("INFO"):
        for second in range(0, 7300, 15):
            clock.now = float(second)
            reachability.unreachable(CameraUnreachable("GET /visits: timed out"))
        clock.now = 7300.0
        reachability.reachable()
        reachability.reachable()
    assert [record.getMessage() for record in caplog.records] == [
        "camera unreachable: GET /visits: timed out",
        "camera still unreachable after 60 min: GET /visits: timed out",
        "camera still unreachable after 120 min: GET /visits: timed out",
        "camera reachable again after 122 min",
    ]


def test_stalled_capture_is_logged_when_it_starts_and_when_it_ends(caplog) -> None:
    health = CaptureHealth()
    with caplog.at_level("INFO"):
        health.report({"in_visit": False})  # an agent that reports no capture health
        health.report({"capture": {"flowing": True, "restarts": 0}})
        health.report({"capture": {"flowing": False, "restarts": 1}})
        health.report({"capture": {"flowing": False, "restarts": 2}})
        health.report({"capture": {"flowing": True, "restarts": 2}})
    assert [record.getMessage() for record in caplog.records] == [
        "camera is not producing detection frames (capture restarts so far: 1)",
        "camera is producing detection frames again",
    ]


def test_a_blank_time_zone_means_utc_and_bad_settings_are_usage_errors(monkeypatch, capsys) -> None:
    monkeypatch.setenv("PETCAM_CAMERA_AGENT_URL", "http://cam.example:8765")
    monkeypatch.setenv("PETCAM_TIMEZONE", "")
    assert str(parse_args([]).timezone) == "UTC"
    for name, value in [
        ("PETCAM_TIMEZONE", "Mars/Olympus"),
        ("PETCAM_KEEP_VISITS_DAYS", "a week"),
        ("PETCAM_DELIVER_EVERY_MINUTES", "hourly"),
    ]:
        monkeypatch.setenv(name, value)
        with pytest.raises(SystemExit) as stopped:
            parse_args([])
        assert stopped.value.code == 2
        monkeypatch.delenv(name)
    assert "Mars/Olympus" in capsys.readouterr().err


def test_a_visit_that_has_failed_thousands_of_times_still_gets_a_wait() -> None:
    retries = Retries(clock=lambda: 0.0)
    assert [retries.failed("stuck") for _ in range(2000)][-1] == 600

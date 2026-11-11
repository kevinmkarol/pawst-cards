# ABOUTME: Tests the collector's client for the camera agent against the agent's real HTTP API,
# ABOUTME: served in-process with a temporary clip directory.

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator, Tuple

import pytest
from petcam.api import ControlApi
from petcam.controller import Controller
from petcam.detections import Box, Frame

from collector.camera_client import CameraClient, CameraError, CameraUnreachable


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
    api = ControlApi(
        controller=controller,
        gimbal=idle,
        host="127.0.0.1",
        port=0,
        clip_dir=tmp_path,
        slice_seconds=10,
    )
    api.start()
    try:
        yield CameraClient(f"http://127.0.0.1:{api.port}"), controller, tmp_path
    finally:
        api.stop()


def finish_visit(controller: Controller, clip_dir: Path) -> str:
    controller.observe(
        Frame(
            boxes=[Box(cx=320, cy=320, w=50, h=50, score=0.9, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=1,
        ),
        now=1000.0,
    )
    (clip_dir / "clip.mp4").write_bytes(b"video" * 1000)
    os.utime(clip_dir / "clip.mp4", (1010.0, 1010.0))
    controller.tick(now=1060.0)
    return controller.visits.finished()[0].id


def test_status_and_empty_visits(camera) -> None:
    client, _, _ = camera
    assert client.status()["in_visit"] is False
    assert client.finished_visits() == []


def test_finished_visit_is_listed_downloaded_and_deleted(camera, tmp_path: Path) -> None:
    client, controller, clip_dir = camera
    visit_id = finish_visit(controller, clip_dir)
    visits = client.finished_visits()
    assert [v.id for v in visits] == [visit_id]
    visit = visits[0]
    assert visit.started_at == 1000.0 and visit.ended_at == 1060.0
    assert [c.name for c in visit.clips] == ["clip.mp4"]
    assert visit.sightings[0].score == 0.9

    dest = tmp_path / "downloaded" / "clip.mp4"
    client.download_clip("clip.mp4", dest)
    assert dest.read_bytes() == b"video" * 1000

    client.delete_visit(visit_id)
    assert client.finished_visits() == []
    assert not (clip_dir / "clip.mp4").exists()


def test_errors_are_raised_as_camera_error(camera, tmp_path: Path) -> None:
    client, _, _ = camera
    with pytest.raises(CameraError):
        client.download_clip("missing.mp4", tmp_path / "x.mp4")
    with pytest.raises(CameraError):
        client.delete_visit("nope")
    with pytest.raises(CameraError):
        CameraClient("http://127.0.0.1:9").status()


def test_a_download_that_comes_up_short_is_not_a_complete_clip(camera, tmp_path: Path) -> None:
    # A download cut short (for example by an agent restart) must not pass as a complete clip.
    client, controller, clip_dir = camera
    finish_visit(controller, clip_dir)
    clip = client.finished_visits()[0].clips[0]
    # tmp_path doubles as the camera's clip directory here, so download somewhere else.
    dest = tmp_path / "downloaded" / "clip.mp4"
    with pytest.raises(CameraUnreachable):
        client.download_clip(clip.name, dest, expected_size=clip.size + 1)
    assert not dest.exists()
    # What did arrive is kept for the next attempt to add to.
    assert (tmp_path / "downloaded" / "clip.mp4.part").stat().st_size == clip.size


def test_a_download_carries_on_from_what_an_interrupted_one_left(camera, tmp_path: Path) -> None:
    client, controller, clip_dir = camera
    finish_visit(controller, clip_dir)
    clip = client.finished_visits()[0].clips[0]
    dest = tmp_path / "downloaded" / "clip.mp4"
    dest.parent.mkdir()
    # Bytes that are not the clip's, so the result shows whether they were fetched again.
    (dest.parent / "clip.mp4.part").write_bytes(b"#" * 1200)
    client.download_clip(clip.name, dest, expected_size=clip.size)
    assert dest.read_bytes() == b"#" * 1200 + (b"video" * 1000)[1200:]
    assert not (dest.parent / "clip.mp4.part").exists()


def test_a_download_larger_than_expected_is_rejected(camera, tmp_path: Path) -> None:
    client, controller, clip_dir = camera
    finish_visit(controller, clip_dir)
    clip = client.finished_visits()[0].clips[0]
    dest = tmp_path / "downloaded" / "clip.mp4"
    with pytest.raises(CameraError) as raised:
        client.download_clip(clip.name, dest, expected_size=clip.size - 1)
    assert not isinstance(raised.value, CameraUnreachable)
    assert not dest.exists()
    assert not (tmp_path / "downloaded" / "clip.mp4.part").exists()


def test_visits_come_a_page_at_a_time_with_the_total(camera) -> None:
    client, controller, clip_dir = camera
    for start in (2000.0, 3000.0, 4000.0):
        controller.visits.start(now=start)
        controller.visits.end(now=start + 60)
        (clip_dir / f"{start:.0f}.mp4").write_bytes(b"clip")
        os.utime(clip_dir / f"{start:.0f}.mp4", (start + 5, start + 5))
    visits, total = client.finished_visits_page(limit=2)
    assert [v.started_at for v in visits] == [2000.0, 3000.0]
    assert total == 3


def test_a_visit_recorded_before_the_clock_was_set_carries_its_correction(camera) -> None:
    client, controller, clip_dir = camera
    finish_visit(controller, clip_dir)
    controller.visits.finished()[0].clock_offset = 1_800_000_000.0
    visit = client.finished_visits()[0]
    assert visit.started_at == 1000.0
    assert visit.real_started_at == 1_800_001_000.0


def test_an_unreachable_camera_is_told_apart_from_a_refusal(camera) -> None:
    client, _, _ = camera
    with pytest.raises(CameraError) as refused:
        client.delete_visit("nope")
    assert not isinstance(refused.value, CameraUnreachable)
    with pytest.raises(CameraUnreachable):
        CameraClient("http://127.0.0.1:9").status()


def test_download_skips_a_complete_local_copy(camera, tmp_path: Path) -> None:
    client, controller, clip_dir = camera
    finish_visit(controller, clip_dir)
    clip = client.finished_visits()[0].clips[0]
    dest = tmp_path / "downloaded" / "clip.mp4"
    client.download_clip(clip.name, dest, expected_size=clip.size)
    first_mtime = dest.stat().st_mtime_ns
    (clip_dir / clip.name).unlink()  # a second fetch would now fail, so it must not happen
    client.download_clip(clip.name, dest, expected_size=clip.size)
    assert dest.stat().st_mtime_ns == first_mtime


def visits_listing(visit_id: str, clip_name: str) -> dict:
    clip = {"name": clip_name, "size": 4, "mtime": 1010.0}
    return {
        "visits": [
            {
                "id": visit_id,
                "started_at": 1000.0,
                "ended_at": 1060.0,
                "sightings": [],
                "clips": [clip],
            }
        ]
    }


@pytest.fixture
def answering() -> Iterator[Tuple[CameraClient, dict]]:
    """A client for a server that answers GET /visits with whatever listing the test puts in
    the dict, standing in for a camera that cannot be trusted."""
    listing: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps(listing).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield CameraClient(f"http://127.0.0.1:{server.server_port}"), listing
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "visit_id",
    [
        "../../outside",
        "/tmp/outside",
        "nested/visit",
        "..",
        ".",
        "",
        "nul\x00byte",
        "<img src=x>",
        "visit\n",
        7,
    ],
)
def test_a_visit_id_that_is_not_a_plain_name_is_refused(answering, caplog, visit_id) -> None:
    # The id becomes a directory under the archive, so it must not be able to point elsewhere.
    client, listing = answering
    listing.update(visits_listing(visit_id, "clip.mp4"))
    assert client.finished_visits() == []
    assert "unusable visit id" in caplog.text


@pytest.mark.parametrize(
    "clip_name",
    [
        "../../../.zshenv",
        "/tmp/outside.mp4",
        "clips/nested.mp4",
        "..",
        "",
        "nul\x00byte.mp4",
        '"><script>.mp4',
        None,
    ],
)
def test_a_clip_name_that_is_not_a_plain_name_is_refused(answering, caplog, clip_name) -> None:
    # The name becomes a file in the visit's archive directory, so it must stay inside it.
    client, listing = answering
    listing.update(visits_listing("20261101-102524-53", clip_name))
    assert client.finished_visits() == []
    assert "clip name" in caplog.text


def test_plain_names_from_the_camera_are_accepted(answering) -> None:
    client, listing = answering
    listing.update(visits_listing("20261101-102524-53", "20261101_102528.mp4"))
    [visit] = client.finished_visits()
    assert visit.id == "20261101-102524-53"
    assert [clip.name for clip in visit.clips] == ["20261101_102528.mp4"]


def test_names_with_url_characters_are_asked_for_whole() -> None:
    assert (
        CameraClient("http://camera")._path("/clips/", "a b?c#d.mp4") == "/clips/a%20b%3Fc%23d.mp4"
    )


def test_one_unusable_visit_does_not_hide_the_rest(answering) -> None:
    client, listing = answering
    good = visits_listing("20261101-102524-53", "clip.mp4")["visits"][0]
    listing.update({"visits": [{"id": "20261101-090000-12"}, good], "total": 2})
    visits, total = client.finished_visits_page(limit=None)
    assert [visit.id for visit in visits] == ["20261101-102524-53"] and total == 2


def test_a_listing_with_no_usable_visit_is_empty_and_logged_once(answering, caplog) -> None:
    client, listing = answering
    listing.update({"visits": [{"id": "20261101-090000-12"}], "total": 1})
    assert client.finished_visits_page(limit=5) == ([], 1)
    assert client.finished_visits_page(limit=5) == ([], 1)
    assert caplog.text.count("its record is unusable") == 1

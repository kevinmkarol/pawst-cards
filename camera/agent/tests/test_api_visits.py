# ABOUTME: Tests the visit collection endpoints: listing finished visits with their clips,
# ABOUTME: downloading a clip, and deleting a collected visit together with its clips.

import io
import json
import os
import time
from email.message import Message
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Tuple

import pytest

from petcam.api import ControlApi, copy_bytes
from petcam.controller import Controller
from petcam.detections import Box, Frame


class FakeGimbal:
    def move(
        self, yaw_degrees: float, yaw_speed: int, pitch_degrees: float, pitch_speed: int
    ) -> None:
        pass


class FakeRecorder:
    def set_recording(self, enabled: bool) -> None:
        pass


@pytest.fixture
def served(tmp_path: Path) -> Iterator[Tuple[str, Controller, Path]]:
    gimbal = FakeGimbal()
    controller = Controller(
        gimbal=gimbal,
        recorder=FakeRecorder(),
        gone_after_seconds=60,
        move_interval_seconds=0.3,
    )
    api = ControlApi(
        controller=controller,
        gimbal=gimbal,
        host="127.0.0.1",
        port=0,
        clip_dir=tmp_path,
        slice_seconds=10,
    )
    api.start()
    try:
        yield f"http://127.0.0.1:{api.port}", controller, tmp_path
    finally:
        api.stop()


def finish_a_visit(controller: Controller, clip_dir: Path) -> str:
    controller.observe(
        Frame(
            boxes=[Box(cx=320, cy=320, w=50, h=50, score=0.9, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=1,
        ),
        now=1000.0,
    )
    for name, mtime in (("a.mp4", 1005.0), ("b.mp4", 1066.0), ("old.mp4", 900.0)):
        (clip_dir / name).write_bytes(name.encode() * 100)
        os.utime(clip_dir / name, (mtime, mtime))
    controller.tick(now=1060.0)
    return controller.visits.finished()[0].id


def request(url: str, method: str = "GET", headers: dict = {}) -> Tuple[int, bytes, Message]:
    req = urllib.request.Request(url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers


def test_visits_lists_finished_visits_with_clips(served) -> None:
    base, controller, clip_dir = served
    visit_id = finish_a_visit(controller, clip_dir)
    status, body, _ = request(f"{base}/visits")
    assert status == 200
    visits = json.loads(body)["visits"]
    assert len(visits) == 1
    assert visits[0]["id"] == visit_id
    assert visits[0]["clips"] == [
        {"name": "a.mp4", "size": 500, "mtime": 1005.0},
        {"name": "b.mp4", "size": 500, "mtime": 1066.0},
    ]
    assert visits[0]["sightings"][0]["score"] == 0.9


def test_clip_download_and_unknown_or_unsafe_names(served) -> None:
    base, controller, clip_dir = served
    finish_a_visit(controller, clip_dir)
    status, body, headers = request(f"{base}/clips/a.mp4")
    assert status == 200
    assert body == b"a.mp4" * 100
    assert headers["Content-Type"] == "video/mp4"
    assert request(f"{base}/clips/missing.mp4")[0] == 404
    assert request(f"{base}/clips/..%2F..%2Fetc%2Fpasswd")[0] == 404


def test_delete_visit_removes_its_clips_only(served) -> None:
    base, controller, clip_dir = served
    visit_id = finish_a_visit(controller, clip_dir)
    status, _, _ = request(f"{base}/visits/{visit_id}", method="DELETE")
    assert status == 200
    assert sorted(p.name for p in clip_dir.iterdir()) == ["old.mp4"]
    assert json.loads(request(f"{base}/visits")[1])["visits"] == []
    assert request(f"{base}/visits/{visit_id}", method="DELETE")[0] == 404


def test_visits_listing_skips_a_clip_that_vanished(served) -> None:
    base, controller, clip_dir = served
    finish_a_visit(controller, clip_dir)
    # The recorder deletes its oldest clips when storage fills. A dangling link is found by the
    # directory scan and gone by the time it is examined, like a clip deleted in between.
    (clip_dir / "gone.mp4").symlink_to(clip_dir / "deleted-by-recorder.mp4")
    status, body, _ = request(f"{base}/visits")
    assert status == 200
    assert [clip["name"] for clip in json.loads(body)["visits"][0]["clips"]] == ["a.mp4", "b.mp4"]


def test_visits_listing_examines_each_clip_once_however_many_visits(
    served, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, controller, clip_dir = served
    finish_a_visit(controller, clip_dir)
    for start in (2000.0, 3000.0, 4000.0):
        controller.visits.start(now=start)
        controller.visits.end(now=start + 60)
        (clip_dir / f"{start:.0f}.mp4").write_bytes(b"clip")
        os.utime(clip_dir / f"{start:.0f}.mp4", (start + 5, start + 5))
    examined = []
    real_stat = Path.stat

    def counting_stat(path: Path, *args, **kwargs) -> os.stat_result:
        if path.suffix == ".mp4":
            examined.append(path.name)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", counting_stat)
    status, body, _ = request(f"{base}/visits")
    assert status == 200
    assert len(json.loads(body)["visits"]) == 4
    assert sorted(examined) == ["2000.mp4", "3000.mp4", "4000.mp4", "a.mp4", "b.mp4", "old.mp4"]


def finish_visits_with_clips(controller: Controller, clip_dir: Path, starts) -> None:
    for start in starts:
        controller.visits.start(now=start)
        controller.visits.end(now=start + 60)
        (clip_dir / f"{start:.0f}.mp4").write_bytes(b"clip")
        os.utime(clip_dir / f"{start:.0f}.mp4", (start + 5, start + 5))


def test_visits_can_be_listed_a_few_at_a_time_oldest_first(served) -> None:
    base, controller, clip_dir = served
    finish_visits_with_clips(controller, clip_dir, (2000.0, 3000.0, 4000.0))
    status, body, _ = request(f"{base}/visits?limit=2")
    assert status == 200
    listing = json.loads(body)
    assert [v["started_at"] for v in listing["visits"]] == [2000.0, 3000.0]
    assert listing["total"] == 3
    assert json.loads(request(f"{base}/visits")[1])["total"] == 3
    assert request(f"{base}/visits?limit=0")[0] == 400
    assert request(f"{base}/visits?limit=many")[0] == 400


def test_a_finished_visit_whose_clips_are_gone_is_dropped(served) -> None:
    base, controller, clip_dir = served
    finish_visits_with_clips(controller, clip_dir, (2000.0, 3000.0))
    (clip_dir / "2000.mp4").unlink()
    listing = json.loads(request(f"{base}/visits")[1])
    assert [v["started_at"] for v in listing["visits"]] == [3000.0]
    assert [v.started_at for v in controller.visits.finished()] == [3000.0]


def test_a_visit_that_just_ended_is_listed_before_its_clip_is_closed(served) -> None:
    base, controller, clip_dir = served
    now = time.time()
    controller.visits.start(now=now - 20)
    controller.visits.end(now=now - 1)
    listing = json.loads(request(f"{base}/visits")[1])
    assert len(listing["visits"]) == 1
    assert listing["visits"][0]["clips"] == []


def test_a_clip_download_can_resume_from_an_offset(served) -> None:
    base, controller, clip_dir = served
    finish_a_visit(controller, clip_dir)
    whole = b"a.mp4" * 100
    status, body, headers = request(f"{base}/clips/a.mp4", headers={"Range": "bytes=123-"})
    assert status == 206
    assert body == whole[123:]
    assert headers["Content-Range"] == "bytes 123-499/500"
    # An offset at or past the end, or any other kind of range, gets the whole clip.
    for unusable in ("bytes=500-", "bytes=0-10", "bytes=-5", "chunks=3-"):
        status, body, _ = request(f"{base}/clips/a.mp4", headers={"Range": unusable})
        assert (status, body) == (200, whole)


def test_status_includes_the_health_report(tmp_path: Path) -> None:
    gimbal = FakeGimbal()
    controller = Controller(
        gimbal=gimbal, recorder=FakeRecorder(), gone_after_seconds=60, move_interval_seconds=0.3
    )
    api = ControlApi(
        controller=controller,
        gimbal=gimbal,
        host="127.0.0.1",
        port=0,
        clip_dir=tmp_path,
        health=lambda: {"capture": {"flowing": True}},
    )
    api.start()
    try:
        status = json.loads(request(f"http://127.0.0.1:{api.port}/status")[1])
    finally:
        api.stop()
    assert status["capture"] == {"flowing": True}
    assert status["in_visit"] is False


def test_deleting_a_visit_leaves_a_clip_the_next_visit_also_needs(served) -> None:
    # A visit that starts as the last one ends shares the slice being written at that moment.
    base, controller, clip_dir = served
    controller.visits.start(now=2000.0)
    first = controller.visits.end(now=2060.0)
    controller.visits.start(now=2060.5)
    second = controller.visits.end(now=2100.0)
    for name, mtime in (("own.mp4", 2010.0), ("shared.mp4", 2070.0), ("later.mp4", 2090.0)):
        (clip_dir / name).write_bytes(b"clip")
        os.utime(clip_dir / name, (mtime, mtime))
    assert request(f"{base}/visits/{first.id}", method="DELETE")[0] == 200
    assert sorted(path.name for path in clip_dir.iterdir()) == ["later.mp4", "shared.mp4"]
    assert request(f"{base}/visits/{second.id}", method="DELETE")[0] == 200
    assert list(clip_dir.iterdir()) == []


def test_deleting_a_visit_leaves_the_clip_a_visit_in_progress_is_recording(served) -> None:
    base, controller, clip_dir = served
    controller.visits.start(now=2000.0)
    first = controller.visits.end(now=2060.0)
    controller.visits.start(now=2060.5)
    for name, mtime in (("own.mp4", 2010.0), ("recording.mp4", 2065.0)):
        (clip_dir / name).write_bytes(b"clip")
        os.utime(clip_dir / name, (mtime, mtime))
    assert request(f"{base}/visits/{first.id}", method="DELETE")[0] == 200
    assert [path.name for path in clip_dir.iterdir()] == ["recording.mp4"]


def test_a_clip_that_grew_since_its_length_was_promised_is_cut_at_that_length() -> None:
    # The recorder is still writing a visit's last slice when the collector asks for it.
    sent = io.BytesIO()
    copy_bytes(io.BytesIO(b"a" * 150000), sent, 100000)
    assert len(sent.getvalue()) == 100000


@pytest.mark.parametrize("host", ["rebound.example:8765", "evil.example"])
def test_a_request_under_a_name_that_is_not_the_cameras_is_refused(served, host: str) -> None:
    # A page elsewhere can point a domain of its own at the camera's address, so that a browser
    # takes the agent for part of that page's site; the request then carries that domain.
    base, _, _ = served
    assert request(f"{base}/status", headers={"Host": host})[0] == 403
    for name in ("recamera.local:8765", "recamera", "recamera.tail1234.ts.net:8765", "[::1]:8765"):
        assert request(f"{base}/status", headers={"Host": name})[0] == 200

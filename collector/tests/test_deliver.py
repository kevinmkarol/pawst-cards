# ABOUTME: Tests the delivery sinks with real processes and a real local HTTP server: a command
# ABOUTME: template runs with the photo and caption as arguments, and a webhook receives them.

import json
import os
import subprocess
import threading
import time
import urllib.error
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator, List

import pytest

from collector.deliver import Delivery, command_sender, webhook_sender


@pytest.fixture
def delivery(tmp_path: Path) -> Delivery:
    visit_dir = tmp_path / "archive" / "20261101-102524-53"
    (visit_dir / "clips").mkdir(parents=True)
    photo = visit_dir / "pet_2026-11-01_6-25am.jpg"
    photo.write_bytes(b"jpegdata")
    clip = visit_dir / "clips" / "20261101_102528.mp4"
    clip.write_bytes(b"mp4data")
    (visit_dir / "visit.json").write_text(json.dumps({"visit": {"id": "20261101-102524-53"}}))
    return Delivery(caption="Pet visit Sun 6:25 AM", photo=photo, visit_dir=visit_dir, clips=[clip])


def test_command_sender_substitutes_placeholders_as_whole_arguments(
    delivery: Delivery, tmp_path: Path
) -> None:
    record = tmp_path / "record.txt"
    # Each placeholder becomes exactly one argument, so a caption with spaces stays one argument.
    send = command_sender(
        f'sh -c \'printf "%s\\n" "$@" > {record}\' sh {{caption}} {{photo}} {{visit_dir}}'
    )
    send(delivery)
    assert record.read_text().splitlines() == [
        "Pet visit Sun 6:25 AM",
        str(delivery.photo),
        str(delivery.visit_dir),
    ]


def test_command_sender_stages_the_photo_first_when_asked(
    delivery: Delivery, tmp_path: Path
) -> None:
    record = tmp_path / "record.txt"
    stage_dir = tmp_path / "openclaw_media" / "petcam"
    send = command_sender(f"sh -c 'echo \"$1\" > {record}' sh {{photo}}", stage_dir=stage_dir)
    send(delivery)
    staged = stage_dir / "pet_2026-11-01_6-25am.jpg"
    assert record.read_text().strip() == str(staged)
    assert staged.read_bytes() == b"jpegdata"


def test_old_staged_photos_are_cleared_and_other_files_left(
    delivery: Delivery, tmp_path: Path
) -> None:
    stage_dir = tmp_path / "media"
    stage_dir.mkdir()
    old_photo = stage_dir / "pet_2026-10-01_6-25am.jpg"
    old_sheet = stage_dir / "pet_2026-10-01_6-25am_and_3_more.jpg"
    recent_photo = stage_dir / "pet_2026-10-30_11-05pm.jpg"
    someone_elses = stage_dir / "holiday.jpg"
    for path in (old_photo, old_sheet, recent_photo, someone_elses):
        path.write_bytes(b"x")
    long_ago = time.time() - 30 * 86400
    os.utime(old_photo, (long_ago, long_ago))
    os.utime(old_sheet, (long_ago, long_ago))
    os.utime(someone_elses, (long_ago, long_ago))
    command_sender("true {photo}", stage_dir=stage_dir)(delivery)
    assert not old_photo.exists() and not old_sheet.exists()
    assert recent_photo.exists() and someone_elses.exists()


def test_command_sender_passes_doubled_braces_through_as_single_ones(
    delivery: Delivery, tmp_path: Path
) -> None:
    record = tmp_path / "record.txt"
    send = command_sender(f"sh -c 'echo \"$1\" > {record}' sh '{{{{\"text\": 1}}}}'")
    send(delivery)
    assert record.read_text().strip() == '{"text": 1}'


def test_command_sender_rejects_an_unknown_placeholder(delivery: Delivery) -> None:
    with pytest.raises(ValueError, match="channel"):
        command_sender("post {channel} {photo}")


def test_command_sender_raises_with_the_command_output_on_failure(delivery: Delivery) -> None:
    send = command_sender("sh -c 'echo gateway refused >&2; exit 3'")
    with pytest.raises(subprocess.CalledProcessError) as raised:
        send(delivery)
    assert "gateway refused" in raised.value.stderr


def test_command_sender_names_only_the_program_when_the_command_hangs(
    delivery: Delivery, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("collector.deliver.COMMAND_TIMEOUT_SECONDS", 0.2)
    send = command_sender("sh -c 'sleep 5' s3cret")
    with pytest.raises(TimeoutError) as raised:
        send(delivery)
    assert "sh did not finish" in str(raised.value)
    assert "s3cret" not in str(raised.value)


class Received:
    def __init__(self) -> None:
        self.requests: List[dict] = []


@pytest.fixture
def webhook() -> Iterator[tuple]:
    received = Received()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            raw = f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + body
            message = BytesParser(policy=email_policy).parsebytes(raw)
            parts = {}
            for part in message.iter_parts():
                parts[part.get_param("name", header="content-disposition")] = {
                    "filename": part.get_filename(),
                    "content_type": part.get_content_type(),
                    "payload": part.get_payload(decode=True),
                }
            received.requests.append({"path": self.path, "parts": parts})
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/hook", received
    finally:
        server.shutdown()


def test_webhook_sender_posts_caption_visit_and_photo_as_multipart(
    delivery: Delivery, webhook
) -> None:
    url, received = webhook
    webhook_sender(url)(delivery)
    [request] = received.requests
    assert request["path"] == "/hook"
    parts = request["parts"]
    assert parts["caption"]["payload"] == b"Pet visit Sun 6:25 AM"
    assert json.loads(parts["visit"]["payload"]) == {"visit": {"id": "20261101-102524-53"}}
    assert parts["visit"]["content_type"] == "application/json"
    assert parts["photo"]["filename"] == "pet_2026-11-01_6-25am.jpg"
    assert parts["photo"]["content_type"] == "image/jpeg"
    assert parts["photo"]["payload"] == b"jpegdata"
    assert "clip" not in parts


def test_webhook_sender_includes_clips_when_asked(delivery: Delivery, webhook) -> None:
    url, received = webhook
    webhook_sender(url, include_clips=True)(delivery)
    parts = received.requests[0]["parts"]
    assert parts["clip"]["filename"] == "20261101_102528.mp4"
    assert parts["clip"]["content_type"] == "video/mp4"
    assert parts["clip"]["payload"] == b"mp4data"


def test_webhook_sender_raises_on_an_error_response(delivery: Delivery) -> None:
    class Refusing(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_error(500, "nope")

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Refusing)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError):
            webhook_sender(f"http://127.0.0.1:{server.server_port}/")(delivery)
    finally:
        server.shutdown()

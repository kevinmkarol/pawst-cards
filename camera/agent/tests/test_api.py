# ABOUTME: Tests the agent's HTTP control API with a real server on a loopback port.
# ABOUTME: Checks status reporting, toggling tracking, manual gimbal moves, and refused requests.

import http.client
import json
import urllib.error
import urllib.request
from typing import Dict, Iterator, List, Tuple

import pytest

from petcam.api import ControlApi
from petcam.controller import Controller
from petcam.detections import Box, Frame


class FakeGimbal:
    def __init__(self) -> None:
        self.moves: List[Tuple[float, int, float, int]] = []

    def move(
        self, yaw_degrees: float, yaw_speed: int, pitch_degrees: float, pitch_speed: int
    ) -> None:
        self.moves.append((yaw_degrees, yaw_speed, pitch_degrees, pitch_speed))


class FakeRecorder:
    def set_recording(self, enabled: bool) -> None:
        pass


@pytest.fixture
def served() -> Iterator[Tuple[str, Controller, FakeGimbal]]:
    gimbal = FakeGimbal()
    controller = Controller(
        gimbal=gimbal,
        recorder=FakeRecorder(),
        gone_after_seconds=60,
        move_interval_seconds=0.3,
    )
    api = ControlApi(controller=controller, gimbal=gimbal, host="127.0.0.1", port=0)
    api.start()
    try:
        yield f"http://127.0.0.1:{api.port}", controller, gimbal
    finally:
        api.stop()


def get(url: str) -> Tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, json.loads(response.read())


def post(url: str, body: dict) -> Tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def post_with_headers(base: str, path: str, headers: Dict[str, str], body: bytes = b"") -> int:
    """POST with exactly these headers, so a test can claim a body it does not send."""
    connection = http.client.HTTPConnection(base[len("http://") :], timeout=3)
    try:
        connection.putrequest("POST", path)
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders(body)
        return connection.getresponse().status
    finally:
        connection.close()


def test_status_reflects_controller(served) -> None:
    base, controller, _ = served
    controller.observe(
        Frame(
            boxes=[Box(cx=320, cy=320, w=50, h=50, score=0.9, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=1,
        ),
        now=100.0,
    )
    status, body = get(f"{base}/status")
    assert status == 200
    assert body["in_visit"] is True
    assert body["frames_seen"] == 1
    assert len(body["code"]) == 16


def test_tracking_toggle(served) -> None:
    base, controller, _ = served
    status, body = post(f"{base}/tracking", {"enabled": False})
    assert status == 200
    assert body["tracking_enabled"] is False
    assert controller.tracking_enabled is False


def test_manual_move_disables_tracking_and_moves(served) -> None:
    base, controller, gimbal = served
    status, body = post(f"{base}/move", {"yaw": -5, "pitch": 2.5})
    assert status == 200
    assert gimbal.moves == [(-5.0, 90, 2.5, 90)]
    assert controller.tracking_enabled is False
    assert body["tracking_enabled"] is False


def test_bad_requests_are_rejected(served) -> None:
    base, _, gimbal = served
    assert post(f"{base}/move", {"yaw": "left"})[0] == 400
    assert post(f"{base}/tracking", {})[0] == 400
    assert gimbal.moves == []
    try:
        urllib.request.urlopen(f"{base}/nope", timeout=5)
    except urllib.error.HTTPError as error:
        assert error.code == 404
    else:
        raise AssertionError("expected 404")


def test_status_names_the_animals_followed(served) -> None:
    base, _, _ = served
    assert get(f"{base}/status")[1]["pets"] == ["cat", "dog"]


def test_a_body_over_the_size_limit_is_refused_without_being_read(served) -> None:
    # Only the header is sent: the agent must answer from the claimed length alone rather than
    # wait for, or make room for, ten megabytes.
    base, _, _ = served
    assert post_with_headers(base, "/tracking", {"Content-Length": "10000000"}) == 400


@pytest.mark.parametrize("content_length", ["lots", "-1"])
def test_an_unusable_content_length_is_refused(served, content_length: str) -> None:
    base, _, _ = served
    assert post_with_headers(base, "/tracking", {"Content-Length": content_length}) == 400


@pytest.mark.parametrize("degrees", [361, -361, 1e30, float("nan"), float("inf")])
def test_a_move_beyond_a_full_turn_is_refused_and_leaves_tracking_on(served, degrees) -> None:
    base, controller, gimbal = served
    assert post(f"{base}/move", {"yaw": degrees})[0] == 400
    assert post(f"{base}/move", {"pitch": degrees})[0] == 400
    assert gimbal.moves == []
    assert controller.tracking_enabled is True


def test_a_request_sent_by_another_web_page_is_refused(served) -> None:
    # A page open in a browser on the same network can make that browser POST here. Browsers
    # name the page in Origin; the agent's own callers (the collector, curl) send none.
    base, controller, gimbal = served
    host = base[len("http://") :]
    body = json.dumps({"yaw": 5}).encode()
    headers = {"Content-Length": str(len(body)), "Content-Type": "text/plain"}
    foreign = {**headers, "Host": host, "Origin": "http://elsewhere.example"}
    assert post_with_headers(base, "/move", foreign, body) == 403
    assert (
        post_with_headers(base, "/move", {**headers, "Host": host, "Origin": "null"}, body) == 403
    )
    assert gimbal.moves == []
    assert controller.tracking_enabled is True
    own = {**headers, "Host": host, "Origin": base}
    assert post_with_headers(base, "/move", own, body) == 200
    assert len(gimbal.moves) == 1


def test_json_nested_too_deep_to_read_is_a_bad_request(served) -> None:
    base, _, _ = served
    body = b"[" * 4000
    assert post_with_headers(base, "/tracking", {"Content-Length": str(len(body))}, body) == 400

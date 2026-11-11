# ABOUTME: Tests the choice of animals on a running agent: it takes effect on the controller at
# ABOUTME: once, survives a restart, refuses animals the model does not know, and can be undone.

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import pytest

from petcam.api import ControlApi
from petcam.controller import Controller
from petcam.detections import Box, Frame
from petcam.pets import Pets


class Recorder:
    def __init__(self) -> None:
        self.calls = []

    def move(self, *args: object) -> None:
        pass

    def move_to(self, *args: object) -> None:
        pass

    def set_recording(self, enabled: bool) -> None:
        self.calls.append(enabled)


def controller() -> Tuple[Controller, Recorder]:
    recorder = Recorder()
    return (
        Controller(
            gimbal=recorder, recorder=recorder, gone_after_seconds=60, move_interval_seconds=0.3
        ),
        recorder,
    )


def bird() -> Frame:
    box = Box(cx=320, cy=320, w=80, h=80, score=0.8, class_id=14)
    return Frame(boxes=[box], resolution=(640, 640), timestamp_ms=0)


def call(url: str, body: Optional[dict] = None) -> Tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_a_chosen_animal_is_followed_at_once() -> None:
    running, recorder = controller()
    pets = Pets(running)
    assert pets.status()["pets"] == ["cat", "dog"] and not pets.status()["changed"]
    running.observe(bird(), now=100.0)
    assert recorder.calls == []
    pets.set(["bird", "cat"])
    running.observe(bird(), now=101.0)
    assert recorder.calls == [True]
    assert pets.status()["pets"] == ["bird", "cat"] and pets.status()["default"] == ["cat", "dog"]
    assert "giraffe" in pets.status()["known"]


def test_the_choice_survives_a_restart_and_can_be_undone(tmp_path: Path) -> None:
    kept = tmp_path / "pets.json"
    Pets(controller()[0], kept).set(["bird"])
    restarted, _ = controller()
    pets = Pets(restarted, kept)
    assert restarted.pets == (14,) and pets.changed
    pets.reset()
    assert restarted.pets == (15, 16) and not kept.exists()
    assert Pets(controller()[0], kept).status()["pets"] == ["cat", "dog"]


def test_a_spoiled_file_leaves_the_agent_on_what_it_was_started_with(tmp_path: Path) -> None:
    kept = tmp_path / "pets.json"
    for spoiled in ("{", "[]", '["rabbit"]', '{"pets": ["bird"]}'):
        kept.write_text(spoiled)
        running, _ = controller()
        assert not Pets(running, kept).changed and running.pets == (15, 16)


def test_unknown_animals_change_nothing() -> None:
    running, _ = controller()
    pets = Pets(running)
    for wrong in (["rabbit"], [], "cat", None, ["cat", "rabbit"]):
        with pytest.raises(ValueError):
            pets.set(wrong)
    assert running.pets == (15, 16) and not pets.changed


def test_pets_are_read_and_changed_through_the_api(tmp_path: Path) -> None:
    running, _ = controller()
    api = ControlApi(running, Recorder(), "127.0.0.1", 0, pets=Pets(running, tmp_path / "p.json"))
    api.start()
    base = f"http://127.0.0.1:{api.port}"
    try:
        assert call(f"{base}/pets")[1]["pets"] == ["cat", "dog"]
        assert call(f"{base}/pets", {"pets": ["dog", "bird"]})[1]["pets"] == ["dog", "bird"]
        assert call(f"{base}/status")[1]["pets"] == ["dog", "bird"]
        status, answer = call(f"{base}/pets", {"pets": ["rabbit"]})
        assert status == 400 and "rabbit" in answer["error"]
        assert call(f"{base}/pets", {"default": True})[1]["pets"] == ["cat", "dog"]
    finally:
        api.stop()


def test_an_agent_without_the_choice_has_no_such_address() -> None:
    running, _ = controller()
    api = ControlApi(running, Recorder(), "127.0.0.1", 0)
    api.start()
    try:
        assert call(f"http://127.0.0.1:{api.port}/pets")[0] == 404
    finally:
        api.stop()

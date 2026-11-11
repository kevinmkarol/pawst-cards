# ABOUTME: Tests the collector's use of the trained selector: extra frames, both picks recorded,
# ABOUTME: the switch deciding which is delivered, and the built-in scoring as the fallback.

# ruff: noqa: F811  (the camera fixture is imported, then named as a test argument)
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

from petcam.detections import Box, Frame

from collector.deliver import Delivery
from collector.process import SPREAD_FRAMES, collect_visit
from collector.selector import HEURISTIC, MODEL, ModelSelector
from tests.test_process import EASTERN, FIXTURE, camera  # noqa: F401


class FakeSelector(ModelSelector):
    """A selector whose model likes the frame listed last, or fails when told to."""

    def __init__(self, data_dir: Path, fails: bool = False) -> None:
        super().__init__(data_dir)
        self.fails = fails
        # The score of the frame the model likes best.
        self.top = 0.9
        self.asked: List[List[Path]] = []

    def scores(self, frames: Sequence[Path]) -> Optional[List[float]]:
        self.asked.append(list(frames))
        if self.fails or not self.trained():
            return None
        return [self.top * (index + 1) / len(frames) for index in range(len(frames))]


def data_dir(tmp_path: Path, active: str = HEURISTIC, trained: bool = True) -> Path:
    directory = tmp_path / "selector"
    (directory / "models").mkdir(parents=True)
    (directory / "frames").mkdir()
    state = {"active": active}
    if trained:
        state["model"] = "m1"
        (directory / "models" / "m1.npz").write_bytes(b"")
    (directory / "state.json").write_text(json.dumps(state))
    return directory


def one_visit(camera):
    client, controller, clip_dir = camera
    controller.observe(
        Frame(
            boxes=[Box(cx=320, cy=320, w=220, h=210, score=0.95, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=1,
        ),
        now=1001.0,
    )
    shutil.copy(FIXTURE, clip_dir / "20261101_000000.mp4")
    os.utime(clip_dir / "20261101_000000.mp4", (1003.0, 1003.0))
    controller.tick(now=1062.5)
    return client, client.finished_visits()[0]


def collect(client, visit, tmp_path: Path, selector: Optional[ModelSelector]):
    sent: List[Delivery] = []
    result = collect_visit(
        client, visit, tmp_path / "archive", 1, sent.append, EASTERN, selector=selector
    )
    record = json.loads((tmp_path / "archive" / visit.id / "visit.json").read_text())["result"]
    return result, record, sent


def test_without_a_selector_data_directory_nothing_changes(camera, tmp_path: Path) -> None:
    client, visit = one_visit(camera)
    selector = FakeSelector(tmp_path / "never_made")
    result, record, _ = collect(client, visit, tmp_path, selector)
    assert selector.asked == []
    assert result.chosen_by == HEURISTIC and result.frames_considered == 1
    assert [entry["source"] for entry in record["candidates"]] == ["moment"]
    assert record["model_frame"] is None


def test_while_labelling_extra_frames_are_taken_before_any_model_exists(
    camera, tmp_path: Path
) -> None:
    client, visit = one_visit(camera)
    selector = FakeSelector(data_dir(tmp_path, trained=False))
    result, record, _ = collect(client, visit, tmp_path, selector)
    sources = [entry["source"] for entry in record["candidates"]]
    assert sources[0] == "moment" and set(sources[1:]) == {"spread"}
    assert 2 <= len(sources) <= 1 + SPREAD_FRAMES
    assert all(Path(entry["frame"]).is_file() for entry in record["candidates"])
    assert result.chosen_by == HEURISTIC and record["model_frame"] is None


def test_the_model_pick_is_recorded_but_not_posted_while_the_switch_says_heuristic(
    camera, tmp_path: Path
) -> None:
    client, visit = one_visit(camera)
    result, record, sent = collect(client, visit, tmp_path, FakeSelector(data_dir(tmp_path)))
    last = record["candidates"][-1]
    assert record["model_frame"] == last["frame"] != record["heuristic_frame"]
    assert last["model_score"] == max(entry["model_score"] for entry in record["candidates"])
    assert result.chosen_by == HEURISTIC
    assert record["best_frame"] == record["heuristic_frame"]
    assert sent[0].photo.read_bytes() == Path(record["heuristic_frame"]).read_bytes()


def test_the_model_pick_is_posted_when_the_switch_says_model(camera, tmp_path: Path) -> None:
    client, visit = one_visit(camera)
    selector = FakeSelector(data_dir(tmp_path, active=MODEL))
    result, record, sent = collect(client, visit, tmp_path, selector)
    assert result.chosen_by == MODEL
    assert result.frames_considered == len(record["candidates"])
    assert record["best_frame"] == record["model_frame"] != record["heuristic_frame"]
    assert sent[0].photo.read_bytes() == Path(record["model_frame"]).read_bytes()


def test_a_failing_model_leaves_the_choice_to_the_built_in_scoring(camera, tmp_path: Path) -> None:
    client, visit = one_visit(camera)
    selector = FakeSelector(data_dir(tmp_path, active=MODEL), fails=True)
    result, record, sent = collect(client, visit, tmp_path, selector)
    assert result.chosen_by == HEURISTIC
    assert record["best_frame"] == record["heuristic_frame"] and len(sent) == 1


def test_a_model_command_that_cannot_run_gives_no_scores(tmp_path: Path, caplog) -> None:
    selector = ModelSelector(data_dir(tmp_path, active=MODEL), project_dir=tmp_path)
    photo = tmp_path / "frame.jpg"
    photo.write_bytes(b"jpeg")
    assert selector.active() == MODEL and selector.trained()
    assert selector.scores([photo]) is None
    assert "selector model failed" in caplog.text


def test_the_switch_defaults_to_the_built_in_scoring(tmp_path: Path) -> None:
    selector = ModelSelector(tmp_path / "selector")
    assert not selector.in_use() and not selector.trained()
    assert selector.active() == HEURISTIC
    assert selector.scores([tmp_path / "frame.jpg"]) is None


def test_a_visit_the_model_finds_nothing_good_in_is_archived_but_not_posted(
    camera, tmp_path: Path
) -> None:
    client, visit = one_visit(camera)
    selector = FakeSelector(data_dir(tmp_path, active=MODEL))
    selector.top = 0.3
    result, record, sent = collect(client, visit, tmp_path, selector)
    assert sent == [] and not result.posted and result.score == 0.3
    assert record["posted"] is False and record["chosen_by"] == MODEL
    visit_dir = tmp_path / "archive" / visit.id
    assert "not posted" in (visit_dir / "delivered").read_text()
    assert not list(visit_dir.glob("pet_*.jpg"))
    assert client.finished_visits() == []


def test_the_built_in_scoring_posts_every_visit_whatever_the_model_thinks(
    camera, tmp_path: Path
) -> None:
    client, visit = one_visit(camera)
    selector = FakeSelector(data_dir(tmp_path, active=HEURISTIC))
    selector.top = 0.1
    result, record, sent = collect(client, visit, tmp_path, selector)
    assert len(sent) == 1 and result.posted and record["posted"] is True


def test_the_posting_threshold_can_be_set_in_the_state_file(camera, tmp_path: Path) -> None:
    client, visit = one_visit(camera)
    directory = data_dir(tmp_path, active=MODEL)
    state = json.loads((directory / "state.json").read_text())
    (directory / "state.json").write_text(json.dumps({**state, "post_at_least": 0.2}))
    selector = FakeSelector(directory)
    selector.top = 0.3
    assert selector.post_at_least() == 0.2
    result, _, sent = collect(client, visit, tmp_path, selector)
    assert len(sent) == 1 and result.posted

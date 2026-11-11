# ABOUTME: Tests that postcards do not repeat themselves within a day: waiting visits are weighed
# ABOUTME: against the photos already delivered, and get another frame or are held back.

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from collector.deliver import Delivery
from collector.outbox import Outbox
from collector.repeats import RepeatSifter
from collector.selector import HEURISTIC, MODEL, ModelSelector

HOUR = 3600.0


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


class FakeSelector(ModelSelector):
    """A selector for which two photos look alike when their file names start the same way
    ("pot-1.jpg" and "pot-2.jpg"), and which then posts only the first of a kind."""

    def __init__(self, data_dir: Path, active: str = MODEL, fails: bool = False) -> None:
        super().__init__(data_dir)
        self._active = active
        self.fails = fails
        self.asked: List[Tuple[List[List[Tuple[Path, float]]], List[Path]]] = []

    def active(self) -> str:
        return self._active

    def fresh(
        self, visits: Sequence[Sequence[Tuple[Path, float]]], earlier: Sequence[Path]
    ) -> Optional[List[dict]]:
        self.asked.append(([list(visit) for visit in visits], list(earlier)))
        if self.fails:
            return None
        seen = {photo.name.split("-")[0] for photo in earlier}
        picks = []
        for visit in visits:
            new = [(frame, score) for frame, score in visit if frame.name.split("-")[0] not in seen]
            if not new:
                picks.append({"frame": None, "worth": None, "alike": 0.95})
                continue
            frame, score = max(new, key=lambda entry: entry[1])
            seen.add(frame.name.split("-")[0])
            picks.append({"frame": str(frame), "worth": score, "alike": 0.3})
        return picks


def visit(archive: Path, name: str, frames: Dict[str, float], chosen_by: str = MODEL) -> Delivery:
    """An archived visit waiting in the outbox; `frames` maps frame names to model scores and
    the best-scored one is the photo chosen when the visit was collected."""
    visit_dir = archive / name
    (visit_dir / "frames").mkdir(parents=True)
    for shade, frame in enumerate(frames):
        Image.new("RGB", (40, 30), (40 * shade, 90, 90)).save(visit_dir / "frames" / frame)
    best = max(frames, key=lambda frame: frames[frame])
    photo = visit_dir / f"pet_{name}.jpg"
    photo.write_bytes((visit_dir / "frames" / best).read_bytes())
    result = {
        "best_frame": str(visit_dir / "frames" / best),
        "best_frame_at": 100.0,
        "chosen_by": chosen_by,
        "posted": True,
        "score": frames[best],
        "candidates": [
            {"frame": str(visit_dir / "frames" / frame), "at": 100.0 + at, "model_score": score}
            for at, (frame, score) in enumerate(frames.items())
        ],
    }
    (visit_dir / "visit.json").write_text(json.dumps({"visit": {"id": name}, "result": result}))
    (visit_dir / "delivered").write_text("then\n")
    return Delivery(caption=f"Pet visit {name}", photo=photo, visit_dir=visit_dir, clips=[])


def result_of(delivery: Delivery) -> dict:
    return json.loads(delivery.visit_json.read_text())["result"]


def test_a_visit_that_repeats_an_earlier_delivery_is_held_back(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    clock = Clock()
    outbox = Outbox(
        tmp_path, sent.append, HOUR, clock=clock, sifter=RepeatSifter(FakeSelector(tmp_path))
    )
    first = visit(tmp_path, "a", {"pot-1.jpg": 0.9})
    outbox.add(first)
    assert outbox.flush() == 1

    clock.now += HOUR
    again = visit(tmp_path, "b", {"pot-2.jpg": 0.95})
    outbox.add(again)
    assert outbox.flush() == 0
    assert sent == [first]
    assert outbox.waiting() == 0
    assert result_of(again)["posted"] is False
    assert result_of(again)["repeat"]["held_back"] is True
    assert "not posted" in (again.visit_dir / "delivered").read_text()
    assert not again.photo.exists()


def test_a_repeating_visit_posts_another_of_its_frames_when_one_is_new(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    clock = Clock()
    outbox = Outbox(
        tmp_path, sent.append, HOUR, clock=clock, sifter=RepeatSifter(FakeSelector(tmp_path))
    )
    outbox.add(visit(tmp_path, "a", {"pot-1.jpg": 0.9}))
    outbox.flush()

    clock.now += HOUR
    second = visit(tmp_path, "b", {"pot-2.jpg": 0.95, "post-1.jpg": 0.7})
    outbox.add(second)
    assert outbox.flush() == 1
    assert sent[-1] == second
    frame = second.visit_dir / "frames" / "post-1.jpg"
    assert second.photo.read_bytes() == frame.read_bytes()
    result = result_of(second)
    assert result["best_frame"] == str(frame)
    assert result["best_frame_at"] == 101.0
    assert result["score"] == 0.7
    assert result["repeat"]["first_choice"].endswith("pot-2.jpg")
    assert result["posted"] is True


def test_waiting_visits_are_weighed_against_each_other(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    outbox = Outbox(
        tmp_path, sent.append, HOUR, clock=Clock(), sifter=RepeatSifter(FakeSelector(tmp_path))
    )
    kept = visit(tmp_path, "a", {"pot-1.jpg": 0.9})
    dropped = visit(tmp_path, "b", {"pot-2.jpg": 0.8})
    other = visit(tmp_path, "c", {"door-1.jpg": 0.8})
    for delivery in (kept, dropped, other):
        outbox.add(delivery)
    assert outbox.flush() == 2
    assert "Pet visit a" in sent[0].caption and "Pet visit c" in sent[0].caption
    assert "Pet visit b" not in sent[0].caption
    assert sent[0].caption.startswith("2 visits")


def test_a_photo_stops_counting_once_it_is_a_day_old(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    clock = Clock()
    selector = FakeSelector(tmp_path)
    outbox = Outbox(tmp_path, sent.append, HOUR, clock=clock, sifter=RepeatSifter(selector))
    outbox.add(visit(tmp_path, "a", {"pot-1.jpg": 0.9}))
    outbox.flush()
    clock.now += 25 * HOUR
    outbox.add(visit(tmp_path, "b", {"pot-2.jpg": 0.9}))
    assert outbox.flush() == 1
    assert selector.asked[-1][1] == []


def test_delivered_photos_are_remembered_across_a_restart(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    clock = Clock()
    outbox = Outbox(
        tmp_path, sent.append, HOUR, clock=clock, sifter=RepeatSifter(FakeSelector(tmp_path))
    )
    outbox.add(visit(tmp_path, "a", {"pot-1.jpg": 0.9}))
    outbox.flush()

    clock.now += HOUR
    restarted = Outbox(
        tmp_path, sent.append, HOUR, clock=clock, sifter=RepeatSifter(FakeSelector(tmp_path))
    )
    restarted.add(visit(tmp_path, "b", {"pot-2.jpg": 0.9}))
    assert restarted.flush() == 0
    assert len(sent) == 1


def test_visits_picked_by_the_built_in_scoring_are_posted_and_count_as_posted(
    tmp_path: Path,
) -> None:
    sent: List[Delivery] = []
    selector = FakeSelector(tmp_path)
    outbox = Outbox(tmp_path, sent.append, HOUR, clock=Clock(), sifter=RepeatSifter(selector))
    plain = visit(tmp_path, "a", {"pot-1.jpg": 0.9}, chosen_by=HEURISTIC)
    judged = visit(tmp_path, "b", {"pot-2.jpg": 0.9})
    outbox.add(plain)
    outbox.add(judged)
    assert outbox.flush() == 1
    assert sent == [plain]
    assert selector.asked[-1][1] == [plain.visit_dir / "frames" / "pot-1.jpg"]


def test_nothing_is_held_back_while_the_built_in_scoring_is_active(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    selector = FakeSelector(tmp_path, active=HEURISTIC)
    outbox = Outbox(tmp_path, sent.append, HOUR, clock=Clock(), sifter=RepeatSifter(selector))
    outbox.add(visit(tmp_path, "a", {"pot-1.jpg": 0.9}))
    outbox.add(visit(tmp_path, "b", {"pot-2.jpg": 0.9}))
    assert outbox.flush() == 2
    assert selector.asked == []


def test_strictness_zero_in_the_state_file_turns_the_check_off(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(json.dumps({"repeat_strictness": 0}))
    sent: List[Delivery] = []
    selector = FakeSelector(tmp_path)
    outbox = Outbox(tmp_path, sent.append, HOUR, clock=Clock(), sifter=RepeatSifter(selector))
    outbox.add(visit(tmp_path, "a", {"pot-1.jpg": 0.9}))
    outbox.add(visit(tmp_path, "b", {"pot-2.jpg": 0.9}))
    assert outbox.flush() == 2
    assert selector.asked == []


def test_a_selector_that_fails_delivers_every_visit(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    outbox = Outbox(
        tmp_path,
        sent.append,
        HOUR,
        clock=Clock(),
        sifter=RepeatSifter(FakeSelector(tmp_path, fails=True)),
    )
    outbox.add(visit(tmp_path, "a", {"pot-1.jpg": 0.9}))
    outbox.add(visit(tmp_path, "b", {"pot-2.jpg": 0.9}))
    assert outbox.flush() == 2


def test_a_failed_delivery_keeps_the_sifted_visits_and_does_not_remember_them(
    tmp_path: Path,
) -> None:
    sent: List[Delivery] = []
    fail = [True]

    def send(delivery: Delivery) -> None:
        if fail[0]:
            raise RuntimeError("no network")
        sent.append(delivery)

    clock = Clock()
    selector = FakeSelector(tmp_path)
    outbox = Outbox(tmp_path, send, HOUR, clock=clock, sifter=RepeatSifter(selector))
    outbox.add(visit(tmp_path, "a", {"pot-1.jpg": 0.9}))
    try:
        outbox.flush()
    except RuntimeError:
        pass
    assert outbox.waiting() == 1
    fail[0] = False
    clock.now += 600
    assert outbox.flush() == 1
    assert selector.asked[-1][1] == []


def test_the_selector_is_asked_with_scores_and_the_state_files_settings(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(
        json.dumps({"repeat_strictness": 1.5, "repeat_hours": 12, "post_at_least": 0.6})
    )
    selector = ModelSelector(tmp_path)
    assert selector.repeat_strictness() == 1.5
    assert selector.repeat_seconds() == 12 * HOUR
    asked = {}

    def ask(arguments, timeout, request=None):
        asked.update(json.loads(request), arguments=arguments)
        return {"picks": [{"frame": None, "worth": None, "alike": 0.9}]}

    selector._ask = ask
    picks = selector.fresh([[(Path("/a.jpg"), 0.8)]], [Path("/b.jpg")])
    assert picks == [{"frame": None, "worth": None, "alike": 0.9}]
    assert asked == {
        "arguments": ["fresh"],
        "visits": [[{"frame": "/a.jpg", "score": 0.8}]],
        "earlier": ["/b.jpg"],
        "post_at_least": 0.6,
        "strictness": 1.5,
    }

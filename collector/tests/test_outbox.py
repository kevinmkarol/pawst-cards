# ABOUTME: Tests the outbox: visits wait for the delivery interval, leave together as one
# ABOUTME: delivery with their photos in one picture, and survive a restart of the collector.

import json
from pathlib import Path
from typing import List

import pytest
from PIL import Image

from collector.archive import prune_visits
from collector.deliver import Delivery
from collector.outbox import MOST_PHOTOS, RETRY_SECONDS, Outbox, contact_sheet

HOUR = 3600.0
COLOURS = [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0), (0, 200, 200)]


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


def visit(archive: Path, number: int) -> Delivery:
    visit_dir = archive / f"visit-{number:02d}"
    (visit_dir / "clips").mkdir(parents=True)
    photo = visit_dir / f"pet_{number:02d}.jpg"
    Image.new("RGB", (400, 300), COLOURS[number % len(COLOURS)]).save(photo)
    clip = visit_dir / "clips" / f"{number:02d}.mp4"
    clip.write_bytes(b"mp4")
    (visit_dir / "visit.json").write_text(json.dumps({"visit": {"id": f"visit-{number:02d}"}}))
    return Delivery(caption=f"Pet visit {number}", photo=photo, visit_dir=visit_dir, clips=[clip])


def test_the_first_visit_goes_out_at_once_and_unchanged(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    outbox = Outbox(tmp_path, sent.append, HOUR, clock=Clock())
    first = visit(tmp_path, 1)
    outbox.add(first)
    assert outbox.flush() == 1
    assert sent == [first]


def test_visits_inside_the_interval_wait_and_leave_as_one_delivery(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    clock = Clock()
    outbox = Outbox(tmp_path, sent.append, HOUR, clock=clock)
    outbox.add(visit(tmp_path, 1))
    outbox.flush()
    waiting = [visit(tmp_path, number) for number in (2, 3, 4)]
    for delivery in waiting:
        clock.now += 600
        outbox.add(delivery)
        assert outbox.flush() == 0
    assert len(sent) == 1

    clock.now += HOUR - 1800
    assert outbox.flush() == 3
    assert outbox.waiting() == 0
    batch = sent[1]
    assert batch.caption.splitlines() == [
        "3 visits, oldest first:",
        "Pet visit 2",
        "Pet visit 3",
        "Pet visit 4",
    ]
    assert batch.clips == [delivery.clips[0] for delivery in waiting]
    assert batch.photo.parent == batch.visit_dir
    assert json.loads(batch.visit_json.read_text()) == {
        "visits": [{"visit": {"id": f"visit-{number:02d}"}} for number in (2, 3, 4)]
    }
    # Three photos make a two by two grid: 2, 3 on top, 4 below left, nothing below right.
    with Image.open(batch.photo) as sheet:
        assert sheet.size == (2560, 1920)
        for (x, y), colour in {
            (640, 480): COLOURS[2],
            (1920, 480): COLOURS[3],
            (640, 1440): COLOURS[4],
            (1920, 1440): (0, 0, 0),
        }.items():
            pixel = sheet.getpixel((x, y))
            assert all(abs(got - want) < 12 for got, want in zip(pixel, colour))


def test_waiting_visits_and_the_last_delivery_time_survive_a_restart(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    clock = Clock()
    outbox = Outbox(tmp_path, sent.append, HOUR, clock=clock)
    outbox.add(visit(tmp_path, 1))
    outbox.flush()
    outbox.add(visit(tmp_path, 2))
    outbox.add(visit(tmp_path, 3))

    restarted = Outbox(tmp_path, sent.append, HOUR, clock=clock)
    assert restarted.waiting() == 2
    clock.now += HOUR - 1
    assert restarted.flush() == 0
    clock.now += 1
    assert restarted.flush() == 2
    assert len(sent) == 2


def test_a_failed_delivery_keeps_its_visits_and_is_retried_later(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    down = True

    def send(delivery: Delivery) -> None:
        if down:
            raise RuntimeError("gateway refused")
        sent.append(delivery)

    clock = Clock()
    outbox = Outbox(tmp_path, send, HOUR, clock=clock)
    outbox.add(visit(tmp_path, 1))
    outbox.add(visit(tmp_path, 2))
    with pytest.raises(RuntimeError):
        outbox.flush()
    assert outbox.waiting() == 2
    down = False
    assert outbox.flush() == 0
    clock.now += RETRY_SECONDS
    assert outbox.flush() == 2
    assert len(sent) == 1


def test_a_backlog_is_one_delivery_picturing_the_latest_visits(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    outbox = Outbox(tmp_path, sent.append, HOUR, clock=Clock())
    for number in range(MOST_PHOTOS + 3):
        outbox.add(visit(tmp_path, number))
    assert outbox.flush() == MOST_PHOTOS + 3
    [batch] = sent
    lines = batch.caption.splitlines()
    assert (
        lines[0]
        == f"{MOST_PHOTOS + 3} visits; the latest {MOST_PHOTOS} are pictured, oldest first:"
    )
    assert len(lines) == MOST_PHOTOS + 4
    assert batch.photo.name == f"pet_03_and_{MOST_PHOTOS - 1}_more.jpg"


def test_a_waiting_visit_whose_photo_is_gone_is_dropped(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    outbox = Outbox(tmp_path, sent.append, HOUR, clock=Clock())
    gone, kept = visit(tmp_path, 1), visit(tmp_path, 2)
    outbox.add(gone)
    outbox.add(kept)
    gone.photo.unlink()
    assert outbox.flush() == 1
    assert sent == [kept]


def test_a_delivered_batch_is_pruned_from_the_archive_like_a_visit(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    clock = Clock()
    outbox = Outbox(tmp_path, sent.append, HOUR, clock=clock)
    outbox.add(visit(tmp_path, 1))
    outbox.add(visit(tmp_path, 2))
    outbox.flush()
    batch_dir = sent[0].visit_dir
    week_later = batch_dir.joinpath("visit.json").stat().st_mtime + 7 * 86400
    prune_visits(tmp_path, 3, week_later)
    assert not batch_dir.exists()
    assert (tmp_path / "outbox.json").is_file()


def test_contact_sheet_fits_photos_of_another_shape_inside_their_cell(tmp_path: Path) -> None:
    wide, tall = tmp_path / "wide.jpg", tmp_path / "tall.jpg"
    Image.new("RGB", (400, 300), (200, 0, 0)).save(wide)
    Image.new("RGB", (300, 400), (0, 0, 200)).save(tall)
    with Image.open(contact_sheet([wide, tall], tmp_path / "sheet.jpg")) as sheet:
        assert sheet.size == (2560, 960)
        # The tall photo is centred in the right-hand cell with black either side of it.
        assert sheet.getpixel((1290, 480))[2] < 40
        assert sheet.getpixel((1920, 480))[2] > 160


def test_visits_waiting_in_the_outbox_are_not_pruned(tmp_path: Path) -> None:
    def down(delivery: Delivery) -> None:
        raise RuntimeError("gateway refused")

    clock = Clock()
    outbox = Outbox(tmp_path, down, HOUR, clock=clock)
    waiting = visit(tmp_path, 1)
    (waiting.visit_dir / "delivered").write_text("handed to the outbox\n")
    outbox.add(waiting)
    with pytest.raises(RuntimeError):
        outbox.flush()
    week_later = waiting.visit_json.stat().st_mtime + 7 * 86400
    assert prune_visits(tmp_path, 3, week_later, keep=outbox.visit_dirs()) == (0, 0)
    assert waiting.photo.is_file() and outbox.waiting() == 1


def test_a_failed_batch_leaves_nothing_behind_in_the_archive(tmp_path: Path) -> None:
    sent: List[Delivery] = []
    down = True

    def send(delivery: Delivery) -> None:
        if down:
            raise RuntimeError("gateway refused")
        sent.append(delivery)

    clock = Clock()
    outbox = Outbox(tmp_path, send, HOUR, clock=clock)
    outbox.add(visit(tmp_path, 1))
    outbox.add(visit(tmp_path, 2))
    for _ in range(3):
        with pytest.raises(RuntimeError):
            outbox.flush()
        clock.now += RETRY_SECONDS
    assert not list(tmp_path.glob("batch-*"))
    down = False
    assert outbox.flush() == 2
    [batch_dir] = tmp_path.glob("batch-*")
    assert (batch_dir / "delivered").is_file() and sent[0].visit_dir == batch_dir


def test_a_batch_whose_picture_cannot_be_drawn_waits_and_leaves_nothing_behind(
    tmp_path: Path,
) -> None:
    sent: List[Delivery] = []
    clock = Clock()
    outbox = Outbox(tmp_path, sent.append, HOUR, clock=clock)
    outbox.add(visit(tmp_path, 1))
    outbox.add(visit(tmp_path, 2))
    (tmp_path / "visit-02" / "pet_02.jpg").write_bytes(b"cut short")
    with pytest.raises(Exception):
        outbox.flush()
    assert list(tmp_path.glob("batch-*")) == []
    # Not tried again at the next poll, only after the wait.
    clock.now += RETRY_SECONDS - 1
    assert outbox.flush() == 0
    assert sent == []

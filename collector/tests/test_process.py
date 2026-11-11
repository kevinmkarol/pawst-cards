# ABOUTME: End-to-end test of collecting one visit: pull its clips from a real in-process agent API,
# ABOUTME: extract and rank frames from the real recorded fixture, build the caption, and clean up.

import dataclasses
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

import numpy as np
from PIL import Image

from collector.burst import BURST_SECONDS
from collector.camera_client import CameraClient, CameraUnreachable, Visit
from collector.frames import extract_frame_at
from collector.deliver import Delivery
from collector.measures import read_measures
from collector.process import Photographer, archive_dir_for, caption, collect_visit, photo_name
from collector.scoring import Sighting
from tests.test_frames import shorten_video

FIXTURE = Path(__file__).parent / "fixtures" / "visit_clip.mp4"
ONE_FRAME = Path(__file__).parent / "fixtures" / "one_frame_clip.mp4"
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


def assert_is_mirror_of_fixture_frame(image: Path, offset_seconds: float, scratch: Path) -> None:
    plain = extract_frame_at(FIXTURE, offset_seconds, out_dir=scratch)
    with Image.open(image) as got, Image.open(plain.path) as expected:
        flipped_back = np.asarray(got.transpose(Image.Transpose.FLIP_LEFT_RIGHT), dtype=np.int16)
        original = np.asarray(expected, dtype=np.int16)
    assert np.abs(flipped_back - original).mean() < 3


def test_collect_visit_picks_a_frame_and_cleans_up(camera, tmp_path: Path) -> None:
    client, controller, clip_dir = camera
    # The 3 s clip finished being written at t=1003, so it spans about t=999.9 to 1003. A big
    # centered pet is logged at t=1001 and a small off-center one at t=1002; frames are sampled
    # at both moments and the big pet's frame should win.
    controller.observe(
        Frame(
            boxes=[Box(cx=320, cy=320, w=220, h=210, score=0.95, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=1,
        ),
        now=1001.0,
    )
    controller.observe(
        Frame(
            boxes=[Box(cx=40, cy=40, w=40, h=40, score=0.5, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=2,
        ),
        now=1002.0,
    )
    shutil.copy(FIXTURE, clip_dir / "20261101_000000.mp4")
    os.utime(clip_dir / "20261101_000000.mp4", (1003.0, 1003.0))
    controller.tick(now=1062.5)
    visit = client.finished_visits()[0]

    sent: List[Delivery] = []
    archive = tmp_path / "archive"
    result = collect_visit(
        client,
        visit,
        archive_dir=archive,
        every_seconds=1,
        send=sent.append,
        mirror=True,
        zone=EASTERN,
        subject="pet",
    )

    assert result.best_frame.exists() and result.best_frame.suffix == ".jpg"
    assert result.best_frame_at == pytest.approx(1001.0, abs=BURST_SECONDS + 0.1)
    assert_is_mirror_of_fixture_frame(
        result.best_frame, 1001.0 - (1003.0 - 3.072), tmp_path / "plain"
    )
    assert result.frames_considered == 2
    # t=1001 is 7:16 PM Eastern on New Year's Eve 1969; the photo is named for that moment.
    assert sent == [
        Delivery(
            caption=caption(visit, result.frames_considered, EASTERN, "pet"),
            photo=archive / visit.id / "pet_1969-12-31_7-16pm.jpg",
            visit_dir=archive / visit.id,
            clips=[archive / visit.id / "clips" / "20261101_000000.mp4"],
        )
    ]
    assert sent[0].photo.read_bytes() == result.best_frame.read_bytes()
    assert sent[0].clips[0].exists()
    assert sent[0].visit_json.exists()
    assert client.finished_visits() == []
    assert list(clip_dir.iterdir()) == []


def test_collect_visit_keeps_what_it_measured_of_every_candidate(camera, tmp_path: Path) -> None:
    client, controller, clip_dir = camera
    one_visit(controller, clip_dir)
    visit = client.finished_visits()[0]
    archive = tmp_path / "archive"

    collect_visit(client, visit, archive, 1, lambda delivery: None, EASTERN)

    record = json.loads((archive / visit.id / "visit.json").read_text())
    measured = read_measures(archive / visit.id / "frames")
    assert set(measured) == {Path(c["frame"]).name for c in record["result"]["candidates"]}
    taken = next(iter(measured.values()))
    assert taken.sharpness > 0 and taken.frame_change is not None
    # This camera kept no record of where it pointed, and saw the pet only once.
    assert taken.gimbal_turn is None and taken.box_speed is None


def test_a_still_moment_is_worth_more_than_a_larger_one_in_motion() -> None:
    running = [
        Sighting(at=1000.0 + step / 10, cx=100 + 40 * step, cy=320, w=300, h=300, score=0.9)
        for step in range(11)
    ]
    sitting = [
        Sighting(at=1010.0 + step / 10, cx=320, cy=320, w=200, h=200, score=0.9)
        for step in range(11)
    ]
    # The camera turned to follow the running pet and then stayed where it was.
    gaze = [[999.0, 90.0, 10.0], [1000.4, 96.0, 10.0], [1000.8, 104.0, 10.0], [1009.0, 104.0, 10.0]]
    visit = Visit(
        id="v", started_at=1000.0, ended_at=1011.0, sightings=running + sitting, clips=[], gaze=gaze
    )
    photographer = Photographer(visit, recorded=[], frames_dir=Path("unused"), mirror=False)
    assert photographer.worth(sitting[5]) > photographer.worth(running[5])


def test_collect_visit_skips_a_one_frame_clip(camera, tmp_path: Path) -> None:
    """Stopping the recorder just as a new slice opens leaves a clip holding one frame and no
    duration; ffmpeg cannot sample it, so the visit is collected from its other clips."""
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
    shutil.copy(ONE_FRAME, clip_dir / "20261101_000003.mp4")
    os.utime(clip_dir / "20261101_000003.mp4", (1003.1, 1003.1))
    controller.tick(now=1062.5)
    visit = client.finished_visits()[0]
    assert [clip.name for clip in visit.clips] == ["20261101_000000.mp4", "20261101_000003.mp4"]

    archive = tmp_path / "archive"
    result = collect_visit(
        client,
        visit,
        archive_dir=archive,
        every_seconds=1,
        send=lambda delivery: None,
        zone=EASTERN,
    )

    assert result.frames_considered == 1
    assert (archive / visit.id / "clips" / "20261101_000003.mp4").exists()
    assert client.finished_visits() == []
    assert list(clip_dir.iterdir()) == []


def test_collect_visit_uses_the_first_frame_for_a_sighting_just_before_recording_began(
    camera, tmp_path: Path
) -> None:
    """The sightings that open a visit come a few seconds before the recorder's first clip
    starts; they are photographed with that clip's first frame."""
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
    os.utime(clip_dir / "20261101_000000.mp4", (1006.0, 1006.0))
    controller.tick(now=1062.5)
    visit = client.finished_visits()[0]

    result = collect_visit(
        client,
        visit,
        archive_dir=tmp_path / "archive",
        every_seconds=1,
        send=lambda delivery: None,
        mirror=True,
        zone=EASTERN,
    )

    assert result.frames_considered == 1
    assert result.best_frame_at == pytest.approx(1001.0, abs=BURST_SECONDS + 0.1)
    assert_is_mirror_of_fixture_frame(result.best_frame, 0.0, tmp_path / "plain")


def test_collect_visit_counts_a_frame_once_when_several_moments_land_on_it(
    camera, tmp_path: Path
) -> None:
    """Two sightings before the first clip starts are both photographed with its first frame;
    that frame is one candidate, recorded once."""
    client, controller, clip_dir = camera
    for now in (1001.0, 1002.5):
        controller.observe(
            Frame(
                boxes=[Box(cx=320, cy=320, w=220, h=210, score=0.95, class_id=15)],
                resolution=(640, 640),
                timestamp_ms=1,
            ),
            now=now,
        )
    shutil.copy(FIXTURE, clip_dir / "20261101_000000.mp4")
    os.utime(clip_dir / "20261101_000000.mp4", (1006.0, 1006.0))
    controller.tick(now=1062.5)
    visit = client.finished_visits()[0]
    archive = tmp_path / "archive"

    result = collect_visit(
        client,
        visit,
        archive_dir=archive,
        every_seconds=1,
        send=lambda delivery: None,
        zone=EASTERN,
    )

    assert result.frames_considered == 1
    record = json.loads(next(archive.glob("*/visit.json")).read_text())
    assert len(record["result"]["candidates"]) == 1


def test_collect_visit_samples_the_interval_when_no_sighting_falls_in_a_clip(
    camera, tmp_path: Path
) -> None:
    """A sighting whose time lies outside every clip gives no moment to sample, so the frames
    come from the fixed interval instead and sharpness decides."""
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
    os.utime(clip_dir / "20261101_000000.mp4", (1010.0, 1010.0))
    controller.tick(now=1062.5)
    visit = client.finished_visits()[0]

    result = collect_visit(
        client,
        visit,
        archive_dir=tmp_path / "archive",
        every_seconds=1,
        send=lambda delivery: None,
        zone=EASTERN,
    )

    assert result.frames_considered == 3
    assert 1006.9 < result.best_frame_at < 1010.0


def test_collect_visit_uses_the_last_video_frame_for_a_sighting_after_the_video_ends(
    camera, tmp_path: Path
) -> None:
    """Under load the camera drops video frames, so a clip's video can end before its audio
    does; a sighting in that tail is photographed with the last video frame there is."""
    client, controller, clip_dir = camera
    controller.observe(
        Frame(
            boxes=[Box(cx=320, cy=320, w=220, h=210, score=0.95, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=1,
        ),
        now=1002.5,
    )
    shorten_video(FIXTURE, 2.0, clip_dir / "20261101_000000.mp4")
    os.utime(clip_dir / "20261101_000000.mp4", (1003.0, 1003.0))
    controller.tick(now=1062.5)
    visit = client.finished_visits()[0]

    result = collect_visit(
        client,
        visit,
        archive_dir=tmp_path / "archive",
        every_seconds=1,
        send=lambda delivery: None,
        zone=EASTERN,
    )

    assert result.frames_considered == 1
    assert result.best_frame_at == pytest.approx(1002.5, abs=BURST_SECONDS + 0.1)
    with Image.open(result.best_frame) as image:
        assert image.size == (2592, 1944)


def test_caption_describes_the_visit(camera) -> None:
    client, controller, clip_dir = camera
    controller.observe(
        Frame(
            boxes=[Box(cx=320, cy=320, w=50, h=50, score=0.9, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=1,
        ),
        now=1700000000.0,
    )
    shutil.copy(FIXTURE, clip_dir / "clip.mp4")
    os.utime(clip_dir / "clip.mp4", (1700000003.0, 1700000003.0))
    controller.tick(now=1700000000.0 + 185.0)
    visit = client.finished_visits()[0]
    # 1700000000 is Tue 14 Nov 2023 22:13:20 UTC, 5:13 PM in New York.
    text = caption(visit, frames_considered=37, zone=EASTERN, subject="pet")
    assert text == "Pet visit Tue 5:13 PM, about 3 min. Best of 37 frames, 1 sightings."
    assert caption(visit, 37, EASTERN, "McFluff").startswith("McFluff visit Tue")
    assert photo_name(visit, EASTERN, "pet") == "pet_2023-11-14_5-13pm.jpg"
    assert caption(visit, 37, ZoneInfo("America/Los_Angeles"), "pet").startswith(
        "Pet visit Tue 2:13 PM"
    )
    assert caption(visit, 37, EASTERN, "dog").startswith("Dog visit Tue 5:13 PM")
    assert photo_name(visit, EASTERN, "Dog") == "dog_2023-11-14_5-13pm.jpg"


def one_visit(controller: Controller, clip_dir: Path, start: float = 1000.0) -> None:
    controller.observe(
        Frame(
            boxes=[Box(cx=320, cy=320, w=220, h=210, score=0.95, class_id=15)],
            resolution=(640, 640),
            timestamp_ms=1,
        ),
        now=start + 1,
    )
    shutil.copy(FIXTURE, clip_dir / f"clip_{start:.0f}.mp4")
    os.utime(clip_dir / f"clip_{start:.0f}.mp4", (start + 3, start + 3))
    controller.tick(now=start + 62.5)


def test_caption_gives_the_real_time_of_a_visit_recorded_before_the_clock_was_set(
    camera,
) -> None:
    client, controller, clip_dir = camera
    # The camera booted without a network: by its clock the visit was minutes into 1970.
    one_visit(controller, clip_dir, start=1000.0)
    controller.visits.finished()[0].clock_offset = 1700000000.0 - 1001.0
    visit = client.finished_visits()[0]
    assert caption(visit, 5, EASTERN, "pet").startswith("Pet visit Tue 5:13 PM")
    assert photo_name(visit, EASTERN, "pet") == "pet_2023-11-14_5-13pm.jpg"


def test_a_visit_delivered_but_not_cleared_is_cleared_without_a_second_delivery(
    camera, tmp_path: Path
) -> None:
    client, controller, clip_dir = camera
    one_visit(controller, clip_dir)
    visit = client.finished_visits()[0]
    sent: List[Delivery] = []

    class LinkDropsAfterDelivery(CameraClient):
        def delete_visit(self, visit_id: str) -> None:
            raise CameraUnreachable("DELETE: timed out")

    flaky = LinkDropsAfterDelivery(client.base_url)
    with pytest.raises(CameraUnreachable):
        collect_visit(flaky, visit, tmp_path / "archive", 1, sent.append, EASTERN)
    assert len(sent) == 1
    assert len(client.finished_visits()) == 1

    first = sent[0]
    result = collect_visit(client, visit, tmp_path / "archive", 1, sent.append, EASTERN)
    assert sent == [first]
    assert client.finished_visits() == []
    assert result.best_frame.exists()


def test_a_second_visit_with_the_same_id_gets_its_own_archive_directory(
    camera, tmp_path: Path
) -> None:
    client, controller, clip_dir = camera
    archive = tmp_path / "archive"
    sent: List[Delivery] = []
    one_visit(controller, clip_dir)
    first = client.finished_visits()[0]
    collect_visit(client, first, archive, 1, sent.append, EASTERN)
    # The camera booted again without a clock and gave a later visit the same id.
    one_visit(controller, clip_dir, start=5000.0)
    later = dataclasses.replace(client.finished_visits()[0], id=first.id, boot="another-boot")
    assert archive_dir_for(archive, later) == archive / f"{first.id}_2"
    assert archive_dir_for(archive, later) == archive / f"{first.id}_2"
    assert archive_dir_for(archive, first) == archive / first.id


def test_a_subject_with_odd_characters_still_makes_a_plain_file_name() -> None:
    visit = Visit(id="v", started_at=1700000000.0, ended_at=None, sightings=[], clips=[])
    assert photo_name(visit, EASTERN, "cat/dog 100%") == "cat_dog_100_2023-11-14_5-13pm.jpg"

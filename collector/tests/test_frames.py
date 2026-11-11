# ABOUTME: Tests frame extraction and duration reading on real clips recorded by the camera: a 3 s
# ABOUTME: 2592x1944 H264 clip and the one-frame stub the recorder leaves at a slice boundary.

import subprocess
from pathlib import Path

import pytest
from PIL import Image

import numpy as np

from collector.frames import ClipLength, clip_length, extract_frame_at, extract_frames

FIXTURE = Path(__file__).parent / "fixtures" / "visit_clip.mp4"
ONE_FRAME = Path(__file__).parent / "fixtures" / "one_frame_clip.mp4"


def shorten_video(clip: Path, video_seconds: float, out: Path) -> Path:
    """A copy of the clip whose video stops after `video_seconds` while the audio runs on, the
    shape the camera leaves when it drops video frames under load."""
    # The clip is read twice: input 0 is cut off after `video_seconds` and supplies the video,
    # input 1 is read whole and supplies the audio.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        + ["-t", str(video_seconds), "-i", str(clip), "-i", str(clip)]
        + ["-map", "0:v", "-map", "1:a", "-c", "copy", str(out)],
        check=True,
    )
    return out


def test_clip_length_tells_video_and_whole_clip_apart(tmp_path: Path) -> None:
    whole = clip_length(FIXTURE)
    assert whole.total == pytest.approx(3.07, abs=0.1)
    assert whole.video == pytest.approx(3.0, abs=0.1)
    short = clip_length(shorten_video(FIXTURE, 2.0, tmp_path / "short.mp4"))
    assert short.total == pytest.approx(3.07, abs=0.1)
    assert short.video == pytest.approx(2.0, abs=0.1)


def test_one_frame_clip_has_zero_duration() -> None:
    # The recorder writes this when recording stops just as a new slice starts; ffprobe reports no
    # duration for the container, only a zero-length video stream.
    assert clip_length(ONE_FRAME) == ClipLength(total=0.0, video=0.0)


def test_truncated_clip_is_reported_as_unreadable(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(FIXTURE.read_bytes()[:200_000])
    with pytest.raises(subprocess.CalledProcessError):
        clip_length(truncated)


def cut_clip(clip: Path, seconds: float, out: Path) -> Path:
    """The first `seconds` of the clip, the shape of the slice the recorder closes when a visit
    ends."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-t", str(seconds), "-i", str(clip)]
        + ["-c", "copy", str(out)],
        check=True,
    )
    return out


def test_extract_frames_from_a_clip_shorter_than_the_interval(tmp_path: Path) -> None:
    short = cut_clip(FIXTURE, 1.5, tmp_path / "short.mp4")
    frames = extract_frames(short, every_seconds=5, out_dir=tmp_path / "frames")
    assert [f.offset_seconds for f in frames] == [0.0]


def test_extract_one_frame_every_five_seconds(tmp_path: Path) -> None:
    frames = extract_frames(FIXTURE, every_seconds=5, out_dir=tmp_path)
    assert [f.offset_seconds for f in frames] == [0.0]
    with Image.open(frames[0].path) as image:
        assert image.size == (2592, 1944)
        assert image.format == "JPEG"


def test_extract_one_frame_per_second(tmp_path: Path) -> None:
    frames = extract_frames(FIXTURE, every_seconds=1, out_dir=tmp_path)
    assert [f.offset_seconds for f in frames] == [0.0, 1.0, 2.0]
    assert all(f.path.parent == tmp_path and f.path.suffix == ".jpg" for f in frames)


def test_mirrored_extraction_flips_frames_left_to_right(tmp_path: Path) -> None:
    plain = extract_frames(FIXTURE, every_seconds=5, out_dir=tmp_path / "plain", mirror=False)
    mirrored = extract_frames(FIXTURE, every_seconds=5, out_dir=tmp_path / "mirrored", mirror=True)
    with Image.open(plain[0].path) as a, Image.open(mirrored[0].path) as b:
        flipped_back = np.asarray(b.transpose(Image.Transpose.FLIP_LEFT_RIGHT), dtype=np.int16)
        original = np.asarray(a, dtype=np.int16)
    assert flipped_back.shape == original.shape
    # JPEG re-encoding adds small differences; the unflipped mirror image differs far more.
    recoding_noise = np.abs(flipped_back - original).mean()
    mirror_difference = np.abs(np.asarray(b, dtype=np.int16) - original).mean()
    assert recoding_noise < 3
    assert mirror_difference > 3 * recoding_noise


def test_extract_frame_at_an_offset_matches_the_interval_frame_there(tmp_path: Path) -> None:
    frame = extract_frame_at(FIXTURE, offset_seconds=1.0, out_dir=tmp_path / "at")
    assert frame.offset_seconds == 1.0
    interval = extract_frames(FIXTURE, every_seconds=1, out_dir=tmp_path / "every")[1]
    with Image.open(frame.path) as got, Image.open(interval.path) as expected:
        assert got.size == (2592, 1944)
        diff = np.abs(np.asarray(got, dtype=np.int16) - np.asarray(expected, dtype=np.int16))
    assert diff.mean() < 3

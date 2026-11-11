# ABOUTME: Tests photographing a moment from a burst of frames, on the real recorded fixture and on
# ABOUTME: a made clip that is blurred but for one stretch, which is where the photo must come from.

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from collector.burst import BURST_SECONDS, clearest_frame
from collector.frames import extract_burst, extract_frame_at

FIXTURE = Path(__file__).parent / "fixtures" / "visit_clip.mp4"
PICTURE = (640, 640)


def assert_same_picture(one: Path, other: Path) -> None:
    with Image.open(one) as a, Image.open(other) as b:
        difference = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
    assert difference.mean() < 1


def test_a_burst_is_every_frame_of_the_stretch_with_where_it_is_in_the_clip(
    tmp_path: Path,
) -> None:
    frames = extract_burst(FIXTURE, 1.0, 1.0, tmp_path / "burst")
    # The camera records about 14 frames a second.
    assert 13 <= len(frames) <= 16
    offsets = [frame.offset_seconds for frame in frames]
    assert offsets == sorted(offsets) and 1.0 <= offsets[0] < 1.1 and offsets[-1] < 2.1
    third = extract_frame_at(FIXTURE, offsets[2] - 0.002, tmp_path / "at")
    assert_same_picture(frames[2].path, third.path)


def blurred_but_for(clip: Path, clear_from: float, clear_until: float) -> Path:
    """Three seconds of a striped test picture at 15 frames a second, out of focus except
    between the two times."""
    blur = f"gblur=sigma=12:enable='not(between(t,{clear_from},{clear_until}))'"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi"]
        + ["-i", "testsrc2=size=1296x972:rate=15:duration=3", "-vf", blur]
        + ["-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    return clip


def test_the_clearest_frame_near_the_moment_is_the_one_kept(tmp_path: Path) -> None:
    clip = blurred_but_for(tmp_path / "clip.mp4", 1.25, 1.45)
    shot = clearest_frame(clip, 1.0, 3.0, None, PICTURE, tmp_path / "frames")
    assert 1.25 <= shot.frame.offset_seconds <= 1.45
    assert shot.frame.path.parent == tmp_path / "frames"
    assert shot.frame.path.name == f"clip_at{shot.frame.offset_seconds:06.2f}s.jpg"
    assert shot.measures.frame_change is not None
    at_the_moment = clearest_frame(clip, 2.5, 3.0, None, PICTURE, tmp_path / "other")
    assert shot.measures.sharpness > 3 * at_the_moment.measures.sharpness
    # Nothing of the burst is left behind but the frame that was kept.
    assert [path.name for path in (tmp_path / "frames").iterdir()] == [shot.frame.path.name]


def test_a_clear_stretch_further_off_than_the_burst_reaches_is_not_used(tmp_path: Path) -> None:
    clip = blurred_but_for(tmp_path / "clip.mp4", 2.2, 2.6)
    shot = clearest_frame(clip, 1.0, 3.0, None, PICTURE, tmp_path / "frames")
    assert abs(shot.frame.offset_seconds - 1.0) <= BURST_SECONDS + 0.1


def test_a_moment_at_the_very_start_or_end_of_the_video_still_gets_a_frame(
    tmp_path: Path,
) -> None:
    first = clearest_frame(FIXTURE, 0.0, 3.0, None, PICTURE, tmp_path / "frames", mirror=True)
    assert 0.0 <= first.frame.offset_seconds <= BURST_SECONDS + 0.1
    last = clearest_frame(FIXTURE, 2.9, 3.0, None, PICTURE, tmp_path / "frames", mirror=True)
    assert last.frame.offset_seconds >= 2.4
    with Image.open(last.frame.path) as image:
        assert image.size == (2592, 1944)


def test_among_frames_about_as_clear_the_one_at_the_moment_is_kept(tmp_path: Path) -> None:
    # The fixture is a still scene: no frame is clearer than another by enough to leave the moment.
    shot = clearest_frame(FIXTURE, 1.5, 3.0, None, PICTURE, tmp_path / "frames")
    assert abs(shot.frame.offset_seconds - 1.5) < 0.2

# ABOUTME: Tests the measurements of a frame: where the pet's region is, sharpness and change from
# ABOUTME: real synthetic images, the clarity made of them, and the file they are kept in.

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from collector.measures import (
    WHOLE_FRAME,
    Measures,
    change,
    clarity,
    pet_region,
    read_measures,
    region_pixels,
    sharpness,
    write_measures,
)
from collector.scoring import Sighting

FRAME = (2592, 1944)
PICTURE = (640, 640)


def stripes(path: Path, blur: float = 0, shift: int = 0, size=FRAME) -> Path:
    image = Image.new("L", size, 40)
    draw = ImageDraw.Draw(image)
    for x in range(shift, size[0], 80):
        draw.rectangle([x, 0, x + 40, size[1]], fill=220)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    image.save(path, quality=95)
    return path


def test_the_pets_region_allows_for_the_bars_above_and_below_a_wide_frame() -> None:
    # A 4:3 frame fills rows 80 to 560 of the square picture; this box is the middle of it.
    middle = Sighting(at=0.0, cx=320, cy=320, w=200, h=120, score=0.9)
    left, top, right, bottom = pet_region(middle, PICTURE, FRAME, mirror=False)
    assert (left + right) / 2 == pytest.approx(0.5) and (top + bottom) / 2 == pytest.approx(0.5)
    # The box and its margin: 200 of 640 pixels wide, 120 of the frame's 480 rows high.
    assert right - left == pytest.approx(200 / 640 * 1.3)
    assert bottom - top == pytest.approx(120 / 480 * 1.3)
    # A box reaching the top bar is at the top of the frame.
    high = Sighting(at=0.0, cx=320, cy=100, w=100, h=40, score=0.9)
    assert pet_region(high, PICTURE, FRAME, mirror=False)[1] == 0.0


def test_the_pets_region_is_on_the_other_side_of_a_mirrored_frame() -> None:
    on_the_left = Sighting(at=0.0, cx=100, cy=320, w=100, h=100, score=0.9)
    plain = pet_region(on_the_left, PICTURE, FRAME, mirror=False)
    flipped = pet_region(on_the_left, PICTURE, FRAME, mirror=True)
    assert plain[2] < 0.5 < flipped[0]
    assert flipped[0] == pytest.approx(1 - plain[2]) and flipped[2] == pytest.approx(1 - plain[0])


def test_without_a_sighting_the_whole_frame_is_the_region(tmp_path: Path) -> None:
    assert pet_region(None, PICTURE, FRAME, mirror=False) == WHOLE_FRAME
    pixels = region_pixels(stripes(tmp_path / "a.jpg"), None, PICTURE, mirror=False)
    assert pixels.shape == (972, 1296)


def test_a_region_is_measured_finely_enough_to_see_a_slight_blur(tmp_path: Path) -> None:
    pet = Sighting(at=0.0, cx=320, cy=320, w=200, h=200, score=0.9)
    sharp = region_pixels(stripes(tmp_path / "sharp.jpg"), pet, PICTURE, mirror=False)
    soft = region_pixels(stripes(tmp_path / "soft.jpg", blur=3), pet, PICTURE, mirror=False)
    # 260 of 640 pixels of a frame measured 1296 wide.
    assert sharp.shape[1] == pytest.approx(526, abs=2)
    assert sharpness(sharp) > 3 * sharpness(soft) > 0


def test_change_is_small_between_like_frames_and_large_when_the_picture_moved(
    tmp_path: Path,
) -> None:
    still = region_pixels(stripes(tmp_path / "a.jpg"), None, PICTURE, mirror=False)
    same = region_pixels(stripes(tmp_path / "b.jpg"), None, PICTURE, mirror=False)
    moved = region_pixels(stripes(tmp_path / "c.jpg", shift=30), None, PICTURE, mirror=False)
    assert change(still, same) < 0.01
    assert change(still, moved) > 0.3


def test_change_counts_the_same_for_a_dim_picture_as_for_a_bright_one() -> None:
    rng = np.random.default_rng(0)
    bright = rng.uniform(0, 200, (100, 100))
    moved = np.roll(bright, 5, axis=1)
    assert change(bright / 10, moved / 10) == pytest.approx(change(bright, moved))


def test_clarity_is_judged_against_fixed_marks_not_the_other_frames() -> None:
    clear = clarity(Measures(sharpness=400.0, frame_change=0.005))
    soft = clarity(Measures(sharpness=20.0, frame_change=0.005))
    moving = clarity(Measures(sharpness=400.0, frame_change=0.1))
    assert clear > 0.8 and clear > soft > moving
    # A frame taken alone is judged on its sharpness.
    assert clarity(Measures(sharpness=400.0)) == 1.0


def test_measures_are_kept_beside_the_frames_without_what_was_not_measured(
    tmp_path: Path,
) -> None:
    taken = {
        "a.jpg": Measures(sharpness=120.5, frame_change=0.01, gimbal_turn=2.0, box_speed=0.1),
        "b.jpg": Measures(sharpness=80.0),
    }
    write_measures(tmp_path, taken)
    assert read_measures(tmp_path) == taken
    assert '"frame_change"' not in (tmp_path / "measures.json").read_text().split("b.jpg")[1]
    assert read_measures(tmp_path / "nowhere") == {}

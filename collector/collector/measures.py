# ABOUTME: What the collector measures about a candidate frame to tell a clear photo from a
# ABOUTME: blurred one, where in the frame it measures, and the file the measurements are kept in.

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from PIL import Image

from collector.files import write_whole
from collector.scoring import Sighting

# Left, top, right and bottom of a part of a frame, as fractions of its width and height.
Region = Tuple[float, float, float, float]
WHOLE_FRAME: Region = (0.0, 0.0, 1.0, 1.0)

# Kept in a visit's frames directory: frame name -> its measurements. The selector reads it
# (MEASURES_FILE in selector/selector/measures.py).
MEASURES_FILE = "measures.json"
# Frames are measured at about this size, half the camera's 5 MP each way. The video is
# compressed too hard for single pixels to hold detail, and JPEG decodes to half size quickly;
# the 640 pixels blur used to be judged at hid most of it.
MEASURE_SIZE = (1296, 972)
# The pet's region takes in this much more than its box on every side: the box was seen a
# moment before or after the frame, and the pet may have moved since.
REGION_MARGIN = 0.15
# No region is measured smaller than this many pixels a side.
SMALLEST_REGION = 16
# Edge contrast at which a photo is as sharp as it needs to be.
AMPLE_SHARPNESS = 150.0
# A photo is worth half as much when its region differs this much from the neighbouring frames.
CHANGE_HALVES_AT = 0.03
# Grey levels of contrast below which a region is too flat for its changes to be told from noise.
MEASURABLE_CONTRAST = 4.0


@dataclass(frozen=True)
class Measures:
    # Edge contrast inside the pet's region (the whole frame when the pet was not seen then):
    # variance of the Laplacian, higher is crisper. The same number means the same crispness
    # in any visit.
    sharpness: float
    # How much the region differs from the frame before or after, as a share of its contrast;
    # what moves during the exposure moves between frames too. None for a frame taken alone.
    frame_change: Optional[float] = None
    # Degrees the camera turned within half a second either side; None when not recorded.
    gimbal_turn: Optional[float] = None
    # Picture widths a second the pet's box was crossing; None when the pet was not being seen.
    box_speed: Optional[float] = None


def pet_region(
    sighting: Optional[Sighting],
    resolution: Tuple[int, int],
    frame_size: Tuple[int, int],
    mirror: bool,
) -> Region:
    """Where in a frame of `frame_size` the pet of `sighting` is, with a margin; the whole frame
    without a sighting. The camera's detector sees the frame fitted inside a picture of
    `resolution` with bars filling the rest (a 4:3 frame in a square picture has one above and
    one below), in the clip's own left and right: `mirror` says the frame was flipped since."""
    if sighting is None:
        return WHOLE_FRAME
    width, height = resolution
    shrink = min(width / frame_size[0], height / frame_size[1])
    shown_width, shown_height = frame_size[0] * shrink, frame_size[1] * shrink
    bar_x, bar_y = (width - shown_width) / 2, (height - shown_height) / 2
    reach_x = sighting.w * (0.5 + REGION_MARGIN)
    reach_y = sighting.h * (0.5 + REGION_MARGIN)
    left = (sighting.cx - reach_x - bar_x) / shown_width
    right = (sighting.cx + reach_x - bar_x) / shown_width
    top = (sighting.cy - reach_y - bar_y) / shown_height
    bottom = (sighting.cy + reach_y - bar_y) / shown_height
    if mirror:
        left, right = 1.0 - right, 1.0 - left
    region = (max(0.0, left), max(0.0, top), min(1.0, right), min(1.0, bottom))
    # A box that lies outside the frame altogether describes nothing in it.
    return region if region[0] < region[2] and region[1] < region[3] else WHOLE_FRAME


def region_pixels(
    path: Path,
    sighting: Optional[Sighting],
    resolution: Tuple[int, int],
    mirror: bool,
) -> np.ndarray:
    """The grey levels of the pet's region of the frame at `path`, at measuring size."""
    with Image.open(path) as image:
        left, top, right, bottom = pet_region(sighting, resolution, image.size, mirror)
        image.draft("L", MEASURE_SIZE)
        gray = image.convert("L")
        width, height = gray.size
        x, y = int(left * width), int(top * height)
        x, y = min(x, max(0, width - SMALLEST_REGION)), min(y, max(0, height - SMALLEST_REGION))
        box = (
            x,
            y,
            min(width, max(int(right * width), x + SMALLEST_REGION)),
            min(height, max(int(bottom * height), y + SMALLEST_REGION)),
        )
        return np.asarray(gray.crop(box), dtype=np.float64)


def sharpness(pixels: np.ndarray) -> float:
    """Variance of the Laplacian of grey levels; higher means crisper edges."""
    if min(pixels.shape) < 3:
        return 0.0
    laplacian = (
        -4 * pixels[1:-1, 1:-1]
        + pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
    )
    return float(laplacian.var())


def change(pixels: np.ndarray, other: np.ndarray) -> float:
    """How much two frames' regions differ, as a share of the contrast there is in them: 0 for
    the same picture, about 1 for unrelated ones. A dark pet in a dim room moves as far as a
    bright one by this measure."""
    contrast = (pixels.std() + other.std()) / 2
    return float(np.abs(pixels - other).mean() / max(contrast, MEASURABLE_CONTRAST))


def clarity(measures: Measures) -> float:
    """How clear the photo is, about 0.1 to 1: half for being sharp, all of it lost by degrees
    to movement between frames. Measured against fixed marks, so a visit whose frames are all
    blurred has no clear one."""
    sharp = 0.5 + 0.5 * min(1.0, measures.sharpness / AMPLE_SHARPNESS)
    if measures.frame_change is None:
        return sharp
    return sharp / (1.0 + measures.frame_change / CHANGE_HALVES_AT)


def write_measures(frames_dir: Path, measures: Mapping[str, Measures]) -> None:
    """Keep `measures` (frame name -> measurements) beside the frames. Each is written with
    only what was measured; what could not be is left out."""
    record = {
        name: {key: round(value, 5) for key, value in asdict(taken).items() if value is not None}
        for name, taken in measures.items()
    }
    write_whole(frames_dir / MEASURES_FILE, json.dumps(record, indent=1))


def read_measures(frames_dir: Path) -> Dict[str, Measures]:
    """The measurements kept in `frames_dir`; none when there are none or they cannot be read."""
    try:
        record = json.loads((frames_dir / MEASURES_FILE).read_text())
        return {name: Measures(**taken) for name, taken in record.items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}

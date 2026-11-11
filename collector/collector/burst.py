# ABOUTME: Photographs a moment of a clip by looking at every frame within half a second of it
# ABOUTME: and keeping the clearest: sharp on the pet and unlike its neighbours the least.

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from collector.frames import ExtractedFrame, extract_burst, extract_frame_at
from collector.measures import Measures, change, clarity, region_pixels, sharpness
from collector.scoring import Sighting

# Frames this long either side of the moment are looked at: about fifteen at the camera's rate.
# It also takes in the slack between when the detector saw the pet and where that is in the clip.
BURST_SECONDS = 0.5
# What a frame at the far end of the burst loses, of its clarity, to one at the moment itself.
# The moment was chosen for how the pet looked then, and half a second on it may look otherwise,
# so a frame that far off has to be a third clearer to be the one kept.
FAR_FRAME_COST = 0.25


@dataclass(frozen=True)
class Shot:
    frame: ExtractedFrame
    # The frame's sharpness and change from its neighbours; the rest is for the caller to add.
    measures: Measures


def _measured(
    frames: List[ExtractedFrame],
    sighting: Optional[Sighting],
    resolution: Tuple[int, int],
    mirror: bool,
) -> List[Shot]:
    """Each frame of a burst with its measurements. A frame's change is from whichever
    neighbour it is more like: the last frame before something moved is still a clear one."""
    pixels = [region_pixels(frame.path, sighting, resolution, mirror) for frame in frames]
    steps = [change(before, after) for before, after in zip(pixels, pixels[1:])]
    shots = []
    for index, frame in enumerate(frames):
        beside = steps[max(0, index - 1) : index + 1]
        shots.append(
            Shot(frame, Measures(sharpness(pixels[index]), min(beside) if beside else None))
        )
    return shots


def clearest_frame(
    clip: Path,
    offset_seconds: float,
    video_seconds: float,
    sighting: Optional[Sighting],
    resolution: Tuple[int, int],
    out_dir: Path,
    mirror: bool = False,
) -> Shot:
    """The clearest frame within BURST_SECONDS of `offset_seconds` into the clip, saved in
    `out_dir` as a JPEG named for where in the clip it is. Clearness is judged on the pet of
    `sighting`, a box in the detector's picture of `resolution`, or on the whole frame without
    one. A frame counts for a little less the further it is from the moment, so among frames
    about as clear as each other the one nearest the moment is kept."""
    start = max(0.0, offset_seconds - BURST_SECONDS)
    seconds = min(video_seconds, offset_seconds + BURST_SECONDS) - start
    # The whole burst is cut into a scratch directory; only the kept frame joins the others.
    with tempfile.TemporaryDirectory(dir=_made(out_dir), prefix=".burst-") as scratch:
        frames = extract_burst(clip, start, seconds, Path(scratch), mirror) if seconds > 0 else []
        if not frames:
            # Nothing to choose from so close to the clip's end: the frame at the moment itself.
            frame = extract_frame_at(clip, offset_seconds, out_dir, mirror=mirror)
            pixels = region_pixels(frame.path, sighting, resolution, mirror)
            return Shot(frame, Measures(sharpness(pixels)))

        def worth(shot: Shot) -> float:
            away = abs(shot.frame.offset_seconds - offset_seconds) / BURST_SECONDS
            return clarity(shot.measures) * (1.0 - FAR_FRAME_COST * min(1.0, away))

        best = max(_measured(frames, sighting, resolution, mirror), key=worth)
        kept = out_dir / f"{clip.stem}_at{best.frame.offset_seconds:06.2f}s.jpg"
        shutil.move(best.frame.path, kept)
    return Shot(ExtractedFrame(kept, best.frame.offset_seconds), best.measures)


def _made(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory

# ABOUTME: Reads what the collector measured about a frame when it took it (sharpness on the pet,
# ABOUTME: change from the neighbouring frames, how the camera and the pet were moving).

import json
import os
from pathlib import Path
from typing import Dict

# Kept by the collector in a visit's frames directory: frame name -> its measurements
# (MEASURES_FILE in collector/collector/measures.py). The selector keeps the same file beside
# its copies of the frames you rate.
MEASURES_FILE = "measures.json"
# What the collector measures, in the order the model is given them: edge contrast inside the
# pet's region, how much that region differs from the frame before or after, the degrees the
# camera turned within half a second, and the picture widths a second the pet's box crossed.
MEASURED = ("sharpness", "frame_change", "gimbal_turn", "box_speed")


def _read(directory: Path) -> dict:
    try:
        record = json.loads((directory / MEASURES_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def measures_of(frame: Path) -> Dict[str, float]:
    """What was measured of the frame, by name: only what the collector could measure, and
    nothing at all for a frame taken before it measured any."""
    taken = _read(frame.parent).get(frame.name)
    if not isinstance(taken, dict):
        return {}
    return {
        name: float(taken[name])
        for name in MEASURED
        if isinstance(taken.get(name), (int, float)) and not isinstance(taken[name], bool)
    }


def keep_measures(source: Path, kept: Path) -> None:
    """Keep what was measured of the frame at `source` beside the copy of it at `kept`."""
    taken = measures_of(source)
    if not taken or source == kept:
        return
    record = _read(kept.parent)
    if record.get(kept.name) == taken:
        return
    record[kept.name] = taken
    kept.parent.mkdir(parents=True, exist_ok=True)
    scratch = kept.parent / (MEASURES_FILE + ".new")
    scratch.write_text(json.dumps(record, indent=1))
    os.replace(scratch, kept.parent / MEASURES_FILE)

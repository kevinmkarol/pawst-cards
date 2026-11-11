# ABOUTME: Gives visits collected before the selector was in use the same extra frames newer
# ABOUTME: visits get: stills at even steps through the archived clips, cut with ffmpeg.

import json
import subprocess
from pathlib import Path
from typing import List, Tuple

# As many as the collector takes for a new visit (SPREAD_FRAMES in collector/collector/process.py).
SPREAD_FRAMES = 12
# Written in a visit's frames directory once its extra frames have been cut; lists them.
SPREAD_MARKER = ".spread"
TOOL_TIMEOUT_SECONDS = 120
# Seeking closer than this to the end of a clip's video can land past its last frame.
LAST_FRAME_MARGIN_SECONDS = 0.1


def _video_seconds(clip: Path) -> float:
    done = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration",
         "-of", "json", str(clip)],
        capture_output=True, text=True, timeout=TOOL_TIMEOUT_SECONDS,
    )  # fmt: skip
    try:
        return float(json.loads(done.stdout)["streams"][0]["duration"])
    except (ValueError, KeyError, IndexError, TypeError):
        return 0.0


def add_spread_frames(visit_dir: Path, mirror: bool, count: int = SPREAD_FRAMES) -> int:
    """Cut `count` frames at even steps through the visit's archived clips into its frames
    directory, once; returns how many were added. A visit whose clips are gone gets none."""
    frames_dir = visit_dir / "frames"
    marker = frames_dir / SPREAD_MARKER
    if marker.exists() or not frames_dir.is_dir():
        return 0
    clips: List[Tuple[Path, float]] = []
    for clip in sorted((visit_dir / "clips").glob("*.mp4")):
        seconds = _video_seconds(clip)
        if seconds > 0:
            clips.append((clip, seconds))
    total = sum(seconds for _, seconds in clips)
    added: List[str] = []
    for step in range(count if total > 0 else 0):
        position = (step + 0.5) * total / count
        for clip, seconds in clips:
            if position < seconds:
                offset = max(0.0, min(position, seconds - LAST_FRAME_MARGIN_SECONDS))
                out = frames_dir / f"{clip.stem}_at{offset:06.2f}s.jpg"
                if not out.exists():
                    done = subprocess.run(
                        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss",
                         f"{offset:.3f}", "-i", str(clip), "-frames:v", "1",
                         *(["-vf", "hflip"] if mirror else []), "-q:v", "2", str(out)],
                        capture_output=True, timeout=TOOL_TIMEOUT_SECONDS,
                    )  # fmt: skip
                    if done.returncode == 0 and out.exists():
                        added.append(out.name)
                break
            position -= seconds
    marker.write_text(json.dumps(added))
    return len(added)

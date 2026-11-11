# ABOUTME: Pulls still frames out of a recorded clip with ffmpeg as JPEGs: one at a chosen offset,
# ABOUTME: every frame of a short stretch, or one every fixed interval. Reads clip durations too.

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Longest ffmpeg or ffprobe may take on one clip before it is given up on, in seconds.
TOOL_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class ExtractedFrame:
    path: Path
    offset_seconds: float


@dataclass(frozen=True)
class ClipLength:
    # Seconds the clip spans; the audio runs for the whole slice even when video frames were lost.
    total: float
    # Seconds of video there are to seek in; shorter than `total` when the camera dropped frames.
    video: float


def clip_length(clip: Path) -> ClipLength:
    """How long the clip is, and how far into it there is video.

    The container normally carries the total. A clip the recorder closed right after opening it
    holds a single frame and no container duration, only a zero-length video stream, so the stream
    durations are the fallback. A clip with neither cannot be read at all.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,duration",
            "-of",
            "json",
            str(clip),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=TOOL_TIMEOUT_SECONDS,
    )
    probe = json.loads(result.stdout)
    streams = [s for s in probe.get("streams", []) if "duration" in s]
    video = max(
        (float(s["duration"]) for s in streams if s.get("codec_type") == "video"), default=0.0
    )
    if "duration" in probe.get("format", {}):
        return ClipLength(total=float(probe["format"]["duration"]), video=video)
    if streams:
        return ClipLength(total=max(float(s["duration"]) for s in streams), video=video)
    raise RuntimeError(f"{clip} is not a readable clip: {result.stderr.strip() or 'no duration'}")


# A frame's time in a line of the showinfo filter's log.
_SHOWN_FRAME_TIME = re.compile(r"Parsed_showinfo.*\bpts_time:\s*(-?[0-9.]+)")


def _run_ffmpeg(args: List[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=TOOL_TIMEOUT_SECONDS,
    )


def _video_filter(filters: List[str], mirror: bool) -> List[str]:
    """ffmpeg arguments applying `filters`; with `mirror` the frames are also flipped left to
    right, which turns the camera's mirrored picture into a true one."""
    if mirror:
        filters = filters + ["hflip"]
    return ["-vf", ",".join(filters)] if filters else []


def extract_frame_at(
    clip: Path, offset_seconds: float, out_dir: Path, mirror: bool = False
) -> ExtractedFrame:
    """The JPEG of the frame `offset_seconds` into the clip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{clip.stem}_at{offset_seconds:06.2f}s.jpg"
    _run_ffmpeg(
        [
            "-ss",
            f"{offset_seconds:.3f}",
            "-i",
            str(clip),
            "-frames:v",
            "1",
            *_video_filter([], mirror),
            "-q:v",
            "2",
            str(path),
        ]
    )
    return ExtractedFrame(path=path, offset_seconds=offset_seconds)


def extract_burst(
    clip: Path, start_seconds: float, seconds: float, out_dir: Path, mirror: bool = False
) -> List[ExtractedFrame]:
    """Every frame of the `seconds` of the clip from `start_seconds`, in order, each with where
    in the clip it is. The camera drops frames under load, so they are not evenly spaced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{clip.stem}_burst{start_seconds:07.3f}_%04d.jpg"
    # showinfo reports each frame's time, counted from the seek, in ffmpeg's log.
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-y"]
        + ["-ss", f"{start_seconds:.3f}", "-t", f"{seconds:.3f}", "-i", str(clip)]
        + _video_filter(["showinfo"], mirror)
        + ["-fps_mode", "passthrough", "-q:v", "2", str(pattern)],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=TOOL_TIMEOUT_SECONDS,
    )
    times = [float(time) for time in _SHOWN_FRAME_TIME.findall(result.stderr)]
    paths = sorted(out_dir.glob(pattern.name.replace("%04d", "[0-9][0-9][0-9][0-9]")))
    if len(times) != len(paths):
        raise RuntimeError(f"{clip}: {len(paths)} frames cut but {len(times)} times reported")
    return [
        ExtractedFrame(path=path, offset_seconds=start_seconds + time)
        for path, time in zip(paths, times)
    ]


def extract_frames(
    clip: Path, every_seconds: float, out_dir: Path, mirror: bool = False
) -> List[ExtractedFrame]:
    """The first frame of the clip, then the first frame at least `every_seconds` after the
    previous one, each taken to be `every_seconds` after the one before. A clip shorter than
    the interval still yields its first frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{clip.stem}_%04d.jpg"
    keep = f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{every_seconds})"
    _run_ffmpeg(
        [
            "-i",
            str(clip),
            *_video_filter([keep], mirror),
            "-fps_mode",
            "passthrough",
            "-q:v",
            "2",
            str(pattern),
        ]
    )
    frames = sorted(out_dir.glob(f"{clip.stem}_[0-9][0-9][0-9][0-9].jpg"))
    return [
        ExtractedFrame(path=path, offset_seconds=index * float(every_seconds))
        for index, path in enumerate(frames)
    ]

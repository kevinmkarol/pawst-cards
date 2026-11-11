# ABOUTME: Reads the collector's archive: each visit's candidate frames, which one was posted,
# ABOUTME: and which one each selector picked, for the labelling and comparison pages.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from selector.frames import SPREAD_MARKER
from selector.store import Store


@dataclass(frozen=True)
class Frame:
    name: str
    path: Path
    # How the collector came by the frame: "moment" (where the camera saw the pet best),
    # "interval" or "spread" (taken at regular steps through the video), or "" when unknown.
    source: str = ""


@dataclass(frozen=True)
class Visit:
    key: str
    started_at: float
    frames: List[Frame]
    posted: Optional[str]
    heuristic: Optional[str]
    model: Optional[str]
    in_archive: bool
    # True when the collector kept the visit back: the model found no frame worth sending, or
    # its best photo repeated one already sent that day.
    withheld: bool = False
    # True when the collector's record lists the frames it considered.
    candidates_recorded: bool = False

    def paths(self) -> Dict[str, Path]:
        return {frame.name: frame.path for frame in self.frames}


def _name(path: Optional[str]) -> Optional[str]:
    return Path(path).name if path else None


def _cut_later(visit_dir: Path) -> set:
    """Names of the frames the labelling page cut for an older visit."""
    try:
        return set(json.loads((visit_dir / "frames" / SPREAD_MARKER).read_text()))
    except (OSError, ValueError, TypeError):
        return set()


def archived_visit(visit_dir: Path) -> Optional[Visit]:
    """The visit archived in `visit_dir`, or None when the directory is not a collected visit
    (one still being collected, a batch of postcards, anything else)."""
    try:
        record = json.loads((visit_dir / "visit.json").read_text())
        visit, result = record["visit"], record["result"]
        started_at = visit["started_at"] + visit.get("clock_offset", 0.0)
        listed = result.get("candidates")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    frames: List[Frame] = []
    if isinstance(listed, list):
        # Collectors before 2026-11-14 could list one frame several times; it is shown once.
        seen: set = set()
        for entry in listed:
            if not isinstance(entry, dict) or not isinstance(entry.get("frame"), str):
                continue
            path = visit_dir / "frames" / Path(entry["frame"]).name
            if path.is_file() and path.name not in seen:
                seen.add(path.name)
                frames.append(Frame(path.name, path, entry.get("source", "")))
    else:
        # Visits collected before candidates were recorded: every extracted frame was one.
        # Their extra frames, if any, were cut later by the labelling page.
        cut_later = _cut_later(visit_dir)
        frames = [
            Frame(path.name, path, "spread" if path.name in cut_later else "")
            for path in sorted((visit_dir / "frames").glob("*.jpg"))
        ]
    if not frames:
        return None
    posted = _name(result.get("best_frame"))
    return Visit(
        key=visit_dir.name,
        started_at=started_at,
        frames=frames,
        posted=posted,
        heuristic=_name(result.get("heuristic_frame")) or posted,
        model=_name(result.get("model_frame")),
        in_archive=True,
        withheld=result.get("posted") is False,
        candidates_recorded=isinstance(listed, list),
    )


def all_visits(archive_dir: Path, store: Store) -> List[Visit]:
    """Every visit there is something to show for, newest first: those in the archive, and
    labelled ones the archive has since deleted, which keep only their labelled frames."""
    visits: Dict[str, Visit] = {}
    if archive_dir.is_dir():
        for visit_dir in archive_dir.iterdir():
            visit = archived_visit(visit_dir) if visit_dir.is_dir() else None
            if visit is not None:
                visits[visit.key] = visit
    for key, entry in store.labels()["visits"].items():
        if key in visits:
            continue
        frames = [
            Frame(name, store.kept_frame(key, name))
            for name in entry["frames"]
            if store.kept_frame(key, name).is_file()
        ]
        if frames:
            visits[key] = Visit(
                key=key,
                started_at=entry.get("started_at", 0.0),
                frames=frames,
                posted=entry.get("posted"),
                heuristic=entry.get("heuristic") or entry.get("posted"),
                model=None,
                in_archive=False,
            )
    return sorted(visits.values(), key=lambda visit: visit.started_at, reverse=True)

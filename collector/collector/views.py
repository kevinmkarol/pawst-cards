# ABOUTME: Works out where the camera should rest and sweep from what it has seen: a map of the
# ABOUTME: directions around it (time watched, pets found, how good their photos were) and a plan.

import calendar
import math
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from collector.selector import MODEL

# The map divides the directions around the camera into cells this many degrees on a side.
CELL_DEGREES = 10
YAW_CELLS = 360 // CELL_DEGREES
PITCH_CELLS = 180 // CELL_DEGREES
# The camera's field of view, as the agent's tracking uses it.
HORIZONTAL_FOV_DEGREES = 60.0
VERTICAL_FOV_DEGREES = 40.0
# What happened this many days ago counts half as much as what happened today.
HALF_LIFE_DAYS = 7.0
# A model score at which a photo is just worth posting; a visit is worth how far its best photo
# was above this, from 0 to 1.
POSTABLE = 0.5
# What a visit is worth when only the detection arithmetic judged it and it was posted.
POSTED_UNSCORED = 0.5
# A direction the camera has not watched is taken to be an average one, on evidence worth this
# many hours of watching.
PRIOR_HOURS = 0.25
# The camera's home moves only once the visits it has seen are worth this much in all, and only
# to a place that looks this much better than where it rests now.
HOME_EVIDENCE = 3.0
HOME_MARGIN = 1.15
# At full exploration a direction counts for its expected worth plus this many standard
# deviations of doubt.
BOLDEST = 3.0
# Idle seconds before a sweep with no exploration and with all of it.
PATROL_AFTER_SECONDS = (1200.0, 120.0)
# Poses the planner chooses among are this far apart.
YAW_STEP = 15
PITCH_STEP = 10
# Sightings within this long of a visit's first say where the pet was found.
FOUND_WINDOW_SECONDS = 2.0
# A sighting further than this from a photo's moment does not say where the pet was in it.
PHOTO_WINDOW_SECONDS = 1.0
# The camera's agent takes a pose as reached within this many degrees of it.
REACH_TOLERANCE_DEGREES = 5.0

Pose = Tuple[float, float]


@dataclass(frozen=True)
class Limits:
    """The poses the planner may choose, in degrees."""

    yaw_low: float = 0.0
    yaw_high: float = 345.0
    pitch_low: float = 70.0
    pitch_high: float = 140.0

    def poses(self) -> List[Pose]:
        yaws = np.arange(self.yaw_low, self.yaw_high + 0.01, YAW_STEP)
        pitches = np.arange(self.pitch_low, self.pitch_high + 0.01, PITCH_STEP)
        return [(float(yaw), float(pitch)) for yaw in yaws for pitch in pitches]

    def as_dict(self) -> Dict[str, float]:
        return {
            "yaw_low": self.yaw_low,
            "yaw_high": self.yaw_high,
            "pitch_low": self.pitch_low,
            "pitch_high": self.pitch_high,
        }

    def clamp(self, pose: Pose) -> Pose:
        """The pose within the limits that is nearest `pose`."""
        return (
            min(self.yaw_high, max(self.yaw_low, pose[0])),
            min(self.pitch_high, max(self.pitch_low, pose[1])),
        )

    def within_reach(self, missed: Iterable[Tuple[Pose, Pose]]) -> "Limits":
        """These limits, drawn in to what the gimbal has shown it can reach. Each of `missed`
        is a pose the camera was sent to and where it stopped instead: on an axis it stopped
        short on, nothing beyond where it stopped is in reach. A miss that would leave no
        pose at all is not believed."""
        ends = self.as_dict()
        for asked, reached in missed:
            for axis, wanted, got in (
                ("yaw", asked[0], reached[0]),
                ("pitch", asked[1], reached[1]),
            ):
                if abs(wanted - got) <= REACH_TOLERANCE_DEGREES:
                    continue
                low, high = ends[axis + "_low"], ends[axis + "_high"]
                if got < wanted:
                    high = min(high, float(math.floor(got)))
                else:
                    low = max(low, float(math.ceil(got)))
                if low <= high:
                    ends[axis + "_low"], ends[axis + "_high"] = low, high
        return Limits(**ends)

    def seeable(self) -> np.ndarray:
        """Which cells the camera sees from at least one pose within the limits, as a
        yaw-by-pitch grid of booleans."""
        cells = np.zeros((YAW_CELLS, PITCH_CELLS), dtype=bool)
        for pose in self.poses():
            cells |= in_view(pose)
        return cells


@dataclass(frozen=True)
class VisitView:
    """One visit, as the map needs it."""

    key: str
    # When it began, in real time.
    at: float
    # From 0 to 1: how far its best photo was above the posting bar.
    worth: float
    # Which way the pet was when the camera first saw it, and in the best photo.
    found: Pose
    photo_at: Pose
    # The best photo's file and score; the score is None when no model judged it.
    photo: Optional[str] = None
    score: Optional[float] = None


def pose_at(gaze: Sequence[Sequence[float]], at: float) -> Optional[Pose]:
    """Where the camera pointed at time `at`, from a visit's (time, yaw, pitch) track."""
    if not gaze:
        return None
    index = bisect_right([looked[0] for looked in gaze], at)
    if index == 0:
        return (gaze[0][1], gaze[0][2])
    if index == len(gaze):
        return (gaze[-1][1], gaze[-1][2])
    (t0, yaw0, pitch0), (t1, yaw1, pitch1) = gaze[index - 1], gaze[index]
    part = (at - t0) / (t1 - t0) if t1 > t0 else 0.0
    return (yaw0 + (yaw1 - yaw0) * part, pitch0 + (pitch1 - pitch0) * part)


def direction(pose: Pose, cx: float, cy: float, resolution: Tuple[int, int]) -> Pose:
    """Which way something at pixel (cx, cy) of the detection picture lies when the camera
    points at `pose`. The signs are the tracker's: something right of center in the camera's
    own picture is at a greater yaw, something below center at a greater pitch."""
    width, height = resolution
    yaw = math.degrees(
        math.atan((cx - width / 2) * 2 * math.tan(math.radians(HORIZONTAL_FOV_DEGREES / 2)) / width)
    )
    pitch = math.degrees(
        math.atan((cy - height / 2) * 2 * math.tan(math.radians(VERTICAL_FOV_DEGREES / 2)) / height)
    )
    return ((pose[0] + yaw) % 360, min(180.0, max(0.0, pose[1] + pitch)))


def worth_of(result: dict) -> Tuple[float, Optional[float]]:
    """What a visit was worth, 0 to 1, and its best model score if a model judged it."""
    scores = [
        candidate["model_score"]
        for candidate in result.get("candidates") or []
        if candidate.get("model_score") is not None
    ]
    if not scores and result.get("chosen_by") == MODEL and result.get("score") is not None:
        scores = [result["score"]]
    if scores:
        best = float(max(scores))
        return (min(1.0, max(0.0, (best - POSTABLE) / (1 - POSTABLE))), best)
    return (POSTED_UNSCORED if result.get("posted") else 0.0, None)


def view_of(key: str, record: dict, resolution: Tuple[int, int]) -> Optional[VisitView]:
    """The archived visit `record` (the contents of its visit.json) as the map needs it; None
    when the camera did not yet record where it pointed, or saw nothing."""
    visit, result = record.get("visit") or {}, record.get("result") or {}
    gaze, sightings = visit.get("gaze") or [], visit.get("sightings") or []
    if not gaze or not sightings:
        return None

    def where(sighting: dict) -> Pose:
        return direction(pose_at(gaze, sighting["at"]), sighting["cx"], sighting["cy"], resolution)

    first = [s for s in sightings if s["at"] - sightings[0]["at"] <= FOUND_WINDOW_SECONDS]
    found = where(first[len(first) // 2])
    photo_at = found
    best_at = result.get("best_frame_at")
    if best_at is not None:
        nearest = min(sightings, key=lambda s: abs(s["at"] - best_at))
        if abs(nearest["at"] - best_at) <= PHOTO_WINDOW_SECONDS:
            photo_at = where(nearest)
        else:
            looked = pose_at(gaze, best_at)
            photo_at = (looked[0] % 360, looked[1])
    worth, score = worth_of(result)
    return VisitView(
        key=key,
        at=visit["started_at"] + visit.get("clock_offset", 0.0),
        worth=worth,
        found=found,
        photo_at=photo_at,
        photo=result.get("best_frame"),
        score=score,
    )


def cell_of(where: Pose) -> Tuple[int, int]:
    return (
        int(where[0] % 360 // CELL_DEGREES) % YAW_CELLS,
        min(PITCH_CELLS - 1, max(0, int(where[1] // CELL_DEGREES))),
    )


def cell_center(cell: Tuple[int, int]) -> Pose:
    return ((cell[0] + 0.5) * CELL_DEGREES, (cell[1] + 0.5) * CELL_DEGREES)


_CENTER_YAWS = (np.arange(YAW_CELLS) + 0.5) * CELL_DEGREES
_CENTER_PITCHES = (np.arange(PITCH_CELLS) + 0.5) * CELL_DEGREES


def in_view(pose: Pose) -> np.ndarray:
    """Which cells the camera sees when it points at `pose`, as a yaw-by-pitch grid of
    booleans."""
    around = np.abs((_CENTER_YAWS - pose[0] + 180) % 360 - 180) < HORIZONTAL_FOV_DEGREES / 2
    up_down = np.abs(_CENTER_PITCHES - pose[1]) < VERTICAL_FOV_DEGREES / 2
    return np.outer(around, up_down)


def view_weight(pose: Pose) -> np.ndarray:
    """How well the camera sees each cell from `pose`: 1 in the middle of the picture, falling
    to a little over half at its edges, 0 outside it. A pet in the middle of the picture is
    detected sooner and photographed whole."""
    around = np.abs((_CENTER_YAWS - pose[0] + 180) % 360 - 180) / (HORIZONTAL_FOV_DEGREES / 2)
    up_down = np.abs(_CENTER_PITCHES - pose[1]) / (VERTICAL_FOV_DEGREES / 2)
    off_center = np.maximum.outer(around, up_down)
    return np.where(off_center < 1, 1 - 0.4 * off_center, 0.0)


def hour_start(name: str) -> Optional[float]:
    """When the hour the camera calls `name` (YYYYMMDDHH, UTC) began; None if it is no hour."""
    try:
        return float(calendar.timegm(time.strptime(name, "%Y%m%d%H")))
    except ValueError:
        return None


def _grid() -> np.ndarray:
    return np.zeros((YAW_CELLS, PITCH_CELLS))


@dataclass
class ViewMap:
    """What is known about each direction around the camera. Every grid is yaw by pitch cells;
    older happenings count for less in all of them."""

    # Hours the camera watched the cell while waiting for a pet.
    watched: np.ndarray = field(default_factory=_grid)
    # Worth of the visits that began with the pet in the cell.
    found: np.ndarray = field(default_factory=_grid)
    # Worth of the visits whose best photo had the pet in the cell.
    photos: np.ndarray = field(default_factory=_grid)
    # The best photo taken with the pet in each cell.
    best: Dict[Tuple[int, int], VisitView] = field(default_factory=dict)
    visits: int = 0

    def expected(self) -> Tuple[np.ndarray, np.ndarray]:
        """For each cell, the worth of visits expected per hour of watching it, and how unsure
        that is (a standard deviation). A cell watched little is taken to be near the average
        of all cells, and is the most unsure."""
        watched = float(self.watched.sum())
        average = max(float(self.found.sum()) / watched if watched else 0.0, 1e-3)
        evidence = self.found + PRIOR_HOURS * average
        hours = self.watched + PRIOR_HOURS
        return evidence / hours, np.sqrt(evidence) / hours


def build_map(
    visits: Iterable[VisitView],
    gaze_hours: Dict[str, Dict[str, float]],
    now: float,
    since: Optional[float] = None,
) -> ViewMap:
    """The map as of `now`, from the visits and from the camera's record of where it looked
    (hour -> "yaw,pitch" -> seconds). Anything before `since` is left out: the camera stood
    somewhere else then."""

    def weight(at: float) -> float:
        return 0.5 ** (max(0.0, now - at) / 86400 / HALF_LIFE_DAYS)

    view_map = ViewMap()
    for name, looked in gaze_hours.items():
        began = hour_start(name)
        if began is None or began > now or (since is not None and began < since):
            continue
        for where, seconds in looked.items():
            try:
                yaw, pitch = (float(part) for part in where.split(","))
            except ValueError:
                continue
            view_map.watched[in_view((yaw, pitch))] += seconds / 3600 * weight(began + 1800)
    for visit in visits:
        if since is not None and visit.at < since:
            continue
        view_map.visits += 1
        worth = visit.worth * weight(visit.at)
        view_map.found[cell_of(visit.found)] += worth
        cell = cell_of(visit.photo_at)
        view_map.photos[cell] += worth
        if visit.photo is not None and visit.worth > 0:
            held = view_map.best.get(cell)
            if held is None or visit.worth > held.worth:
                view_map.best[cell] = visit
    return view_map


def boldness(explore: float) -> float:
    return BOLDEST * min(1.0, max(0.0, explore))


def patrol_after(explore: float) -> float:
    calm, restless = PATROL_AFTER_SECONDS
    return round(calm * (restless / calm) ** min(1.0, max(0.0, explore)))


def make_plan(
    view_map: ViewMap,
    explore: float,
    home_now: Pose,
    stops: int,
    limits: Limits = Limits(),
) -> dict:
    """Where the camera should rest and sweep, in the form the camera's agent takes.

    Home is the pose whose view has brought the most photo worth per hour, counting only what
    is fairly sure; it stays at `home_now`, brought within the limits if it lies outside
    them, until there is evidence of a clearly better place.
    The stops are chosen one at a time, each the pose that adds the most to what the home and
    the stops before it already see, where a cell counts for its expected worth plus a bonus
    for doubt that grows with `explore` (0 to 1). With no exploration the stops go where pets
    have turned up; with all of it they go where the camera has hardly looked. The more
    exploration, the sooner an idle camera sets off on a sweep."""
    mean, doubt = view_map.expected()
    poses = limits.poses()
    sure = np.maximum(0.0, mean - doubt)

    def home_worth(pose: Pose) -> float:
        return float((sure * view_weight(pose)).sum())

    home = limits.clamp(home_now)
    if float(view_map.found.sum()) >= HOME_EVIDENCE and poses:
        best = max(poses, key=home_worth)
        if home_worth(best) > HOME_MARGIN * home_worth(home):
            home = best

    hopeful = mean + boldness(explore) * doubt
    seen = in_view(home)
    chosen: List[Pose] = []
    remaining = [pose for pose in poses if pose != home]
    while remaining and len(chosen) < stops:
        gains = [float(hopeful[in_view(pose) & ~seen].sum()) for pose in remaining]
        index = int(np.argmax(gains))
        if gains[index] <= 0:
            break
        pose = remaining.pop(index)
        chosen.append(pose)
        seen |= in_view(pose)
    # One pass around rather than back and forth: the yaw motor is the slow one.
    chosen.sort()
    return {
        "home": {"yaw": home[0], "pitch": home[1]},
        "stops": [{"yaw": yaw, "pitch": pitch} for yaw, pitch in chosen],
        "patrol_after": patrol_after(explore),
    }


def describe(view_map: ViewMap) -> List[dict]:
    """The map's cells that have anything in them, as JSON-ready values for the page that
    draws it."""
    mean, doubt = view_map.expected()
    cells = []
    for yaw_cell in range(YAW_CELLS):
        for pitch_cell in range(PITCH_CELLS):
            cell = (yaw_cell, pitch_cell)
            watched = float(view_map.watched[cell])
            found = float(view_map.found[cell])
            photos = float(view_map.photos[cell])
            if not (watched or found or photos):
                continue
            yaw, pitch = cell_center(cell)
            entry = {
                "yaw": yaw,
                "pitch": pitch,
                "watched": round(watched, 4),
                "found": round(found, 4),
                "photos": round(photos, 4),
                "expected": round(float(mean[cell]), 4),
                "doubt": round(float(doubt[cell]), 4),
            }
            best = view_map.best.get(cell)
            if best is not None:
                entry["best"] = {
                    "visit": best.key,
                    "frame": best.photo.rsplit("/", 1)[-1],
                    "score": best.score,
                    "at": best.at,
                }
            cells.append(entry)
    return cells

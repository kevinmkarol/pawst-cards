# ABOUTME: Scores candidate photos of a visit by how large and centered the pet was in the camera's
# ABOUTME: detections at that moment and how clear the photo is, and picks the moments to photograph.

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple

# Sightings further than this from a frame's timestamp do not describe that frame.
SIGHTING_WINDOW_SECONDS = 1.0


@dataclass(frozen=True)
class Sighting:
    at: float
    cx: float
    cy: float
    w: float
    h: float
    score: float
    class_id: int = -1


@dataclass(frozen=True)
class Candidate:
    path: Path
    at: float

    @staticmethod
    def nearest_sighting(at: float, sightings: List[Sighting], window: float) -> Optional[Sighting]:
        close = [s for s in sightings if abs(s.at - at) <= window]
        return min(close, key=lambda s: abs(s.at - at)) if close else None


def pet_prominence(sighting: Optional[Sighting], resolution: Tuple[int, int]) -> float:
    """0 when there is no pet; otherwise the pet's share of the frame, discounted by distance
    from center."""
    if sighting is None:
        return 0.0
    width, height = resolution
    area_share = (sighting.w * sighting.h) / (width * height)
    offset = ((sighting.cx - width / 2) / (width / 2)) ** 2 + (
        (sighting.cy - height / 2) / (height / 2)
    ) ** 2
    centeredness = max(0.0, 1.0 - offset**0.5)
    return area_share * (0.5 + 0.5 * centeredness) * sighting.score


def best_moments(
    sightings: List[Sighting],
    resolution: Tuple[int, int],
    limit: int,
    spacing_seconds: float,
    worth: Optional[Callable[[Sighting], float]] = None,
) -> List[Sighting]:
    """The sightings worth most, at most `limit` of them and at least `spacing_seconds` apart.
    A sighting is worth what `worth` says, or the pet's prominence without it.

    Detections arrive many times a second, so without the spacing the picks would all be
    near-identical frames from the same second.
    """
    if worth is None:

        def worth(sighting: Sighting) -> float:
            return pet_prominence(sighting, resolution)

    chosen: List[Sighting] = []
    for sighting in sorted(sightings, key=worth, reverse=True):
        if len(chosen) == limit:
            break
        if all(abs(sighting.at - taken.at) >= spacing_seconds for taken in chosen):
            chosen.append(sighting)
    return sorted(chosen, key=lambda s: s.at)


def rank(
    candidates: List[Candidate],
    sightings: List[Sighting],
    resolution: Tuple[int, int],
    clarity: Mapping[Path, float],
) -> List[Candidate]:
    """Best photo first. `clarity` says how clear each candidate's frame is, 0 to 1.

    Any frame with a pet logged at its moment beats every frame without one. Among pet frames the
    pet's prominence decides, marked down for want of clarity; among the rest clarity decides.
    """

    def score(candidate: Candidate) -> Tuple[bool, float]:
        nearest = Candidate.nearest_sighting(candidate.at, sightings, SIGHTING_WINDOW_SECONDS)
        if nearest is None:
            return (False, clarity[candidate.path])
        return (True, pet_prominence(nearest, resolution) * clarity[candidate.path])

    return sorted(candidates, key=score, reverse=True)

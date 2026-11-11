# ABOUTME: Collects one finished visit: downloads its clips from the camera, extracts frames at the
# ABOUTME: best moments, ranks them, delivers the winner, archives the visit, and clears the camera.

import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from collector.camera_client import CameraClient, Visit
from collector.deliver import Delivery, Sender
from collector.files import write_whole
from collector.burst import clearest_frame
from collector.frames import clip_length, extract_frames
from collector.repeats import DELIVERED_FILE
from collector.measures import Measures, clarity, region_pixels, sharpness, write_measures
from collector.motion import VisitMotion
from collector.scoring import (
    SIGHTING_WINDOW_SECONDS,
    Candidate,
    Sighting,
    best_moments,
    pet_prominence,
    rank,
)
from collector.selector import HEURISTIC, MODEL, ModelSelector

# Resolution of the camera's detection model; sightings are expressed in these pixels.
DETECTION_RESOLUTION = (640, 640)
# How many moments of a visit are photographed, and how far apart they must be.
MOMENTS_PER_VISIT = 8
MOMENT_SPACING_SECONDS = 1.0
# Extra frames taken at even steps through a visit's video when the trained selector is in
# use: more for it to choose from, and more for its owner to label.
SPREAD_FRAMES = 12
# Seeking closer than this to the end of a clip's video can land past its last frame.
LAST_FRAME_MARGIN_SECONDS = 0.1
# The recorder's first clip starts this long, at most, after the sightings that opened the visit.
RECORDING_LEAD_SECONDS = 4.0

# Kept in a visit's archive directory: which visit the directory holds.
SOURCE_FILE = "source.json"

log = logging.getLogger("collector")


@dataclass(frozen=True)
class CollectResult:
    best_frame: Path
    best_frame_at: float
    frames_considered: int
    # Which selector picked the frame: the built-in scoring or the trained model.
    chosen_by: str = HEURISTIC
    # False when the trained model found no frame worth sending, so nothing was delivered.
    posted: bool = True
    # The model's score for the chosen frame, when the model scored the visit.
    score: Optional[float] = None


@dataclass(frozen=True)
class RecordedClip:
    path: Path
    start: float
    duration: float
    video_duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    def covers(self, at: float) -> bool:
        """Whether this clip is the one to photograph the moment `at` from. The sightings that
        open a visit come a little before the recorder's first clip starts, so a moment shortly
        before the clip counts as well."""
        return self.start - RECORDING_LEAD_SECONDS <= at < self.end

    def offset_of(self, at: float) -> float:
        """Where to seek for the moment `at`. A moment before the clip gets its first frame. The
        camera drops video frames under load, so the video can end before the clip does; a
        moment in that tail gets the last frame there is."""
        return max(0.0, min(at - self.start, self.video_duration - LAST_FRAME_MARGIN_SECONDS))


def started_at(visit: Visit, zone: ZoneInfo) -> datetime:
    """When the visit really started; the camera's own time for it is wrong when the visit was
    recorded before the camera's clock was set."""
    return datetime.fromtimestamp(visit.real_started_at, zone)


def _source(visit: Visit) -> dict:
    """What tells one visit from another that the camera gave the same id."""
    return {"started_at": visit.started_at, "ended_at": visit.ended_at, "boot": visit.boot}


def _holds(directory: Path, visit: Visit) -> bool:
    """Whether an existing archive directory belongs to `visit`. One written before sources
    were recorded is judged by its visit record, and one with neither is taken to be an earlier
    attempt at this visit."""
    try:
        if (directory / SOURCE_FILE).is_file():
            return json.loads((directory / SOURCE_FILE).read_text()) == _source(visit)
        if (directory / "visit.json").is_file():
            recorded = json.loads((directory / "visit.json").read_text())["visit"]
            return (recorded["started_at"], recorded["ended_at"]) == (
                visit.started_at,
                visit.ended_at,
            )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def archive_dir_for(archive_dir: Path, visit: Visit) -> Path:
    """The directory this visit is archived in, created if need be. Visit ids are made from the
    camera's clock, which starts over when the camera boots without a network, so two visits
    can share an id; the later one gets a numbered directory of its own."""
    number = 1
    while True:
        directory = archive_dir / (visit.id if number == 1 else f"{visit.id}_{number}")
        if not directory.exists():
            directory.mkdir(parents=True)
            write_whole(directory / SOURCE_FILE, json.dumps(_source(visit)))
            return directory
        if _holds(directory, visit):
            return directory
        number += 1


def _delivered_before(visit_dir: Path) -> Optional[CollectResult]:
    """The result of an earlier collection of this visit that got as far as delivering."""
    if not (visit_dir / DELIVERED_FILE).is_file():
        return None
    try:
        result = json.loads((visit_dir / "visit.json").read_text())["result"]
        return CollectResult(
            best_frame=Path(result["best_frame"]),
            best_frame_at=result["best_frame_at"],
            frames_considered=result["frames_considered"],
            chosen_by=result.get("chosen_by", HEURISTIC),
            posted=result.get("posted", True),
            score=result.get("score"),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def caption(visit: Visit, frames_considered: int, zone: ZoneInfo, subject: str) -> str:
    """The delivered text: who visited, when the visit started on a 12-hour clock in `zone`, and
    how long it was."""
    started = started_at(visit, zone).strftime("%a %-I:%M %p")
    minutes = max(1, round(((visit.ended_at or visit.started_at) - visit.started_at) / 60))
    return (
        f"{subject[:1].upper()}{subject[1:]} visit {started}, about {minutes} min. "
        f"Best of {frames_considered} frames, {len(visit.sightings)} sightings."
    )


def photo_name(visit: Visit, zone: ZoneInfo, subject: str) -> str:
    """File name for the delivered photo, readable as the visit's start in `zone`."""
    # The subject is whatever the owner typed; only letters, digits, underscores and dashes go in a name.
    plain = re.sub(r"[^\w-]+", "_", subject).strip("_") or "pet"
    return f"{plain}_{started_at(visit, zone).strftime('%Y-%m-%d_%-I-%M%p')}.jpg".lower()


class Photographer:
    """Takes a visit's candidate photos and measures each one: for a moment, the clearest frame
    within half a second of it, with how sharp it is, how unlike its neighbours, and what the
    visit's record says the camera and the pet were doing then."""

    def __init__(
        self, visit: Visit, recorded: List[RecordedClip], frames_dir: Path, mirror: bool
    ) -> None:
        self.sightings = visit.sightings
        self.motion = VisitMotion(visit.sightings, visit.gaze, DETECTION_RESOLUTION)
        self.recorded = recorded
        self.frames_dir = frames_dir
        self.mirror = mirror
        # Every frame taken so far and what was measured of it.
        self.measures: Dict[Path, Measures] = {}

    def worth(self, sighting: Sighting) -> float:
        """What a sighting is worth as a moment to photograph: the pet large and central, and
        neither it nor the camera moving."""
        return pet_prominence(sighting, DETECTION_RESOLUTION) * self.motion.stillness(sighting.at)

    def _pet_at(self, at: float) -> Optional[Sighting]:
        return Candidate.nearest_sighting(at, self.sightings, SIGHTING_WINDOW_SECONDS)

    def _moving(self, measures: Measures, at: float) -> Measures:
        return Measures(
            sharpness=measures.sharpness,
            frame_change=measures.frame_change,
            gimbal_turn=self.motion.gimbal_turn(at),
            box_speed=self.motion.box_speed(at),
        )

    def photograph(self, clip: RecordedClip, at: float) -> Candidate:
        """The clearest frame around the moment `at`, from `clip`."""
        wanted = clip.offset_of(at)
        shot = clearest_frame(
            clip.path,
            wanted,
            clip.video_duration,
            self._pet_at(at),
            DETECTION_RESOLUTION,
            self.frames_dir,
            mirror=self.mirror,
        )
        # The frame stands for the moment asked for, moved by as far as the clearest frame was
        # from it. For a moment inside the clip that is the frame's own time.
        taken_at = at + (shot.frame.offset_seconds - wanted)
        self.measures[shot.frame.path] = self._moving(shot.measures, taken_at)
        return Candidate(path=shot.frame.path, at=taken_at)

    def moments(self, limit: int, spacing_seconds: float) -> List[Candidate]:
        """A photo for each of the visit's best moments that falls inside a clip."""
        candidates: List[Candidate] = []
        for moment in best_moments(
            self.sightings, DETECTION_RESOLUTION, limit, spacing_seconds, worth=self.worth
        ):
            clip = next((clip for clip in self.recorded if clip.covers(moment.at)), None)
            if clip is not None:
                candidates.append(self.photograph(clip, moment.at))
        return candidates

    def spread(self, count: int) -> List[Candidate]:
        """`count` photos at even steps through the visit's video, whatever the camera saw then."""
        total = sum(clip.video_duration for clip in self.recorded)
        candidates: List[Candidate] = []
        if total <= 0:
            return candidates
        for step in range(count):
            position = (step + 0.5) * total / count
            for clip in self.recorded:
                if position < clip.video_duration:
                    candidates.append(self.photograph(clip, clip.start + position))
                    break
                position -= clip.video_duration
        return candidates

    def every(self, seconds: float) -> List[Candidate]:
        """A photo every `seconds` of every clip, each the frame at that instant."""
        candidates: List[Candidate] = []
        for clip in self.recorded:
            for frame in extract_frames(
                clip.path, every_seconds=seconds, out_dir=self.frames_dir, mirror=self.mirror
            ):
                at = clip.start + frame.offset_seconds
                pixels = region_pixels(
                    frame.path, self._pet_at(at), DETECTION_RESOLUTION, self.mirror
                )
                self.measures[frame.path] = self._moving(Measures(sharpness(pixels)), at)
                candidates.append(Candidate(path=frame.path, at=at))
        return candidates


def collect_visit(
    client: CameraClient,
    visit: Visit,
    archive_dir: Path,
    every_seconds: float,
    send: Sender,
    zone: ZoneInfo,
    mirror: bool = False,
    subject: str = "pet",
    selector: Optional[ModelSelector] = None,
) -> CollectResult:
    """Fetch a visit's clips, pick the best frame, deliver it, and clear the visit off the camera.

    Frames are taken around the moments the camera saw the pet best while it and the camera
    were still, each the clearest frame within half a second. When none of those moments
    falls inside a clip, frames are taken every `every_seconds` instead. `mirror` flips the
    frames left to right for a camera whose picture is mirrored. The caption and the delivered
    file's name call the pet `subject` and give the visit's start in `zone`.

    With a `selector` whose owner has started labelling, extra frames are taken through the
    video and the trained model, once there is one, scores every frame. Both picks go in the
    visit's record; the selector's switch decides which is delivered. While the model is the
    one delivering, a visit whose best frame it scores below the selector's threshold is
    archived and cleared off the camera without being delivered. A model that fails leaves
    the choice to the built-in scoring, which delivers every visit.

    A visit that was delivered on an earlier attempt, but could not then be cleared off the
    camera, is only cleared: nobody gets the same postcard twice.
    """
    visit_dir = archive_dir_for(archive_dir, visit)
    delivered = _delivered_before(visit_dir)
    if delivered is not None:
        client.delete_visit(visit.id)
        return delivered
    clips_dir = visit_dir / "clips"
    frames_dir = visit_dir / "frames"
    recorded: List[RecordedClip] = []
    for clip in visit.clips:
        local = clips_dir / clip.name
        client.download_clip(clip.name, local, expected_size=clip.size)
        length = clip_length(local)
        if length.total == 0:
            # The recorder leaves a single-frame clip when it is stopped as a slice begins;
            # there is no span to sample and ffmpeg cannot re-encode its lone frame.
            log.info("visit %s: clip %s holds one frame, skipped", visit.id, clip.name)
            continue
        # The clip's modification time is when its last frame was written.
        recorded.append(
            RecordedClip(
                path=local,
                start=clip.mtime - length.total,
                duration=length.total,
                video_duration=length.video,
            )
        )

    photographer = Photographer(visit, recorded, frames_dir, mirror)
    candidates = photographer.moments(MOMENTS_PER_VISIT, MOMENT_SPACING_SECONDS)
    moments_found = bool(candidates)
    if not candidates:
        candidates = photographer.every(every_seconds)
    if not candidates:
        raise RuntimeError(f"visit {visit.id} produced no frames")

    ranked = rank(
        candidates,
        visit.sightings,
        DETECTION_RESOLUTION,
        {path: clarity(measures) for path, measures in photographer.measures.items()},
    )
    heuristic_best = ranked[0]
    # Several moments can land on one frame (all those before the first clip are photographed
    # from its first half second). Each frame is a candidate once, at the moment the built-in
    # scoring likes best.
    kept: dict = {}
    for candidate in ranked:
        kept.setdefault(candidate.path, candidate)
    candidates = [candidate for candidate in candidates if kept[candidate.path] is candidate]
    best, chosen_by, considered = heuristic_best, HEURISTIC, len(candidates)
    source = "moment" if moments_found else "interval"
    sources = {candidate.path: source for candidate in candidates}
    pool = list(candidates)
    model_best: Optional[Candidate] = None
    scores: Optional[List[float]] = None
    posted, best_score = True, None
    if selector is not None and selector.in_use():
        for candidate in photographer.spread(SPREAD_FRAMES):
            if candidate.path not in sources:
                sources[candidate.path] = "spread"
                pool.append(candidate)
    # Beside the frames, where the selector's model looks for them.
    write_measures(frames_dir, {path.name: taken for path, taken in photographer.measures.items()})
    if selector is not None and selector.in_use():
        scores = selector.scores([candidate.path for candidate in pool])
        if scores is not None:
            best_score, model_best = max(zip(scores, pool), key=lambda scored: scored[0])
            if selector.active() == MODEL:
                best, chosen_by, considered = model_best, MODEL, len(pool)
                # A visit with no frame the model thinks you would call good is kept, not sent.
                posted = best_score >= selector.post_at_least()
    result = CollectResult(
        best_frame=best.path,
        best_frame_at=best.at,
        frames_considered=considered,
        chosen_by=chosen_by,
        posted=posted,
        score=best_score if chosen_by == MODEL else None,
    )
    record = {
        **asdict(result),
        "best_frame": str(best.path),
        "heuristic_frame": str(heuristic_best.path),
        "model_frame": str(model_best.path) if model_best is not None else None,
        "candidates": [
            {
                "frame": str(candidate.path),
                "at": candidate.at,
                "source": sources[candidate.path],
                "model_score": scores[index] if scores is not None else None,
            }
            for index, candidate in enumerate(pool)
        ],
    }
    write_whole(
        visit_dir / "visit.json", json.dumps({"visit": asdict(visit), "result": record}, indent=1)
    )
    handled_at = datetime.now().astimezone().isoformat()
    if result.posted:
        photo = visit_dir / photo_name(visit, zone, subject)
        shutil.copyfile(best.path, photo)
        send(
            Delivery(
                caption=caption(visit, result.frames_considered, zone, subject),
                photo=photo,
                visit_dir=visit_dir,
                clips=[clip.path for clip in recorded],
            )
        )
        (visit_dir / DELIVERED_FILE).write_text(handled_at + "\n")
    else:
        # The marker still says the collector is done with the visit, so it is not collected
        # again and the archive's keep time applies to it.
        (visit_dir / DELIVERED_FILE).write_text(handled_at + " not posted\n")
    client.delete_visit(visit.id)
    return result

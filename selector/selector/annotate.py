# ABOUTME: Serves the selector's web page: the system's health, the collector's settings, aiming
# ABOUTME: the camera, and rating frames to train the model that picks which photo is posted.

import functools
import hashlib
import hmac
import http.client
import io
import ipaddress
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from PIL import Image

from selector.detect import Detector, check_classes, known_classes
from selector.features import Described, Featurizer, rule_of_thumb
from selector.frames import add_spread_frames
from selector.model import Model
from selector.store import BAD, BEST, GOOD, OKAY, Store
from selector.train import FEWEST_OF_EACH, FEWEST_VISITS, NotEnoughLabels, train
from selector.visits import Visit, all_visits, archived_visit

log = logging.getLogger("selector")

PAGE = Path(__file__).with_name("page.html")
# User name and password of the reCamera's video stream as it leaves the factory.
FACTORY_STREAM_LOGIN = ("admin", "admin")
# A visit's or a frame's name as one plain path component: never ".", ".." or a hidden file.
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")
# Visits looked at, newest first, when comparing the two selectors' picks.
COMPARE_VISITS = 40
# A frame the trained model scores at least this is one it takes for good.
MODEL_GOOD = 0.5
# Two frames this alike as whole pictures, with the pet's boxes overlapping this much, are
# near copies: only the better-scored one needs a rating.
COPY_SIMILARITY = 0.97
COPY_OVERLAP = 0.6
# Without a trained model, the favourite frame is proposed as good only above this score.
RULE_GOOD = 0.3
# The page's own requests are small; a visit's ratings are the longest. Nothing longer is read.
MAX_BODY_BYTES = 1_000_000


# The settings for the camera's home and patrol until the page changes them; the collector's
# defaults are the same.
VIEW_DEFAULTS = {"mode": "fixed", "explore": 0.3, "since": None}
# How many stops a sweep has and the poses it may choose from, likewise the collector's.
SWEEP_DEFAULTS = {
    "stops": 8,
    "yaw_low": 0.0,
    "yaw_high": 345.0,
    "pitch_low": 70.0,
    "pitch_high": 140.0,
}
MOST_STOPS = 16
# What the collector goes by until state.json says otherwise: the model's score a visit needs
# to be posted, how hard repeats of posted photos are marked down, and for how many hours a
# posted photo counts as posted.
POSTING_DEFAULTS = {"post_at_least": 0.5, "repeat_strictness": 1.0, "repeat_hours": 24.0}
POSTING_MOST = {"post_at_least": 1.0, "repeat_strictness": 5.0, "repeat_hours": 168.0}
# The collector's settings the page may change; the collector reads the file at every pass.
COLLECTOR_SETTINGS = (
    "paused",
    "deliver",
    "deliver_url",
    "deliver_clips",
    "deliver_every_minutes",
    "subject",
    "timezone",
    "keep_visits_days",
)
LONGEST_SUBJECT = 40
# Seconds the archive's size is remembered before it is added up again.
ARCHIVE_SIZE_EVERY_SECONDS = 300.0
# How many of the collector's latest log lines Home shows.
LOG_LINES_SHOWN = 40
# The archive's keep time is never set shorter than this from the page.
FEWEST_KEEP_DAYS = 1.0
# Seconds the camera is given to answer the page's aiming buttons.
CAMERA_TIMEOUT_SECONDS = 8
# The camera agent's code in this copy of the repository, when the page runs from one.
AGENT_PACKAGE = Path(__file__).resolve().parents[2] / "camera" / "agent" / "petcam"
# Longest one still from the camera's stream may take: it waits for the stream's next whole frame.
PICTURE_TIMEOUT_SECONDS = 25
PICTURE_WIDTH = 960
# The furthest one press of an aiming button turns the camera, in degrees.
LONGEST_NUDGE = 90.0
# The camera takes no more stops than this in a plan.
MOST_PLAN_STOPS = 24
# The header a change carries when the page was started with a passphrase.
PASSPHRASE_HEADER = "X-Pawst-Passphrase"


# Endings of host names that only a home network or a tailnet can give an address to.
PRIVATE_NAME_ENDINGS = (".local", ".lan", ".home.arpa", ".internal", ".ts.net")


def own_address(host_header: Optional[str], also: Sequence[str] = ()) -> bool:
    """Whether a request's Host is one this computer can go by: a numeric address, a name
    without dots, a name only a private network resolves, or one in `also`. A page elsewhere
    can point a domain of its own at this computer and so have a browser treat the selector
    as part of its site; its requests carry that domain."""
    try:
        name = (urlsplit("//" + (host_header or "")).hostname or "").rstrip(".")
    except ValueError:
        return False
    also_named = {other.lower().rstrip(".") for other in also}
    if (name and "." not in name) or name.endswith(PRIVATE_NAME_ENDINGS) or name in also_named:
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def _one_save_at_a_time(method: Callable) -> Callable:
    """Two requests that change the same settings file must not overlap: each reads it,
    changes it and writes it back."""

    @functools.wraps(method)
    def alone(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        with self._saving:
            return method(self, *args, **kwargs)

    return alone


class App:
    """What the page's requests act on: the archive, the store, and one training run at a time."""

    def __init__(
        self,
        store: Store,
        archive_dir: Path,
        featurizer: Featurizer,
        mirror: bool = True,
        detector_for: Optional[Callable[[Sequence[str]], Detector]] = None,
        camera_url: Optional[str] = None,
        stream_url: Optional[str] = None,
        grab: Optional[Callable[[str, bool], bytes]] = None,
        agent_package: Path = AGENT_PACKAGE,
        stream_login: Tuple[str, str] = FACTORY_STREAM_LOGIN,
    ) -> None:
        self.store = store
        # Where the camera's agent answers; without it, where the collector says it does.
        self._camera_url = camera_url
        # Where the camera's video stream is; without it, the stock address on the agent's host.
        self._stream_url = stream_url
        # User name and password of that stock stream.
        self._stream_login = stream_login
        self._grab = grab or grab_still
        self._agent_package = agent_package
        # One still at a time: the camera's stream is slow to start and the page asks often.
        self._picture_lock = threading.Lock()
        # Makes the detector for another choice of animals, when the page changes it.
        self._detector_for = detector_for
        self._known_pets: Optional[List[str]] = None
        # Why the camera could not be told the animals last saved; None when it was.
        self._camera_pets_problem: Optional[str] = None
        # When the archive was last added up, and what it came to.
        self._archive_size: Tuple[float, int] = (0.0, 0)
        self.archive_dir = archive_dir
        self.featurizer = featurizer
        self.mirror = mirror
        self._training = threading.Lock()
        self.training_error: Optional[str] = None
        # The detector and the embedder are used by one request at a time.
        self._looking = threading.Lock()
        # Held while a settings file is read, changed and written back.
        self._saving = threading.RLock()
        self._cutting = threading.Lock()
        # The last comparison made: what was asked, the differing picks, how many agreed.
        self._comparing = threading.Lock()
        self._compared: Optional[Tuple[tuple, List[dict], int]] = None

    def visit(self, key: str) -> Optional[Visit]:
        if not SAFE_NAME.match(key):
            return None
        visit = archived_visit(self.archive_dir / key)
        if visit is not None:
            return visit
        return next((v for v in all_visits(self.archive_dir, self.store) if v.key == key), None)

    def _model(self) -> Optional[Model]:
        path = self.store.current_model()
        return Model.load(path) if path is not None else None

    def status(self) -> dict:
        data = self.store.labels()
        votes = list(self.store.votes().values())
        return {
            "active": self.store.state()["active"],
            "model": self.store.model_meta(),
            "model_outdated": self.store.outdated_model(),
            "history": self.store.history(),
            "ratings": self.store.rating_counts(),
            "visits": len(data["visits"]),
            "votes": {name: votes.count(name) for name in ("model", "heuristic", "same")},
            "training": self._training.locked(),
            "training_error": self.training_error,
            "collector_ready": self.collector_ready(),
        }

    def collector_ready(self) -> bool:
        """Whether the running collector is a version that consults the selector: such a
        collector lists every candidate frame in the visits it archives. True while there is
        no visit to judge by."""
        if not self.archive_dir.is_dir():
            return True

        # Newest by when it was collected: visit names follow the camera's clock, which
        # starts over when the camera boots without a network.
        def collected_at(visit_dir: Path) -> float:
            try:
                return (visit_dir / "visit.json").stat().st_mtime
            except OSError:
                return 0.0

        for visit_dir in sorted(self.archive_dir.iterdir(), key=collected_at, reverse=True):
            try:
                record = json.loads((visit_dir / "visit.json").read_text())
                return "candidates" in record["result"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return True

    def home(self) -> dict:
        """What the Home tab's checklist is drawn from: how much the collector has archived and
        delivered and when it did so last, and when it last heard from the camera."""
        collected: List[float] = []
        delivered: List[float] = []
        if self.archive_dir.is_dir():
            for visit_dir in self.archive_dir.iterdir():
                # The picture of several visits sent as one has a directory of its own.
                if visit_dir.name.startswith("batch-"):
                    continue
                try:
                    collected.append((visit_dir / "visit.json").stat().st_mtime)
                    # The collector leaves this marker on every visit it is done with, and
                    # writes "not posted" in it when it kept the photo back.
                    marker = visit_dir / "delivered"
                    if "not posted" not in marker.read_text():
                        delivered.append(marker.stat().st_mtime)
                except OSError:
                    continue
        views = self.views()
        pose = views["pose"] or {}
        heard = pose.get("at")
        collector = _read_dict(self.store.dir / "collector" / "status.json")
        try:
            lines = json.loads((self.store.dir / "collector" / "log.json").read_text())
        except (OSError, ValueError):
            lines = []
        try:
            free = shutil.disk_usage(self.archive_dir).free
        except OSError:
            free = None
        return {
            # None from a collector too old to say; the page then leaves these lines out.
            "collector": (
                {
                    name: collector.get(name)
                    for name in ("at", "waiting", "next_delivery_at", "problem", "dry_run")
                }
                | {
                    "paused": (collector.get("settings") or {}).get("paused"),
                    "keep_visits_days": (collector.get("settings") or {}).get("keep_visits_days"),
                }
                if collector
                else None
            ),
            "log": lines[-LOG_LINES_SHOWN:] if isinstance(lines, list) else [],
            "archive_bytes": self._archive_bytes(),
            "disk_free_bytes": free,
            "now": time.time(),
            "collected": len(collected),
            "collected_at": max(collected, default=None),
            "delivered": len(delivered),
            "delivered_at": max(delivered, default=None),
            "camera_heard_at": heard if isinstance(heard, (int, float)) else None,
            "aim": views["settings"].get("mode"),
            "fewest_of_each": FEWEST_OF_EACH,
            "fewest_visits": FEWEST_VISITS,
        }

    def _archive_bytes(self) -> int:
        """How much the archive holds, added up at most every few minutes."""
        at, size = self._archive_size
        if time.monotonic() - at < ARCHIVE_SIZE_EVERY_SECONDS and at:
            return size
        size = 0
        if self.archive_dir.is_dir():
            for path in self.archive_dir.rglob("*"):
                try:
                    if path.is_file():
                        size += path.stat().st_size
                except OSError:
                    continue
        self._archive_size = (time.monotonic(), size)
        return size

    def visits(self) -> List[dict]:
        labelled = self.store.labels()["visits"]
        return [
            {
                "key": visit.key,
                "started_at": visit.started_at,
                "frames": len(visit.frames),
                "in_archive": visit.in_archive,
                "withheld": visit.withheld,
                "labelled": visit.key in labelled,
                "good": sum(
                    value >= GOOD
                    for value in labelled.get(visit.key, {}).get("frames", {}).values()
                ),
            }
            for visit in all_visits(self.archive_dir, self.store)
        ]

    def visit_detail(self, key: str) -> Optional[dict]:
        """A visit's frames, best guess first, each with the pet's box, a score, the rating it
        has or the one proposed for it, and which better frame it is a near copy of."""
        visit = self.visit(key)
        if visit is None:
            return None
        if (
            visit.in_archive
            and not visit.candidates_recorded
            and not any(frame.source == "spread" for frame in visit.frames)
        ):
            # A visit from before the collector recorded its candidates and took extra
            # frames: cut them from its clips now.
            with self._cutting:
                if add_spread_frames(self.archive_dir / key, self.mirror):
                    visit = self.visit(key) or visit
        entry = self.store.labels()["visits"].get(key)
        model = self._model()
        with self._looking:
            described = self.featurizer.describe([frame.path for frame in visit.frames])
            if model is not None:
                scores = model.scores(self.featurizer.features(described))
            else:
                scores = rule_of_thumb(described)
            frame_embeddings = self.featurizer.embedder.embed([d.path for d in described])
        order = [int(index) for index in np.argsort(-scores)]
        copy_of = near_copies(described, frame_embeddings, order)
        proposed = propose(described, scores, order, copy_of, trained=model is not None)
        frames = []
        for index in order:
            frame, photo = visit.frames[index], described[index]
            frames.append(
                {
                    "name": frame.name,
                    "source": frame.source,
                    "score": float(scores[index]),
                    "box": (
                        [photo.box.left, photo.box.top, photo.box.right, photo.box.bottom]
                        if photo.has_pet
                        else None
                    ),
                    "copy_of": visit.frames[copy_of[index]].name if index in copy_of else None,
                    "proposed": proposed[index],
                    "rating": (entry or {}).get("frames", {}).get(frame.name),
                }
            )
        return {
            "key": key,
            "started_at": visit.started_at,
            "posted": visit.posted,
            "withheld": visit.withheld,
            "labelled": entry is not None,
            "scored_by": "model" if model is not None else "rule",
            "pets": sum(photo.has_pet for photo in described),
            "frames": frames,
        }

    def label(self, key: str, labels: Dict[str, int]) -> None:
        visit = self.visit(key)
        if visit is None:
            raise ValueError(f"no visit {key}")
        if not isinstance(labels, dict):
            raise ValueError("labels must map frame names to ratings")
        unknown = set(labels) - set(visit.paths())
        if unknown:
            raise ValueError(f"visit {key} has no frame {min(unknown)}")
        self.store.label_visit(
            key, visit.started_at, visit.posted, visit.paths(), labels, heuristic=visit.heuristic
        )

    def compare(self, count: int = COMPARE_VISITS) -> dict:
        """For the latest archived visits, the frame the built-in scoring picked and the frame
        the current model picks from the same candidates. Listed are the visits where the two
        differ and at least one of the picks is worth sending: rated good or best by you, or,
        for a frame you have not rated, scored as good by the model. The rest are counted."""
        path = self.store.current_model()
        if path is None:
            return {"model": None, "entries": [], "same": 0, "neither": 0, "looked_at": 0}
        visits = [v for v in all_visits(self.archive_dir, self.store) if v.in_archive][:count]
        differing, same = self._picks(path, visits)
        votes = self.store.votes()
        rated = self.store.labels()["visits"]
        entries, neither = [], 0
        for pick in differing:
            ratings = rated.get(pick["key"], {}).get("frames", {})

            def worth(side: str) -> bool:
                rating = ratings.get(pick[side])
                if rating is not None:
                    return bool(rating >= GOOD)
                score = pick[side + "_score"]
                return score is not None and score >= MODEL_GOOD

            if worth("model") or worth("heuristic"):
                entries.append({**pick, "vote": votes.get(pick["key"])})
            else:
                neither += 1
        return {
            "model": path.stem,
            "entries": entries,
            "same": same,
            "neither": neither,
            "looked_at": len(visits),
        }

    def _picks(self, path: Path, visits: Sequence[Visit]) -> Tuple[List[dict], int]:
        """The visits among `visits` where the model at `path` and the built-in scoring pick
        different frames, and how many where they pick the same. Scoring takes seconds, so the
        answer is kept until the model or the visits change."""
        asked = (path.stem, tuple((v.key, v.heuristic, len(v.frames)) for v in visits))
        with self._comparing:
            if self._compared is not None and self._compared[0] == asked:
                return self._compared[1], self._compared[2]
            model = Model.load(path)
            differing: List[dict] = []
            same = 0
            for visit in visits:
                with self._looking:
                    described = self.featurizer.describe([frame.path for frame in visit.frames])
                    scores = model.scores(self.featurizer.features(described))
                pick = visit.frames[int(np.argmax(scores))].name
                if pick == visit.heuristic or visit.heuristic is None:
                    same += 1
                    continue
                by_name = {f.name: float(score) for f, score in zip(visit.frames, scores)}
                differing.append(
                    {
                        "key": visit.key,
                        "started_at": visit.started_at,
                        "heuristic": visit.heuristic,
                        "model": pick,
                        "heuristic_score": by_name.get(visit.heuristic),
                        "model_score": by_name[pick],
                    }
                )
            self._compared = (asked, differing, same)
            return differing, same

    def start_training(self) -> bool:
        if not self._training.acquire(blocking=False):
            return False
        self.training_error = None

        def run() -> None:
            try:
                with self._looking:
                    train(self.store, self.featurizer)
            except NotEnoughLabels as error:
                self.training_error = str(error)
            except Exception as error:
                log.exception("training failed")
                self.training_error = f"training failed: {error}"
            finally:
                self._training.release()

        threading.Thread(target=run, daemon=True).start()
        return True

    def views(self) -> dict:
        """What the Camera tab draws: the collector's map of the camera's surroundings, where
        the camera points now, and the settings the collector plans by. The map and the pose
        are None until the collector has written them."""
        directory = self.store.dir / "views"

        def read(name: str) -> Optional[dict]:
            try:
                value = json.loads((directory / name).read_text())
            except (OSError, ValueError):
                return None
            return value if isinstance(value, dict) else None

        return {
            "map": read("map.json"),
            "pose": read("pose.json"),
            "settings": {**VIEW_DEFAULTS, **(read("settings.json") or {})},
            "sweep": SWEEP_DEFAULTS,
        }

    @_one_save_at_a_time
    def set_views(self, body: dict) -> None:
        """Change the settings the collector plans the camera's home and patrol by. `mode` is
        "fixed", "learned" or "manual", `explore` runs from 0 to 1, and `moved` true says the camera
        stands somewhere new from now on (false: count everything again). Saving the sweep's
        ranges also has the camera forget where its gimbal stopped short, so that the whole of
        them is tried again; a camera that cannot be told keeps what it knows."""
        directory = self.store.dir / "views"
        settings = self.views()["settings"]
        if "mode" in body:
            if body["mode"] not in ("fixed", "learned", "manual"):
                raise ValueError("mode must be fixed, learned or manual")
            settings["mode"] = body["mode"]
        if "explore" in body:
            explore = float(body["explore"])
            if not 0 <= explore <= 1:
                raise ValueError("explore must be between 0 and 1")
            settings["explore"] = explore
        if "moved" in body:
            settings["since"] = time.time() if body["moved"] is True else None
        if "stops" in body:
            stops = body["stops"]
            if (
                isinstance(stops, bool)
                or not isinstance(stops, int)
                or not 0 <= stops <= MOST_STOPS
            ):
                raise ValueError(f"stops must be a whole number from 0 to {MOST_STOPS}")
            settings["stops"] = stops
        reach = {name: settings.get(name, SWEEP_DEFAULTS[name]) for name in SWEEP_DEFAULTS}
        for name in ("yaw_low", "yaw_high", "pitch_low", "pitch_high"):
            if name in body:
                reach[name] = _number(body[name], name, 0, 360 if name.startswith("yaw") else 180)
                settings[name] = reach[name]
        if reach["yaw_low"] > reach["yaw_high"] or reach["pitch_low"] > reach["pitch_high"]:
            raise ValueError("the low end of a range must not be above its high end")
        directory.mkdir(parents=True, exist_ok=True)
        scratch = directory / "settings.json.new"
        scratch.write_text(json.dumps(settings))
        scratch.replace(directory / "settings.json")
        if any(name in body for name in ("yaw_low", "yaw_high", "pitch_low", "pitch_high")):
            try:
                self._camera("/patrol", {"forget_missed": True})
            except ValueError:
                pass

    def _camera_base(self) -> str:
        """Where the camera's agent answers; ValueError, worded for the page, when unknown."""
        base = self._camera_url or _read_dict(self.store.dir / "collector" / "status.json").get(
            "camera"
        )
        if not isinstance(base, str) or urlsplit(base).scheme not in ("http", "https"):
            raise ValueError(
                "the page does not know where the camera is: the collector says so once it "
                "runs, or start the page with --camera"
            )
        return base

    def camera_code(self) -> dict:
        """Whether the camera runs the agent code this copy of the repository holds: the
        fingerprint of the code `here` (None when the page does not run from the repository)
        and the one the `camera` reports (None from an agent older than the report);
        `problem` says why when the camera could not be asked."""
        here = code_fingerprint(self._agent_package) if self._agent_package.is_dir() else None
        try:
            status = self._camera("/status")
        except ValueError as error:
            return {"problem": str(error), "here": here, "camera": None}
        code = status.get("code")
        return {"problem": None, "here": here, "camera": code if isinstance(code, str) else None}

    def camera_picture(self) -> bytes:
        """What the camera sees now, as a JPEG from its video stream; ValueError, worded for
        the page, when there is none to be had."""
        stream = self._stream_url
        if not stream:
            host = urlsplit(self._camera_base()).hostname
            if not host:
                raise ValueError("the page does not know where the camera's video is")
            user, password = (quote(part, safe="") for part in self._stream_login)
            stream = f"rtsp://{user}:{password}@{host}:554/live"
        with self._picture_lock:
            return self._grab(stream, self.mirror)

    def _camera(self, path: str, body: Optional[dict] = None) -> dict:
        """Ask the camera's agent; ValueError, worded for the page, when it cannot be asked."""
        base = self._camera_base()
        request = urllib.request.Request(
            base.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=CAMERA_TIMEOUT_SECONDS) as response:
                answer = json.loads(response.read())
        except urllib.error.HTTPError as error:
            try:
                said = json.loads(error.read()).get("error")
            except (ValueError, AttributeError):
                said = None
            if error.code == 404:
                said = "its agent is an older one without this; deploy the agent again"
            raise ValueError(f"the camera refused: {said or error.reason}") from None
        except (OSError, ValueError, http.client.HTTPException) as error:
            raise ValueError(
                f"the camera did not answer: {getattr(error, 'reason', error)}"
            ) from None
        if not isinstance(answer, dict):
            raise ValueError("the camera gave an answer that makes no sense")
        return answer

    def aim(self) -> dict:
        """Where the camera points, whether it follows the pet, and the home and stops it
        goes by; `problem` says why when the camera could not be asked."""
        try:
            status = self._camera("/status")
            patrol = self._camera("/patrol")
        except ValueError as error:
            return {"problem": str(error)}
        return {
            "problem": None,
            "pose": status.get("pose"),
            "tracking": status.get("tracking_enabled"),
            "in_visit": status.get("in_visit"),
            "plan": patrol.get("plan"),
            "default": patrol.get("default"),
            "by_hand_or_learned": patrol.get("learned"),
        }

    def set_aim(self, body: dict) -> None:
        """Turn the camera by `move` degrees (which stops it following the pet), switch
        `tracking`, or change the plan it rests and sweeps by: `keep` "home" or "stop" takes
        the direction it points now, `drop_stop` removes a stop by its place, `patrol_after`
        sets the idle seconds before a sweep. A change to the plan also tells the collector
        the plan is now set by hand, so that it leaves it alone."""
        if "move" in body:
            move = body["move"] if isinstance(body["move"], dict) else {}
            self._camera(
                "/move",
                {
                    name: _number(move.get(name, 0), name, -LONGEST_NUDGE, LONGEST_NUDGE)
                    for name in ("yaw", "pitch")
                },
            )
        if "tracking" in body:
            if not isinstance(body["tracking"], bool):
                raise ValueError("tracking must be true or false")
            self._camera("/tracking", {"enabled": body["tracking"]})
        if not {"keep", "drop_stop", "patrol_after"} & set(body):
            return
        plan = self._camera("/patrol").get("plan")
        if not isinstance(plan, dict) or not isinstance(plan.get("stops"), list):
            raise ValueError("the camera did not say what plan it follows")
        if "keep" in body:
            pose = self._camera("/status").get("pose")
            if not isinstance(pose, dict):
                raise ValueError("the camera does not know where it points right now; try again")
            pose = {"yaw": round(float(pose["yaw"]), 1), "pitch": round(float(pose["pitch"]), 1)}
            if body["keep"] == "home":
                plan["home"] = pose
            elif body["keep"] == "stop":
                if len(plan["stops"]) >= MOST_PLAN_STOPS:
                    raise ValueError(f"a sweep has at most {MOST_PLAN_STOPS} stops")
                plan["stops"] = [*plan["stops"], pose]
            else:
                raise ValueError("keep must be home or stop")
        if "drop_stop" in body:
            index = body["drop_stop"]
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(plan["stops"])
            ):
                raise ValueError("there is no such stop")
            plan["stops"] = plan["stops"][:index] + plan["stops"][index + 1 :]
        if "patrol_after" in body:
            plan["patrol_after"] = _number(body["patrol_after"], "patrol_after", 60, 86400)
        # First, so the collector does not put its own plan back between the two steps.
        mode = self.views()["settings"].get("mode")
        self.set_views({"mode": "manual"})
        try:
            self._camera("/patrol", plan)
        except Exception:
            # The camera did not take the plan, so the collector goes on as it did.
            if mode in ("fixed", "learned"):
                self.set_views({"mode": mode})
            raise

    def camera_timings(self) -> dict:
        """The camera's own timings of visits and sweeps, with their limits and the values
        it was installed with; `problem` says why when the camera could not be asked."""
        try:
            return {"problem": None, **self._camera("/tuning")}
        except ValueError as error:
            return {"problem": str(error)}

    def set_camera_timings(self, body: dict) -> None:
        """Change timings on the camera, or with {"default": true} go back to the ones it
        was installed with. The camera checks them and says which is wrong."""
        self._camera("/tuning", body)

    def settings(self) -> dict:
        """What the Settings tab shows: what the page has saved for the collector, what the
        collector says it goes by (None from a collector too old to say), the thresholds for
        posting, and the animals looked for. The webhook's address may hold a secret, so only
        its host is given."""
        directory = self.store.dir / "collector"
        saved = _read_dict(directory / "settings.json")
        state = self.store.state()
        if self._known_pets is None:
            self._known_pets = sorted(known_classes())
        url = saved.get("deliver_url")
        return {
            "now": time.time(),
            "saved": {
                **{
                    name: saved[name]
                    for name in COLLECTOR_SETTINGS
                    if name in saved and name != "deliver_url"
                },
                "deliver_url_host": urlsplit(url).hostname if isinstance(url, str) else None,
            },
            "collector": _read_dict(directory / "status.json") or None,
            "test_at": saved.get("test_at"),
            "posting": {
                name: _number(state.get(name, default), name, 0, POSTING_MOST[name], default)
                for name, default in POSTING_DEFAULTS.items()
            },
            "active": state["active"],
            "pets": list(self.store.pet_classes()),
            "known_pets": self._known_pets,
            "camera_pets_problem": self._camera_pets_problem,
            "has_model": self.store.model_meta() is not None,
        }

    @_one_save_at_a_time
    def set_settings(self, body: dict) -> None:
        """Change any of the collector's settings, the posting thresholds and the animals
        looked for. `test` true asks the collector for a test postcard and `send_now` true for
        the waiting visits. Nothing is changed unless everything in `body` makes sense. The
        animals are the camera's to follow as well: it is told, and `camera_pets_problem` in
        the settings says why when it could not be."""
        collector = {}
        for name in COLLECTOR_SETTINGS:
            if name in body:
                collector[name] = _collector_setting(name, body[name])
        posting = {
            name: _number(body[name], name, 0, POSTING_MOST[name])
            for name in POSTING_DEFAULTS
            if name in body
        }
        pets = None
        if "pets" in body:
            if not isinstance(body["pets"], list):
                raise ValueError("pets must be a list of names")
            pets = list(check_classes([str(name) for name in body["pets"]]))
        if body.get("test") is True:
            collector["test_at"] = time.time()
        if body.get("send_now") is True:
            collector["send_now_at"] = time.time()
        if collector:
            directory = self.store.dir / "collector"
            directory.mkdir(parents=True, exist_ok=True)
            saved = {**_read_dict(directory / "settings.json"), **collector}
            scratch = directory / "settings.json.new"
            # The webhook's address may hold a secret: the file is the owner's alone.
            with open(os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as out:
                out.write(json.dumps(saved, indent=1))
            scratch.replace(directory / "settings.json")
        if posting:
            self.store.set_numbers(posting)
        if pets is not None and tuple(pets) != self.store.pet_classes():
            self.store.set_pet_classes(pets)
            if self._detector_for is not None:
                with self._looking:
                    self.featurizer = Featurizer(self.featurizer.embedder, self._detector_for(pets))
        if pets is not None:
            try:
                self._camera("/pets", {"pets": pets})
                self._camera_pets_problem = None
            except ValueError as error:
                self._camera_pets_problem = str(error)

    def image(self, key: str, name: str, width: int) -> Optional[bytes]:
        if not SAFE_NAME.match(key) or not SAFE_NAME.match(name):
            return None
        path = self.archive_dir / key / "frames" / name
        if not path.is_file():
            path = self.store.kept_frame(key, name)
        if not path.is_file():
            return None
        if width <= 0:
            return path.read_bytes()
        with Image.open(path) as image:
            image.draft("RGB", (width, width))
            image = image.convert("RGB")
            image.thumbnail((width, width * 2))
            out = io.BytesIO()
            image.save(out, "JPEG", quality=82)
        return out.getvalue()


def _read_dict(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(
    value: object, name: str, low: float, high: float, fallback: Optional[float] = None
) -> float:
    """`value` as a number from `low` to `high`; otherwise `fallback`, or ValueError without."""
    if not isinstance(value, bool) and isinstance(value, (int, float)) and low <= value <= high:
        return float(value)
    if fallback is not None:
        return fallback
    raise ValueError(f"{name.replace('_', ' ')} must be a number from {low:g} to {high:g}")


def _collector_setting(name: str, value: object) -> object:
    """`value` as the collector's setting `name` is saved; ValueError when it cannot be that.
    The collector checks what it reads the same way and passes over what fails."""
    if name == "deliver":
        if value not in ("command", "webhook"):
            raise ValueError("deliver must be command or webhook")
        return value
    if name == "deliver_url":
        if not isinstance(value, str) or urlsplit(value.strip()).scheme not in ("http", "https"):
            raise ValueError("the webhook address must start with http:// or https://")
        if not urlsplit(value.strip()).hostname:
            raise ValueError("the webhook address has no host name")
        return value.strip()
    if name in ("deliver_clips", "paused"):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")
        return value
    if name == "subject":
        if not isinstance(value, str) or not 0 < len(value.strip()) <= LONGEST_SUBJECT:
            raise ValueError(f"the pet's name must be 1 to {LONGEST_SUBJECT} letters")
        return value.strip()
    if name == "timezone":
        try:
            return ZoneInfo(str(value)).key
        except (ZoneInfoNotFoundError, ValueError, OSError):
            raise ValueError(
                f"{value!r} is not a time zone; use a name such as America/New_York"
            ) from None
    if value is None:
        return None
    if name == "keep_visits_days":
        return _number(value, "days kept", FEWEST_KEEP_DAYS, 36500)
    return _number(value, "minutes between deliveries", 0, 7 * 24 * 60) or None


def near_copies(
    described: Sequence[Described], embeddings: np.ndarray, order: Sequence[int]
) -> Dict[int, int]:
    """Frames that show almost exactly what a better-scored frame shows, mapped to that frame:
    the whole picture is alike and the pet, if any, is in the same place."""
    kept: List[int] = []
    copy_of: Dict[int, int] = {}
    for index in order:
        for other in kept:
            alike = float(embeddings[index] @ embeddings[other]) >= COPY_SIMILARITY
            a, b = described[index], described[other]
            same_pet = a.has_pet == b.has_pet and (
                not a.has_pet or a.box.overlap(b.box) >= COPY_OVERLAP
            )
            if alike and same_pet:
                copy_of[index] = other
                break
        else:
            kept.append(index)
    return copy_of


def propose(
    described: Sequence[Described],
    scores: np.ndarray,
    order: Sequence[int],
    copy_of: Dict[int, int],
    trained: bool,
) -> Dict[int, int]:
    """A rating to start each frame at, for the labeller to correct. No pet, or a near copy of
    a better frame: bad. Otherwise the model's opinion when there is one (the best of the good
    ones as best), and without a model only its favourite, as good."""
    proposed = {index: BAD for index in range(len(described))}
    distinct = [index for index in order if index not in copy_of and described[index].has_pet]
    if not distinct:
        return proposed
    if not trained:
        if scores[distinct[0]] >= RULE_GOOD:
            proposed[distinct[0]] = GOOD
        return proposed
    for index in distinct:
        if scores[index] >= MODEL_GOOD:
            proposed[index] = GOOD
        elif scores[index] >= 0.25:
            proposed[index] = OKAY
    if proposed[distinct[0]] == GOOD:
        proposed[distinct[0]] = BEST
    return proposed


def code_fingerprint(package: Path) -> str:
    """The camera agent's own fingerprint of the code in `package`: the same sum as
    petcam.version.fingerprint, which cannot be imported from here."""
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def grab_still(stream_url: str, mirror: bool) -> bytes:
    """One frame of the video at `stream_url` as a JPEG, cut with ffmpeg and flipped left to
    right with `mirror`. Raises ValueError, worded for the page; the address may hold a
    password and is left out of it."""
    filters = ("hflip," if mirror else "") + f"scale={PICTURE_WIDTH}:-2"
    try:
        done = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp"]
            + ["-i", stream_url, "-frames:v", "1", "-vf", filters]
            + ["-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True,
            timeout=PICTURE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        raise ValueError("a picture needs ffmpeg on the page's computer") from None
    except subprocess.TimeoutExpired:
        raise ValueError("the camera's video did not start in time") from None
    if done.returncode != 0 or not done.stdout:
        said = done.stderr.decode(errors="replace").replace(stream_url, "the stream").strip()
        raise ValueError(
            "the camera's video could not be read"
            + (f": {said.splitlines()[-1][:200]}" if said else "")
        )
    return done.stdout


def make_handler(app: App, names: Sequence[str] = (), passphrase: Optional[str] = None) -> type:
    """The page's request handler. `names` are host names the page may be reached by,
    beyond those own_address always takes. With a `passphrase`, no setting, rating
    or camera pose is changed for a request that does not carry it; looking needs none."""

    class Handler(BaseHTTPRequestHandler):
        # A connection that sends nothing must not hold its thread for ever.
        timeout = 60

        def _foreign(self) -> bool:
            """Refuses a request that reached the page under a name that is not its own."""
            if own_address(self.headers.get("Host"), names):
                return False
            self._json(
                {
                    "error": "this page does not go by that host name; open it by its "
                    "address, or start it with that name as --host"
                },
                403,
            )
            return True

        def _send(self, status: int, body: bytes, content_type: str, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: object, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:
            if self._foreign():
                return
            try:
                self._get()
            except Exception as error:
                # The page shows this instead of waiting for ever on a dropped connection.
                log.exception("GET %s failed", self.path)
                self._json({"error": f"{type(error).__name__}: {error}"}, 500)

        def _get(self) -> None:
            url = urlparse(self.path)
            query = {name: values[0] for name, values in parse_qs(url.query).items()}
            if url.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/status":
                self._json(app.status())
            elif url.path == "/api/home":
                self._json(app.home())
            elif url.path == "/api/visits":
                self._json(app.visits())
            elif url.path == "/api/visit":
                detail = app.visit_detail(query.get("key", ""))
                self._json(detail or {"error": "no such visit"}, 200 if detail else 404)
            elif url.path == "/api/compare":
                self._json(app.compare())
            elif url.path == "/api/views":
                self._json(app.views())
            elif url.path == "/api/settings":
                self._json({**app.settings(), "passphrase": passphrase is not None})
            elif url.path == "/api/aim":
                self._json(app.aim())
            elif url.path == "/api/camera":
                self._json(app.camera_timings())
            elif url.path == "/api/camera/code":
                self._json(app.camera_code())
            elif url.path == "/api/aim/picture":
                try:
                    self._send(200, app.camera_picture(), "image/jpeg")
                except ValueError as error:
                    self._json({"error": str(error)}, 502)
            elif url.path == "/img":
                try:
                    width = int(query.get("w", "640"))
                except ValueError:
                    width = 640
                body = app.image(query.get("key", ""), query.get("name", ""), width)
                if body is None:
                    self._json({"error": "no such frame"}, 404)
                else:
                    self._send(200, body, "image/jpeg", cache=True)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self._foreign():
                return
            # Any site open in the same browser can make it POST here without asking. Browsers
            # name the sending site in Origin; only this page's own requests are acted on.
            origin = self.headers.get("Origin")
            if origin is not None and origin.split("://")[-1] != self.headers.get("Host"):
                self._json({"error": "requests from other web pages are refused"}, 403)
                return
            if passphrase is not None and not hmac.compare_digest(
                unquote(self.headers.get(PASSPHRASE_HEADER) or "").encode(), passphrase.encode()
            ):
                self._json({"error": "the passphrase is needed to change anything"}, 401)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 <= length <= MAX_BODY_BYTES:
                    raise ValueError(f"body must be at most {MAX_BODY_BYTES} bytes")
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/label":
                    app.label(body["key"], body["labels"])
                elif self.path == "/api/vote":
                    if app.visit(body["key"]) is None:
                        raise ValueError("no such visit")
                    app.store.vote(body["key"], body["vote"])
                elif self.path == "/api/use":
                    app.store.use(body["selector"])
                elif self.path == "/api/views":
                    app.set_views(body)
                    self._json(app.views())
                    return
                elif self.path == "/api/settings":
                    if not isinstance(body, dict):
                        raise ValueError("expected a JSON object")
                    app.set_settings(body)
                    self._json({**app.settings(), "passphrase": passphrase is not None})
                    return
                elif self.path == "/api/aim":
                    if not isinstance(body, dict):
                        raise ValueError("expected a JSON object")
                    app.set_aim(body)
                    self._json(app.aim())
                    return
                elif self.path == "/api/camera":
                    if not isinstance(body, dict):
                        raise ValueError("expected a JSON object")
                    app.set_camera_timings(body)
                    self._json(app.camera_timings())
                    return
                elif self.path == "/api/train":
                    if not app.start_training():
                        raise ValueError("training is already running")
                else:
                    self._json({"error": "not found"}, 404)
                    return
            except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as error:
                self._json({"error": str(error)}, 400)
                return
            except Exception as error:
                # The page shows this instead of a dropped connection.
                log.exception("POST %s failed", self.path)
                self._json({"error": f"{type(error).__name__}: {error}"}, 500)
                return
            self._json(app.status())

        def log_message(self, *args: object) -> None:
            pass

    return Handler


def serve(
    store: Store,
    archive_dir: Path,
    featurizer: Featurizer,
    host: str,
    port: int,
    mirror: bool,
    detector_for: Optional[Callable[[Sequence[str]], Detector]] = None,
    passphrase: Optional[str] = None,
    camera_url: Optional[str] = None,
    stream_url: Optional[str] = None,
    stream_login: Tuple[str, str] = FACTORY_STREAM_LOGIN,
) -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    app = App(
        store,
        archive_dir,
        featurizer,
        mirror=mirror,
        detector_for=detector_for,
        camera_url=camera_url,
        stream_url=stream_url,
        stream_login=stream_login,
    )
    try:
        server = ThreadingHTTPServer(
            (host, port), make_handler(app, names=(host,), passphrase=passphrase)
        )
    except OSError as error:
        raise SystemExit(
            f"cannot serve the page on {host} port {port}: {error.strerror or error}. If the "
            "page is already running, use that one; otherwise pick another port with --port."
        ) from None
    print(
        f"selector page: http://{host}:{port}/  (archive {archive_dir}, data {store.dir})",
        flush=True,
    )
    if passphrase is None and host not in ("127.0.0.1", "localhost"):
        print(
            "anyone who can reach that address can change where photos are sent; set "
            "PETCAM_PAGE_PASSPHRASE to ask for a passphrase before a change",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

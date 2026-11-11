# ABOUTME: HTTP API for the agent: status, tracking, gimbal moves, the finished visits and their
# ABOUTME: clips, and the patrol plan, timings and animals that can change while the agent runs.

import ipaddress
import json
import logging
import os
import re
import socket
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from petcam.controller import Controller, GimbalControl
from petcam.patrol import Plan, Plans
from petcam.pets import Pets
from petcam.tuning import Tuning
from petcam.version import fingerprint
from petcam.visits import recorded_clips

# Read once: what matters is the code that was loaded, not what a later deploy left on disk.
CODE = fingerprint()
MANUAL_MOVE_SPEED = 90
# A manual move turns the gimbal by at most one full turn either way.
MAX_MANUAL_MOVE_DEGREES = 360.0
# Request bodies are a few dozen bytes of JSON; nothing longer is read.
MAX_BODY_BYTES = 4096
# A client that sends or takes nothing for this long is dropped, so a connection that died
# mid-request on a bad link does not hold a thread for ever.
CLIENT_TIMEOUT_SECONDS = 60
# A finished visit normally has its clips within a slice of ending. One that still has none
# after this many slices never will (the recorder was not running, or it deleted them to make
# room), and is dropped from the list.
CLIPLESS_GRACE_SLICES = 3

log = logging.getLogger(__name__)


class BadRequest(Exception):
    pass


# Endings of host names that only a home network or a tailnet can give an address to.
PRIVATE_NAME_ENDINGS = (".local", ".lan", ".home.arpa", ".internal", ".ts.net")


def from_another_site(headers: Any) -> bool:
    """Whether a browser sent this request on behalf of a page served from somewhere else.
    Browsers name the page's site in Origin on such requests; the collector and command line
    tools send no Origin at all."""
    origin = headers.get("Origin")
    if origin is None:
        return False
    return origin.split("://")[-1] != headers.get("Host")


def own_address(host_header: Optional[str], also: Sequence[str] = ()) -> bool:
    """Whether a request's Host is one the camera can go by: a numeric address, a name without
    dots, a name only a private network resolves, or one listed in `also`. A page elsewhere
    can point a domain of its own at the camera's address so that a browser takes the agent
    for part of that page's site; its requests carry that domain as their Host. A request
    with no Host at all comes from no browser."""
    if not host_header:
        return True
    try:
        name = (urllib.parse.urlsplit("//" + host_header).hostname or "").rstrip(".")
    except ValueError:
        return False
    if "." not in name or name.endswith(PRIVATE_NAME_ENDINGS) or name in also:
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def copy_bytes(source: Any, out: Any, count: int) -> None:
    """Send `count` bytes of `source` to `out`, and no more even if `source` has since grown."""
    while count > 0:
        chunk = source.read(min(65536, count))
        if not chunk:
            break
        out.write(chunk)
        count -= len(chunk)


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A client that went away mid-reply is one log line, not a traceback."""
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionError, socket.timeout)):
            log.info("client %s went away: %s", client_address[0], error)
        else:
            log.exception("request from %s failed", client_address[0])


def parse_range(header: Optional[str], size: int) -> Optional[int]:
    """The offset a `Range: bytes=N-` header asks to start from, or None when the header is
    absent, is any other kind of range, or starts at or past the end of the file; in each of
    those cases the whole file is sent."""
    match = re.fullmatch(r"bytes=(\d+)-", header or "")
    if match is None:
        return None
    start = int(match.group(1))
    return start if start < size else None


class ControlApi:
    def __init__(
        self,
        controller: Controller,
        gimbal: GimbalControl,
        host: str,
        port: int,
        clip_dir: Path = Path("/userdata/Videos"),
        slice_seconds: int = 10,
        lock: Optional[threading.Lock] = None,
        health: Optional[Callable[[], Dict[str, Any]]] = None,
        plans: Optional[Plans] = None,
        names: Sequence[str] = (),
        tuning: Optional[Tuning] = None,
        pets: Optional[Pets] = None,
    ) -> None:
        """`lock` is the one that guards the controller wherever else it is used. `health`
        returns extra entries for the status report. `plans` is what lets a caller change
        where the camera rests and sweeps. `names` are host names the camera goes by beyond
        the ones `own_address` accepts anyway."""
        self.plans = plans
        # What lets a caller change the timings of visits and sweeps while the agent runs.
        self.tuning = tuning
        # What lets a caller change which animals are followed.
        self.pets = pets
        self.names = tuple(name.lower().rstrip(".") for name in names)
        self.controller = controller
        self.gimbal = gimbal
        self.clip_dir = clip_dir
        self.slice_seconds = slice_seconds
        self.lock = lock if lock is not None else threading.Lock()
        self.health = health
        api = self

        class Handler(BaseHTTPRequestHandler):
            timeout = CLIENT_TIMEOUT_SECONDS

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _foreign(self) -> bool:
                """Refuses a request that reached the camera under a name that is not its
                own, or that a browser sent for a page from somewhere else."""
                if not own_address(self.headers.get("Host"), api.names):
                    self._reply(
                        403,
                        {
                            "error": "the camera does not go by this host name; use its "
                            "address, or start the agent with --api-allow-host <name>"
                        },
                    )
                    return True
                if self.command != "GET" and from_another_site(self.headers):
                    # Any page open in a browser on this network could otherwise turn the
                    # camera: a browser sends such a POST without asking.
                    self._reply(403, {"error": "requests from other web pages are refused"})
                    return True
                return False

            def do_GET(self) -> None:
                if self._foreign():
                    return
                url = urllib.parse.urlsplit(self.path)
                if url.path == "/status":
                    self._reply(200, api.status())
                elif url.path == "/visits":
                    try:
                        limit = api.limit_from(url.query)
                    except BadRequest as error:
                        self._reply(400, {"error": str(error)})
                        return
                    visits, total = api.list_visits(limit)
                    self._reply(200, {"visits": visits, "total": total})
                elif url.path == "/gaze":
                    self._reply(200, api.gaze(url.query))
                elif url.path == "/patrol" and api.plans is not None:
                    self._reply(200, api.patrol_status())
                elif url.path == "/tuning" and api.tuning is not None:
                    self._reply(200, api.tuning_status())
                elif url.path == "/pets" and api.pets is not None:
                    self._reply(200, api.pets_status())
                elif url.path.startswith("/clips/"):
                    self._send_clip(api.clip_path(url.path[len("/clips/") :]))
                else:
                    self._reply(404, {"error": "not found"})

            def do_DELETE(self) -> None:
                if self._foreign():
                    return
                if self.path.startswith("/visits/"):
                    if api.delete_visit(self.path[len("/visits/") :]):
                        self._reply(200, {"deleted": True})
                    else:
                        self._reply(404, {"error": "unknown visit"})
                else:
                    self._reply(404, {"error": "not found"})

            def _send_clip(self, path: Optional[Path]) -> None:
                if path is None:
                    self._reply(404, {"error": "unknown clip"})
                    return
                try:
                    clip = path.open("rb")
                except FileNotFoundError:
                    # The recorder deleted it between the lookup and here.
                    self._reply(404, {"error": "unknown clip"})
                    return
                with clip:
                    # The size of the file that was opened, even if it has since been deleted.
                    size = os.fstat(clip.fileno()).st_size
                    start = parse_range(self.headers.get("Range"), size)
                    if start is None:
                        self.send_response(200)
                        start = 0
                    else:
                        # A download that was cut off carries on from where it stopped.
                        self.send_response(206)
                        self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
                        clip.seek(start)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(size - start))
                    self.end_headers()
                    # A slice still being written has grown by now; the length is a promise.
                    copy_bytes(clip, self.wfile, size - start)

            def do_POST(self) -> None:
                if self._foreign():
                    return
                try:
                    body = self._read_json()
                    if self.path == "/tracking":
                        self._reply(200, api.set_tracking(body))
                    elif self.path == "/move":
                        self._reply(200, api.manual_move(body))
                    elif self.path == "/patrol" and api.plans is not None:
                        self._reply(200, api.set_patrol(body))
                    elif self.path == "/tuning" and api.tuning is not None:
                        self._reply(200, api.set_tuning(body))
                    elif self.path == "/pets" and api.pets is not None:
                        self._reply(200, api.set_pets(body))
                    else:
                        self._reply(404, {"error": "not found"})
                except BadRequest as error:
                    self._reply(400, {"error": str(error)})

            def _read_json(self) -> Dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    raise BadRequest("Content-Length must be a number")
                if not 0 <= length <= MAX_BODY_BYTES:
                    raise BadRequest(f"body must be at most {MAX_BODY_BYTES} bytes")
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, RecursionError):
                    raise BadRequest("body must be JSON")
                if not isinstance(body, dict):
                    raise BadRequest("body must be a JSON object")
                return body

            def _reply(self, status: int, body: Dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = QuietServer((host, port), Handler)
        self.port = self.server.server_address[1]
        self._thread = threading.Thread(
            target=self.server.serve_forever, name="control-api", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def set_tracking(self, body: Dict[str, Any]) -> Dict[str, Any]:
        enabled = body.get("enabled")
        if not isinstance(enabled, bool):
            raise BadRequest("enabled must be true or false")
        with self.lock:
            self.controller.tracking_enabled = enabled
        return self.status()

    def manual_move(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Move by the given degrees; a manual move takes over from automatic tracking."""
        try:
            yaw = float(body.get("yaw", 0))
            pitch = float(body.get("pitch", 0))
        except (TypeError, ValueError, OverflowError):
            raise BadRequest("yaw and pitch must be numbers")
        # Written so that NaN, which compares false with everything, is refused as well.
        if not (abs(yaw) <= MAX_MANUAL_MOVE_DEGREES and abs(pitch) <= MAX_MANUAL_MOVE_DEGREES):
            raise BadRequest(f"yaw and pitch must be within {MAX_MANUAL_MOVE_DEGREES:g} degrees")
        with self.lock:
            self.controller.tracking_enabled = False
            self.gimbal.move(yaw, MANUAL_MOVE_SPEED, pitch, MANUAL_MOVE_SPEED)
            # The patrol sends the gimbal home again once tracking is back on.
            self.controller.moved_by_hand()
        return self.status()

    def gaze(self, query: str) -> Dict[str, Any]:
        """Where the camera has looked while waiting, hour by hour; `since` in the query
        string names the first hour wanted."""
        since = urllib.parse.parse_qs(query).get("since")
        with self.lock:
            if self.controller.gaze is None:
                return {"hours": {}}
            return self.controller.gaze.since(since[-1] if since else None)

    def patrol_status(self) -> Dict[str, Any]:
        with self.lock:
            return self.plans.status()

    def set_patrol(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Follow the plan in `body`, or with {"default": true} the one the agent started
        with. {"forget_missed": true} changes no plan: it forgets which poses the gimbal failed
        to reach."""
        with self.lock:
            if body.get("forget_missed") is True:
                self.plans.forget_missed()
            elif body.get("default") is True:
                self.plans.reset()
            else:
                try:
                    self.plans.set(Plan.from_dict(body))
                except ValueError as error:
                    raise BadRequest(str(error))
            return self.plans.status()

    def tuning_status(self) -> Dict[str, Any]:
        with self.lock:
            return self.tuning.status()

    def set_tuning(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Change the timings in `body`, or with {"default": true} go back to the ones the
        agent was started with."""
        with self.lock:
            if body.get("default") is True:
                self.tuning.reset()
            else:
                try:
                    self.tuning.set(body)
                except ValueError as error:
                    raise BadRequest(str(error))
            return self.tuning.status()

    def pets_status(self) -> Dict[str, Any]:
        with self.lock:
            return self.pets.status()

    def set_pets(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Follow the animals in `body["pets"]`, or with {"default": true} go back to the
        ones the agent was started with."""
        with self.lock:
            if body.get("default") is True:
                self.pets.reset()
            else:
                try:
                    self.pets.set(body.get("pets"))
                except ValueError as error:
                    raise BadRequest(str(error))
            return self.pets.status()

    def status(self) -> Dict[str, Any]:
        with self.lock:
            status = self.controller.status()
        if self.health is not None:
            status.update(self.health())
        # The code this agent was started from, for telling a stale deploy.
        status["code"] = CODE
        return status

    def limit_from(self, query: str) -> Optional[int]:
        """The `limit` of a /visits query string; None when it has none."""
        values = urllib.parse.parse_qs(query).get("limit")
        if not values:
            return None
        try:
            limit = int(values[-1])
        except ValueError:
            raise BadRequest("limit must be a whole number")
        if limit < 1:
            raise BadRequest("limit must be at least 1")
        return limit

    def list_visits(self, limit: Optional[int] = None) -> Tuple[list, int]:
        """The finished visits, oldest first and at most `limit` of them, and how many there
        are in all. A camera that was out of reach for days holds hundreds of visits with
        thousands of sightings each; the limit keeps one reply small enough to arrive."""
        # The clip directory is read once for all visits: a camera that has been offline holds
        # hundreds of clips and visits, and reading it per visit takes minutes there.
        recorded = recorded_clips(self.clip_dir)
        with self.lock:
            finished = self.controller.visits.finished()
        listed = []
        clipless = []
        now = time.time()
        for visit in finished:
            clips = self.controller.visits.clips_among(visit, recorded, self.slice_seconds)
            if not clips and self._past_clip_grace(visit.ended_at, now):
                log.warning("visit %s has no clips left on the camera; dropped", visit.id)
                clipless.append(visit.id)
                continue
            listed.append((visit, clips))
        if clipless:
            # All at once: each call writes the whole record of visits.
            with self.lock:
                self.controller.visits.collected(*clipless)
        total = len(listed)
        visits = []
        for visit, clips in listed if limit is None else listed[:limit]:
            data = visit.as_dict()
            data["clips"] = [
                {"name": clip.path.name, "size": clip.size, "mtime": clip.mtime} for clip in clips
            ]
            visits.append(data)
        return visits, total

    def _past_clip_grace(self, ended_at: Optional[float], now: float) -> bool:
        if ended_at is None:
            return False
        # A visit that ended in the future, or absurdly long ago, was timed before the clock
        # was set; either way its last clip is not still being written.
        return not 0 <= now - ended_at <= CLIPLESS_GRACE_SLICES * self.slice_seconds

    def clip_path(self, raw_name: str) -> Optional[Path]:
        """The clip file for a URL path segment, or None unless it is the plain name of an
        existing clip."""
        name = urllib.parse.unquote(raw_name)
        if not name or Path(name).name != name or not name.endswith(".mp4"):
            return None
        path = self.clip_dir / name
        return path if path.is_file() else None

    def delete_visit(self, visit_id: str) -> bool:
        """Forget a collected visit and remove its clips from the camera, except those another
        visit still needs."""
        with self.lock:
            visit = self.controller.visits.get(urllib.parse.unquote(visit_id))
        if visit is None:
            return False
        # A visit that began as this one ended shares the slice written at that moment; a clip
        # another visit still needs stays until that visit is deleted too.
        recorded = recorded_clips(self.clip_dir)
        visits = self.controller.visits
        with self.lock:
            others = [other for other in visits.finished() if other.id != visit.id]
            if visits.open is not None:
                others.append(visits.open)
        needed = {
            clip.path
            for other in others
            for clip in visits.clips_among(other, recorded, self.slice_seconds)
        }
        for clip in visits.clips_among(visit, recorded, self.slice_seconds):
            if clip.path not in needed:
                clip.path.unlink(missing_ok=True)
        with self.lock:
            self.controller.visits.collected(visit.id)
        return True

# ABOUTME: HTTP client for the camera agent's API: status, finished visits with their clips and
# ABOUTME: sightings, clip download, and deleting a visit once it has been collected.

import dataclasses
import errno
import http.client
import json
import logging
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.client import HTTPResponse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from collector.scoring import Sighting

log = logging.getLogger(__name__)

SIGHTING_FIELDS = tuple(field.name for field in dataclasses.fields(Sighting))


class CameraError(Exception):
    """The camera answered, but not with something usable."""


class CameraUnreachable(CameraError):
    """The camera did not answer, or the connection ended part way. Worth trying again later
    without counting it against whatever was being fetched."""


@dataclass(frozen=True)
class Clip:
    name: str
    size: int
    mtime: float


@dataclass(frozen=True)
class Visit:
    id: str
    started_at: float
    ended_at: Optional[float]
    sightings: List[Sighting]
    clips: List[Clip]
    # Seconds to add to the camera's times for this visit to get the real time; not zero when
    # the visit was recorded before the camera's clock was set.
    clock_offset: float = 0.0
    # Which boot of the camera recorded the visit.
    boot: Optional[str] = None
    # Where the camera pointed during the visit: [time, yaw, pitch] each time it had turned.
    # Empty from an agent that does not record it.
    gaze: List[List[float]] = field(default_factory=list)

    @property
    def real_started_at(self) -> float:
        return self.started_at + self.clock_offset


# What the camera's visit ids and clip names are made of. Nothing that separates directories,
# and nothing that means anything to a shell or a web page, where the names are shown.
# Failures of this computer's disk while a clip is written to it.
_DISK_ERRORS = (errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES)
PLAIN_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*\Z")


def _plain_name(value: object, what: str) -> str:
    """`value` when it is a single path component. The collector uses the camera's visit ids and
    clip names as local file names, so one that could lead out of the archive is refused."""
    if not isinstance(value, str) or not PLAIN_NAME.match(value):
        raise CameraError(f"camera sent an unusable {what}: {value!r}")
    return value


def _visit(v: dict) -> Visit:
    return Visit(
        id=_plain_name(v["id"], "visit id"),
        started_at=float(v["started_at"]),
        ended_at=None if v["ended_at"] is None else float(v["ended_at"]),
        # Only what is known here: an agent that says more about a sighting is still understood.
        sightings=[
            Sighting(**{name: s[name] for name in SIGHTING_FIELDS if name in s})
            for s in v["sightings"]
        ],
        clips=[
            Clip(name=_plain_name(c["name"], "clip name"), size=int(c["size"]), mtime=c["mtime"])
            for c in v["clips"]
        ],
        clock_offset=float(v.get("clock_offset", 0.0)),
        boot=v.get("boot"),
        gaze=[[float(part) for part in looked] for looked in v.get("gaze", [])],
    )


class CameraClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Unusable visit records already logged; each is reported once, not at every poll.
        self._unusable: Set[str] = set()
        # How many records of the last listing were unusable.
        self.unusable = 0

    def status(self) -> dict:
        return self._json("GET", "/status")

    def finished_visits(self) -> List[Visit]:
        """Every finished visit on the camera, oldest first."""
        return self.finished_visits_page(limit=None)[0]

    def finished_visits_page(self, limit: Optional[int]) -> Tuple[List[Visit], int]:
        """The oldest `limit` finished visits, and how many the camera holds in all. A camera
        that was out of reach for days holds more than fits in one reply."""
        listing = self._json("GET", "/visits" if limit is None else f"/visits?limit={limit}")
        try:
            entries = list(listing["visits"])
            total = int(listing.get("total", len(entries)))
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise CameraError(f"GET /visits: unusable listing: {error!r}") from error
        visits: List[Visit] = []
        self.unusable = 0
        for entry in entries:
            try:
                visits.append(_visit(entry))
            except (CameraError, KeyError, TypeError, ValueError, AttributeError) as error:
                # Left on the camera; the visits around it are still collected.
                self.unusable += 1
                unusable = str(error) if isinstance(error, CameraError) else repr(error)
                if unusable not in self._unusable:
                    self._unusable.add(unusable)
                    log.warning(
                        "GET /visits: one visit left out, its record is unusable: %s", unusable
                    )
        return visits, total

    def download_clip(self, name: str, dest: Path, expected_size: Optional[int] = None) -> None:
        """Fetch a clip to `dest`.

        A complete copy already on disk is kept, so a retried visit does not re-download its
        clips. A transfer that is cut short leaves what arrived in `dest` + ".part" and raises
        CameraUnreachable; the next call asks the camera for the rest only.
        """
        if expected_size is not None and dest.is_file() and dest.stat().st_size == expected_size:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.is_file() else 0
        resume = expected_size is not None and 0 < have < expected_size
        headers = {"Range": f"bytes={have}-"} if resume else {}
        try:
            with self._open("GET", self._path("/clips/", name), headers) as response:
                # A camera that does not resume answers 200 and sends the clip from its start.
                with part.open("ab" if response.status == 206 else "wb") as out:
                    shutil.copyfileobj(response, out)
        except (OSError, http.client.HTTPException) as error:
            if getattr(error, "errno", None) in _DISK_ERRORS:
                # This computer's disk, not the camera: say so rather than "unreachable".
                raise
            raise CameraUnreachable(f"clip {name}: transfer interrupted: {error}") from error
        actual = part.stat().st_size
        if expected_size is not None and actual < expected_size:
            raise CameraUnreachable(
                f"clip {name}: got {actual} of {expected_size} bytes; the rest next time"
            )
        if expected_size is not None and actual > expected_size:
            part.unlink()
            raise CameraError(f"clip {name}: got {actual} bytes, expected {expected_size}")
        part.replace(dest)

    def delete_visit(self, visit_id: str) -> None:
        self._json("DELETE", self._path("/visits/", visit_id))

    @staticmethod
    def _path(prefix: str, name: str) -> str:
        """The request path for a visit or clip the camera named, whatever is in the name."""
        return prefix + urllib.parse.quote(name, safe="")

    def gaze(self, since: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Seconds the camera spent looking each way while idle, hour -> "yaw,pitch" ->
        seconds, from the hour named `since` on."""
        reply = self._json("GET", "/gaze" if since is None else f"/gaze?since={since}")
        try:
            return {
                str(hour): {str(where): float(seconds) for where, seconds in looked.items()}
                for hour, looked in reply["hours"].items()
            }
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise CameraError(f"GET /gaze: unusable reply: {error!r}") from error

    def patrol(self) -> dict:
        """The plan the camera follows ("plan"), the one it was started with ("default"), and
        whether the first was set from outside ("learned")."""
        return self._json("GET", "/patrol")

    def set_patrol(self, plan: Optional[dict]) -> dict:
        """Have the camera follow `plan`, or with None the one it was started with."""
        return self._json("POST", "/patrol", {"default": True} if plan is None else plan)

    def _json(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        try:
            with self._open(method, path, body=body) as response:
                body = response.read()
        except (OSError, http.client.HTTPException) as error:
            raise CameraUnreachable(f"{method} {path}: reply interrupted: {error}") from error
        try:
            reply = json.loads(body)
        except ValueError as error:
            raise CameraError(f"{method} {path}: reply is not JSON: {body[:200]!r}") from error
        if not isinstance(reply, dict):
            raise CameraError(f"{method} {path}: reply is not a JSON object: {body[:200]!r}")
        return reply

    def _open(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[dict] = None,
    ) -> HTTPResponse:
        request = urllib.request.Request(
            self.base_url + path,
            method=method,
            headers=headers or {},
            data=None if body is None else json.dumps(body).encode(),
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            raise CameraError(
                f"{method} {path}: HTTP {error.code} {error.read()[:200]!r}"
            ) from error
        except (OSError, http.client.HTTPException) as error:
            raise CameraUnreachable(f"{method} {path}: {error}") from error

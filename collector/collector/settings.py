# ABOUTME: The settings the collector takes from a file it reads again at every pass, so the page
# ABOUTME: can change them while it runs; and the status file that tells the page what it uses.

import collections
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from PIL import Image, ImageDraw

from collector.deliver import Delivery, Sender, failure
from collector.files import write_whole
from collector.repeats import DELIVERED_FILE

log = logging.getLogger("collector")

# Both live in the collector's own directory within the selector's data. The page writes the
# first; the collector writes the second.
SETTINGS_FILE = "settings.json"
STATUS_FILE = "status.json"
# The collector's latest log lines, for the page.
LOG_FILE = "log.json"
LOG_LINES_KEPT = 60
# Things the page asks for once, each by writing the time it asked.
REQUESTS = ("test_at", "send_now_at")
# The status is written again this often when nothing in it has changed, so the page can
# tell a collector that is running from one that has stopped.
STATUS_EVERY_SECONDS = 60.0
DELIVERY_METHODS = ("command", "webhook")
LONGEST_SUBJECT = 40
# What of a failed test delivery's error the page is shown.
LONGEST_ERROR = 400
TEST_CAPTION = "Test postcard from Pawst Cards. If you can read this, delivery works."


@dataclass(frozen=True)
class Live:
    """The settings that can change while the collector runs. They start as the command
    line and environment gave them; each one the settings file names replaces its own."""

    deliver: Optional[str]
    deliver_url: Optional[str]
    deliver_clips: bool
    # None delivers every visit as it is collected.
    deliver_every_minutes: Optional[float]
    subject: str
    timezone: str
    # None keeps the archive for ever.
    keep_visits_days: Optional[float]
    # While paused, visits are collected and wait; nothing is delivered.
    paused: bool = False

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


def _amount(value: object) -> Optional[float]:
    """A number of minutes or days: none for null or zero. Raises for anything else."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1e9:
        raise ValueError("expected a number, zero or more")
    return float(value) or None


def _checked(name: str, value: object) -> object:
    """`value` as the setting `name` holds it; raises ValueError or TypeError when it is not
    something that setting can be."""
    if name == "deliver":
        if value not in DELIVERY_METHODS:
            raise ValueError("expected command or webhook")
        return value
    if name == "deliver_url":
        if not isinstance(value, str) or urlsplit(value).scheme not in ("http", "https"):
            raise ValueError("expected an http or https address")
        return value
    if name in ("deliver_clips", "paused"):
        if not isinstance(value, bool):
            raise ValueError("expected true or false")
        return value
    if name == "subject":
        if not isinstance(value, str) or not 0 < len(value.strip()) <= LONGEST_SUBJECT:
            raise ValueError(f"expected a name of at most {LONGEST_SUBJECT} letters")
        return value.strip()
    if name == "timezone":
        try:
            return ZoneInfo(str(value)).key
        except (ZoneInfoNotFoundError, ValueError, OSError):
            raise ValueError("expected a time zone name such as America/New_York") from None
    return _amount(value)


def read_live(path: Path, start: Live) -> Tuple[Live, Dict[str, Optional[float]]]:
    """`start` with what the settings file at `path` changes, and for each of REQUESTS the
    time the page last asked for it, or None. A value that makes no sense is left as `start` has
    it: the page checks what it writes, and a file edited by hand must not stop deliveries."""
    nothing: Dict[str, Optional[float]] = {name: None for name in REQUESTS}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return start, nothing
    if not isinstance(data, dict):
        return start, nothing
    changes = {}
    for name in asdict(start):
        if name in data:
            try:
                changes[name] = _checked(name, data[name])
            except (ValueError, TypeError):
                continue
    asked = {
        name: (
            float(data[name])
            if isinstance(data.get(name), (int, float))
            and not isinstance(data.get(name), bool)
            and abs(data[name]) < 1e12
            else None
        )
        for name in REQUESTS
    }
    return replace(start, **changes), asked


class RecentLog(logging.Handler):
    """Keeps the collector's latest log lines and writes them where the page reads them."""

    def __init__(self, path: Path) -> None:
        super().__init__(level=logging.INFO)
        self._path = path
        self._lines: Deque[dict] = collections.deque(maxlen=LOG_LINES_KEPT)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
        except Exception:
            return
        self._lines.append({"at": record.created, "level": record.levelname, "text": text[:600]})
        # At once: a pass that collects a backlog can take many minutes to finish. Failing to
        # is not worth a log line: that line would be new, and so on.
        try:
            write_whole(self._path, json.dumps(list(self._lines)))
        except OSError:
            pass


def write_status(
    path: Path,
    live: Live,
    command: Optional[str],
    dry_run: bool,
    problem: Optional[str],
    test: Optional[dict],
    camera: Optional[str] = None,
    waiting: int = 0,
    next_delivery_at: Optional[float] = None,
) -> None:
    """Tell the page what the collector goes by now. The webhook's address may hold a secret,
    so only its host is given; of the command, only the program's name."""
    settings = asdict(live)
    url = settings.pop("deliver_url")
    settings["deliver_url_host"] = urlsplit(url).hostname if url else None
    status = {
        "at": time.time(),
        "settings": settings,
        "command_program": Path(command.split()[0]).name if command and command.split() else None,
        "dry_run": dry_run,
        # Where the camera's agent answers, for the page's aiming buttons.
        "camera": camera,
        # Why the settings file's delivery settings are not the ones in use, when they are not.
        "problem": problem,
        "test": test,
        # Visits collected and not yet delivered, and when they are due to go.
        "waiting": waiting,
        "next_delivery_at": next_delivery_at,
    }
    try:
        write_whole(path, json.dumps(status, indent=1))
    except OSError as error:
        log.warning("could not write %s: %s", path, error)


def _latest_photo(archive_dir: Path) -> Optional[Path]:
    """The photo of the visit last delivered or left waiting for its delivery, if the archive
    still has one."""
    newest: Tuple[float, Optional[Path]] = (0.0, None)
    for marker in archive_dir.glob(f"*/{DELIVERED_FILE}"):
        try:
            at = marker.stat().st_mtime
            if at <= newest[0] or "not posted" in marker.read_text():
                continue
            photos = sorted(marker.parent.glob("*.jpg"))
            if photos and (marker.parent / "visit.json").is_file():
                newest = (at, photos[0])
        except OSError:
            continue
    return newest[1]


def _test_card(directory: Path) -> Path:
    """A plain picture to send when the archive has no photo yet, with the record a webhook
    delivery sends along."""
    directory.mkdir(parents=True, exist_ok=True)
    card = Image.new("RGB", (1280, 960), (38, 44, 58))
    ImageDraw.Draw(card).text((80, 440), "Pawst Cards test postcard", fill=(240, 236, 226))
    photo = directory / "test_postcard.jpg"
    card.save(photo, quality=88)
    (directory / "visit.json").write_text(json.dumps({"test": True}))
    return photo


def send_test(send: Sender, archive_dir: Path, scratch_dir: Path, requested_at: float) -> dict:
    """Deliver one test postcard through `send`: the photo delivered last, or a plain card.
    Returns what happened, for the page."""
    outcome = {"requested_at": requested_at, "at": time.time(), "ok": True, "error": None}
    try:
        photo = _latest_photo(archive_dir) or _test_card(scratch_dir)
        send(Delivery(caption=TEST_CAPTION, photo=photo, visit_dir=photo.parent, clips=[]))
    except subprocess.CalledProcessError as error:
        outcome.update(ok=False, error=failure(error))
    except Exception as error:
        outcome.update(ok=False, error=f"{type(error).__name__}: {error}")
    if outcome["error"]:
        outcome["error"] = str(outcome["error"]).strip()[:LONGEST_ERROR]
        log.error("test postcard not delivered: %s", outcome["error"])
    else:
        log.info("test postcard delivered")
    return outcome

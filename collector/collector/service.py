# ABOUTME: Long-running collector on an always-on machine: polls the camera agent for finished
# ABOUTME: visits, collects each one, and delivers the best photo by command or webhook.

import argparse
import fcntl
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from collector.aiming import Aimer
from collector.archive import prune_visits
from collector.camera_client import CameraClient, CameraError, CameraUnreachable
from collector.deliver import Delivery, Sender, command_sender, failure, webhook_sender
from collector.outbox import Outbox
from collector.repeats import RepeatSifter
from collector.process import DETECTION_RESOLUTION, collect_visit
from collector.selector import ModelSelector
from collector.settings import (
    SETTINGS_FILE,
    STATUS_FILE,
    STATUS_EVERY_SECONDS,
    LOG_FILE,
    Live,
    RecentLog,
    read_live,
    send_test,
    write_status,
)

log = logging.getLogger("collector")


TRUE_WORDS = ("1", "true", "yes")

# Visits asked for per request while working through what the camera holds.
VISITS_PER_REQUEST = 5
# Seconds between passes over the archive that delete visits past their keep time.
PRUNE_EVERY_SECONDS = 86400.0
# Held for as long as a collector runs on an archive.
LOCK_FILE = "collector.lock"
# Seconds between reminders in the log while the camera stays out of reach.
UNREACHABLE_REMINDER_SECONDS = 3600.0


class Retries:
    """Spaces out further attempts at visits that failed to collect: the first retry after
    `first_seconds`, each wait double the last up to `longest_seconds`. A visit that cannot be
    collected (an unreadable clip, a delivery target that is down) is then tried a few times an
    hour instead of at every poll."""

    def __init__(
        self,
        first_seconds: float = 30.0,
        longest_seconds: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.first_seconds = first_seconds
        self.longest_seconds = longest_seconds
        self._clock = clock
        # Visit id -> (failures so far, earliest time of the next attempt).
        self._failed: Dict[str, Tuple[int, float]] = {}

    def due(self, visit_id: str) -> bool:
        failed = self._failed.get(visit_id)
        return failed is None or self._clock() >= failed[1]

    def failed(self, visit_id: str) -> float:
        """Record a failed attempt; returns the seconds until the next one."""
        failures = self._failed.get(visit_id, (0, 0.0))[0] + 1
        # The doubling is capped, or a visit that fails for a week overflows a float.
        wait = min(self.first_seconds * 2 ** min(failures - 1, 30), self.longest_seconds)
        self._failed[visit_id] = (failures, self._clock() + wait)
        return wait

    def succeeded(self, visit_id: str) -> None:
        self._failed.pop(visit_id, None)

    def waiting(self) -> int:
        """How many visits have failed and not yet succeeded."""
        return len(self._failed)

    def only(self, visit_ids: Iterable[str]) -> None:
        """Forget every visit not among `visit_ids`: the camera no longer holds it."""
        held = set(visit_ids)
        self._failed = {name: value for name, value in self._failed.items() if name in held}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect pet visits from the camera and deliver the best photo of each"
    )
    parser.add_argument(
        "--camera",
        default=os.environ.get("PETCAM_CAMERA_AGENT_URL") or None,
        required=not os.environ.get("PETCAM_CAMERA_AGENT_URL"),
        help="camera agent base URL (default: $PETCAM_CAMERA_AGENT_URL)",
    )
    parser.add_argument(
        "--subject",
        default=os.environ.get("PETCAM_SUBJECT") or "pet",
        help="what the pet is called in captions and photo names (default: $PETCAM_SUBJECT or pet)",
    )
    parser.add_argument(
        "--deliver",
        choices=["command", "webhook"],
        default=os.environ.get("PETCAM_DELIVER"),
        help="how the photo leaves the house (default: $PETCAM_DELIVER)",
    )
    parser.add_argument(
        "--deliver-command",
        default=os.environ.get("PETCAM_DELIVER_COMMAND"),
        help="command line run per delivery; {photo}, {caption} and {visit_dir} each become one "
        "argument (default: $PETCAM_DELIVER_COMMAND)",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=_env_path("PETCAM_STAGE_DIR"),
        help="copy the photo here before running the command, for tools that only read their "
        "own directories (default: $PETCAM_STAGE_DIR)",
    )
    parser.add_argument(
        "--deliver-url",
        default=os.environ.get("PETCAM_DELIVER_URL"),
        help="URL that receives caption, visit JSON and photo as multipart form data "
        "(default: $PETCAM_DELIVER_URL)",
    )
    parser.add_argument(
        "--deliver-clips",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PETCAM_DELIVER_CLIPS", "").lower() in TRUE_WORDS,
        help="also send the visit's clips to the webhook (default: $PETCAM_DELIVER_CLIPS)",
    )
    parser.add_argument(
        "--deliver-every-minutes",
        type=_amount("minutes"),
        default=os.environ.get("PETCAM_DELIVER_EVERY_MINUTES", ""),
        help="deliver at most once in this many minutes; visits collected in between go out "
        "together as one delivery, their photos in one picture. Unset delivers every visit as "
        "it is collected (default: $PETCAM_DELIVER_EVERY_MINUTES)",
    )
    parser.add_argument(
        "--selector-dir",
        type=Path,
        default=_env_path("PETCAM_SELECTOR_DIR") or Path.home() / "petcam" / "selector",
        help="data directory of the optional trained photo selector (selector/ in this "
        "repository); the photo selector is unused until its tools have set the directory up, "
        "and the map of the camera's views is kept there too (default: $PETCAM_SELECTOR_DIR "
        "or ~/petcam/selector)",
    )
    parser.add_argument("--archive-dir", type=Path, default=Path.home() / "petcam" / "visits")
    parser.add_argument(
        "--keep-visits-days",
        type=_amount("days"),
        default=os.environ.get("PETCAM_KEEP_VISITS_DAYS", ""),
        help="delete an archived visit, clips, frames, photo and record, this many days after "
        "it was collected; unset or 0 keeps the archive for ever "
        "(default: $PETCAM_KEEP_VISITS_DAYS)",
    )
    parser.add_argument(
        "--every",
        type=float,
        default=5.0,
        help="seconds between candidate frames in a clip with no sighting to go by",
    )
    parser.add_argument("--poll", type=float, default=15.0, help="seconds between camera polls")
    parser.add_argument("--once", action="store_true", help="run one polling pass and exit")
    parser.add_argument("--dry-run", action="store_true", help="log instead of delivering")
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="flip frames left to right; the reCamera's picture is mirrored",
    )
    parser.add_argument(
        "--timezone",
        type=_zone,
        default=os.environ.get("PETCAM_TIMEZONE", ""),
        help="time zone for the caption and photo name; where the pets live "
        "(default: $PETCAM_TIMEZONE)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _zone(text: str) -> ZoneInfo:
    """A time zone by name from the command line or environment; empty means UTC."""
    try:
        return ZoneInfo(text.strip() or "UTC")
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise argparse.ArgumentTypeError(
            f"unknown time zone {text!r}; use a name such as America/New_York"
        )


def _amount(unit: str) -> Callable[[str], Optional[float]]:
    """A reader of a number of minutes or days from the command line or environment; empty or
    zero means none."""

    def read(text: str) -> Optional[float]:
        if not text.strip():
            return None
        try:
            amount = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a number of {unit}, got {text!r}")
        if not 0 <= amount < float("inf"):
            raise argparse.ArgumentTypeError(f"a number of {unit} cannot be {text}")
        return amount or None

    return read


def _env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value) if value else None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def sender_for(args: argparse.Namespace, live: Live) -> Sender:
    """The delivery sink `live` asks for, with the command and the directory photos are staged
    in taken from `args`: those are not changed while the collector runs. Raises ValueError
    when the chosen method's own setting is missing. A dry run only logs what would have been
    delivered."""
    if args.dry_run:

        def log_only(delivery: Delivery) -> None:
            log.info("dry run, would deliver: %s [%s]", delivery.caption, delivery.photo)

        return log_only
    if live.deliver == "command":
        if not args.deliver_command:
            raise ValueError("--deliver command needs --deliver-command")
        return command_sender(args.deliver_command, stage_dir=args.stage_dir)
    if live.deliver == "webhook":
        if not live.deliver_url:
            raise ValueError("--deliver webhook needs --deliver-url")
        return webhook_sender(live.deliver_url, include_clips=live.deliver_clips)
    if live.deliver:
        raise ValueError(f"--deliver must be command or webhook, not {live.deliver!r}")
    raise ValueError("--deliver is required unless --dry-run")


def live_from(args: argparse.Namespace) -> Live:
    """The changeable settings as the command line and environment gave them."""
    return Live(
        deliver=args.deliver,
        deliver_url=args.deliver_url,
        deliver_clips=args.deliver_clips,
        deliver_every_minutes=args.deliver_every_minutes,
        subject=args.subject,
        timezone=args.timezone.key,
        keep_visits_days=args.keep_visits_days,
    )


def run_once(
    client: CameraClient,
    archive_dir: Path,
    every_seconds: float,
    send: Sender,
    zone: ZoneInfo,
    subject: str,
    mirror: bool = False,
    retries: Optional[Retries] = None,
    selector: Optional[ModelSelector] = None,
    limit: Optional[int] = None,
) -> int:
    """Collect every finished visit the camera reports and hand each to `send`; returns how
    many were collected. With `limit`, stops after trying that many, so that a long backlog
    does not keep the caller from its other work.

    Visits are fetched a few at a time, oldest first. One that fails is logged and left on the
    camera, and `retries` decides when it is next tried. Raises CameraUnreachable as soon as
    the camera stops answering; what is left waits for the next pass.
    """
    retries = retries if retries is not None else Retries()
    collected = 0
    tried = 0
    while True:
        # Visits that are waiting for a retry, and ones whose record is unusable, stay at the
        # front of the camera's list, so the request reaches past them.
        visits, total = client.finished_visits_page(
            VISITS_PER_REQUEST + retries.waiting() + client.unusable
        )
        if total <= len(visits) + client.unusable:
            retries.only(visit.id for visit in visits)
        due = [visit for visit in visits if retries.due(visit.id)]
        for visit in due:
            if limit is not None and tried >= limit:
                return collected
            tried += 1
            try:
                result = collect_visit(
                    client,
                    visit,
                    archive_dir,
                    every_seconds,
                    send,
                    zone,
                    mirror=mirror,
                    subject=subject,
                    selector=selector,
                )
            except CameraUnreachable:
                raise
            except Exception as error:
                # Whatever went wrong with this visit, the ones after it still get their turn.
                log.error(
                    "visit %s not collected: %s; next attempt in %.0f s",
                    visit.id,
                    failure(error),
                    retries.failed(visit.id),
                )
                continue
            retries.succeeded(visit.id)
            collected += 1
            if not result.posted:
                log.info(
                    "visit %s collected, not posted: the best of %d frames scored %.2f, picked by %s",
                    visit.id,
                    result.frames_considered,
                    result.score or 0.0,
                    result.chosen_by,
                )
                continue
            log.info(
                "visit %s collected: %s (best of %d frames, picked by %s)",
                visit.id,
                result.best_frame.name,
                result.frames_considered,
                result.chosen_by,
            )
        if not due or total <= len(visits):
            return collected


class Reachability:
    """Logs the camera going out of reach and coming back once each, with a reminder every
    hour in between, instead of a line at every poll."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._unreachable_since: Optional[float] = None
        self._last_logged_at = 0.0

    def unreachable(self, error: Exception) -> None:
        now = self._clock()
        if self._unreachable_since is None:
            self._unreachable_since = now
            self._last_logged_at = now
            log.warning("camera unreachable: %s", error)
        elif now - self._last_logged_at >= UNREACHABLE_REMINDER_SECONDS:
            self._last_logged_at = now
            log.warning(
                "camera still unreachable after %.0f min: %s",
                (now - self._unreachable_since) / 60,
                error,
            )

    def reachable(self) -> None:
        if self._unreachable_since is not None:
            log.info(
                "camera reachable again after %.0f min",
                (self._clock() - self._unreachable_since) / 60,
            )
            self._unreachable_since = None


class CaptureHealth:
    """Logs when the camera's agent says its detection frames have stopped, and when they are
    back. The agent restarts capture on its own; this puts the outage in the collector's log,
    where a missing postcard is looked into first."""

    def __init__(self) -> None:
        self._stalled = False

    def report(self, status: dict) -> None:
        capture = status.get("capture")
        if not isinstance(capture, dict):
            return
        stalled = not capture.get("flowing", True)
        if stalled and not self._stalled:
            log.warning(
                "camera is not producing detection frames (capture restarts so far: %s)",
                capture.get("restarts"),
            )
        elif self._stalled and not stalled:
            log.info("camera is producing detection frames again")
        self._stalled = stalled


def deliver_waiting(outbox: Outbox, at_once: bool = False) -> None:
    """Deliver the outbox's visits if a delivery is due, or `at_once` regardless, logging the
    outcome either way."""
    try:
        delivered = outbox.flush(at_once)
    except Exception as error:
        log.error("delivery of %d visits failed: %s", outbox.waiting(), failure(error))
    else:
        if delivered:
            log.info("delivered %d visit(s) in one post", delivered)


def keep_aim(aimer: Aimer, status: dict) -> None:
    """Bring the map of the camera's surroundings and the camera's plan up to date. Nothing
    that goes wrong here is allowed to get in the way of collecting visits."""
    try:
        aimer.saw(status)
        aimer.refresh()
    except CameraUnreachable:
        raise
    except Exception as error:
        log.error("aim not updated: %s", error)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    # One collector to an archive: two would deliver visits twice and lose each other's.
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    with open(args.archive_dir / LOCK_FILE, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log.error("another collector is already running on %s", args.archive_dir)
            return 1
        return run(args)


def run(args: argparse.Namespace) -> int:
    """Collect and deliver until stopped, or for one pass with --once."""
    client = CameraClient(args.camera)
    # What the page may change while this runs is read again at every pass.
    settings_dir = args.selector_dir / "collector"
    started_with = live_from(args)
    recent = RecentLog(settings_dir / LOG_FILE)
    logging.getLogger().addHandler(recent)
    live, handled = read_live(settings_dir / SETTINGS_FILE, started_with)
    refused: Optional[Live] = None
    problem: Optional[str] = None
    try:
        deliver = sender_for(args, live)
    except ValueError as error:
        # What the page saved cannot be used: start as the environment says, and let the page
        # show why. Only starting values that cannot be used either end the collector.
        refused, live, problem = live, started_with, str(error)
        try:
            deliver = sender_for(args, live)
        except ValueError as fatal:
            build_parser().error(str(fatal))
        log.error("settings not taken up: %s", problem)
    selector = ModelSelector(args.selector_dir)
    # With a delivery interval, visits are handed to the outbox and leave from there; on
    # their way out they are checked against the photos delivered in the hours before.
    # Without one the outbox holds only what was collected while paused, and what was waiting
    # when the interval was removed.
    outbox = Outbox(
        args.archive_dir,
        deliver,
        (live.deliver_every_minutes or 0) * 60,
        sifter=RepeatSifter(selector),
    )
    test: Optional[dict] = None
    reported: Optional[tuple] = None
    reported_at = 0.0

    def sink() -> Sender:
        """Where a collected visit goes: straight out, or to wait in the outbox. A dry run
        leaves nothing waiting that a later run would deliver."""
        if args.dry_run:
            return deliver
        return outbox.add if live.paused or live.deliver_every_minutes is not None else deliver

    def report(now_problem: Optional[str] = None) -> None:
        nonlocal problem, reported, reported_at
        problem = now_problem
        reported = (outbox.waiting(), outbox.next_due_at())
        reported_at = time.monotonic()
        write_status(
            settings_dir / STATUS_FILE,
            live,
            args.deliver_command,
            args.dry_run,
            problem,
            test,
            camera=args.camera,
            waiting=outbox.waiting(),
            next_delivery_at=None if live.paused else outbox.next_due_at(),
        )

    report(problem)
    aimer = Aimer(client, args.archive_dir, args.selector_dir / "views", DETECTION_RESOLUTION)
    retries = Retries()
    reachability = Reachability()
    health = CaptureHealth()
    pruned_at: Optional[float] = None
    pruned_days: Optional[float] = None

    while True:
        wanted, asked = read_live(settings_dir / SETTINGS_FILE, started_with)
        if wanted != live and wanted != refused:
            try:
                deliver = sender_for(args, wanted)
            except ValueError as error:
                # Deliveries carry on as they were; the page shows why.
                refused = wanted
                log.error("settings not taken up: %s", error)
                report(str(error))
            else:
                live, refused = wanted, None
                outbox.retarget(deliver, (live.deliver_every_minutes or 0) * 60)
                log.info(
                    "settings changed: %sdeliver by %s%s, caption name %s, time zone %s",
                    "paused, " if live.paused else "",
                    live.deliver,
                    (
                        f" every {live.deliver_every_minutes:g} min"
                        if live.deliver_every_minutes is not None
                        else ""
                    ),
                    live.subject,
                    live.timezone,
                )
                report()
        elif wanted == live and refused is not None:
            # The page went back to what is in use.
            refused = None
            report()
        if asked["test_at"] is not None and asked["test_at"] != handled["test_at"]:
            handled["test_at"] = asked["test_at"]
            test = send_test(deliver, args.archive_dir, settings_dir / "test", asked["test_at"])
            report(problem)
        send_now = (
            asked["send_now_at"] is not None and asked["send_now_at"] != handled["send_now_at"]
        )
        handled["send_now_at"] = asked["send_now_at"]
        if live.keep_visits_days is not None and (
            live.keep_visits_days != pruned_days
            or time.monotonic() - pruned_at >= PRUNE_EVERY_SECONDS
        ):
            pruned_at, pruned_days = time.monotonic(), live.keep_visits_days
            try:
                visits, freed = prune_visits(
                    args.archive_dir,
                    live.keep_visits_days,
                    time.time(),
                    keep=outbox.visit_dirs(),
                )
            except Exception:
                # An archive that cannot be read today is no reason to stop collecting.
                log.exception("archive: pruning failed")
                visits = 0
            if visits:
                log.info(
                    "archive: deleted %d visits collected more than %g days ago, %.1f GB",
                    visits,
                    live.keep_visits_days,
                    freed / 1e9,
                )
        collected = 0
        try:
            # One visit at a time: a visit can take minutes to fetch, and the settings, the
            # deliveries and the camera's pose must not wait for a whole backlog.
            collected = run_once(
                client,
                args.archive_dir,
                args.every,
                sink(),
                live.zone,
                live.subject,
                mirror=args.mirror,
                retries=retries,
                selector=selector,
                limit=None if args.once else 1,
            )
            status = client.status()
            health.report(status)
            if not args.once:
                keep_aim(aimer, status)
            reachability.reachable()
        except CameraUnreachable as error:
            reachability.unreachable(error)
        except CameraError as error:
            log.error("camera gave an unusable answer: %s", error)
        except Exception:
            # Nothing that goes wrong in one pass should end the service.
            log.exception("collection pass failed")
        if not live.paused and not args.dry_run:
            if send_now:
                log.info("sending the %d waiting visits now, as asked", outbox.waiting())
            deliver_waiting(outbox, at_once=send_now)
        if (
            reported != (outbox.waiting(), outbox.next_due_at())
            or time.monotonic() - reported_at >= STATUS_EVERY_SECONDS
        ):
            report(problem)
        if args.once:
            log.info("one pass done: %d visit(s) collected", collected)
            return 0
        if not collected:
            # After a visit there may be another waiting; look straight away.
            time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())

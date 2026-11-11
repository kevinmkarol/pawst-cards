# ABOUTME: Holds collected visits back so deliveries go out at most once per set interval: the
# ABOUTME: waiting visits leave together as one delivery, their photos side by side in one picture.

import json
import logging
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image

from collector.deliver import Delivery, Sender
from collector.files import write_whole
from collector.repeats import DELIVERED_FILE, RepeatSifter

log = logging.getLogger("collector")

# The outbox's own file in the archive directory: the visits waiting and when the last
# delivery went out, so neither is lost when the collector restarts.
OUTBOX_FILE = "outbox.json"
# Most photos one picture holds. With more visits waiting, the latest ones are pictured.
MOST_PHOTOS = 12
# Width in pixels the combined picture aims for; each photo gets an equal share of it.
SHEET_WIDTH = 2560
# Seconds before a delivery that failed is tried again.
RETRY_SECONDS = 300.0


def contact_sheet(photos: Sequence[Path], out: Path) -> Path:
    """One JPEG with `photos` in a grid, in reading order: left to right, top to bottom. Every
    photo is scaled to fit a cell shaped like the first one."""
    columns = math.ceil(math.sqrt(len(photos)))
    rows = math.ceil(len(photos) / columns)
    with Image.open(photos[0]) as first:
        cell_width = SHEET_WIDTH // columns
        cell_height = round(cell_width * first.height / first.width)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height))
    for index, photo in enumerate(photos):
        with Image.open(photo) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_width, cell_height))
            left = (index % columns) * cell_width + (cell_width - image.width) // 2
            top = (index // columns) * cell_height + (cell_height - image.height) // 2
            sheet.paste(image, (left, top))
    sheet.save(out, quality=88)
    return out


class Outbox:
    """Collected visits waiting to be delivered. `add` is the `Sender` a visit is handed to;
    `flush` delivers what is waiting through `send` when the last delivery was at least
    `every_seconds` ago. A single waiting visit goes out as it is. Several go out as one
    delivery: a picture of their photos, their captions as lines of one caption, and a
    directory of its own in the archive holding that picture and the visits' records.

    With a `sifter`, the waiting visits are first weighed against the photos delivered in
    the hours before: some get another frame of theirs, and those that only repeat what was
    posted are dropped without being delivered."""

    def __init__(
        self,
        archive_dir: Path,
        send: Sender,
        every_seconds: float,
        clock: Callable[[], float] = time.time,
        sifter: Optional[RepeatSifter] = None,
    ) -> None:
        self._archive_dir = archive_dir
        self._send = send
        self._every_seconds = every_seconds
        self._clock = clock
        self._path = archive_dir / OUTBOX_FILE
        self._pending: List[Delivery] = []
        self._last_sent_at: Optional[float] = None
        self._retry_at = 0.0
        self._sifter = sifter
        # The photos delivered lately and when, for the sifter to compare the next ones with.
        self._posted: List[Tuple[float, Path]] = []
        self._load()

    def retarget(self, send: Sender, every_seconds: float) -> None:
        """Deliver through `send`, at most once in `every_seconds`, from now on. With zero
        seconds whatever is waiting leaves at the next flush."""
        self._send = send
        self._every_seconds = every_seconds
        # A delivery that failed is worth another try at once through what may be a new sink.
        self._retry_at = 0.0

    def add(self, delivery: Delivery) -> None:
        # A collector stopped between this and marking the visit collects it again.
        self._pending = [d for d in self._pending if d.visit_dir != delivery.visit_dir]
        self._pending.append(delivery)
        self._save()

    def waiting(self) -> int:
        return len(self._pending)

    def visit_dirs(self) -> List[Path]:
        """The archive directories of the visits waiting, which must outlive the archive's
        keep time until they have been delivered."""
        return [delivery.visit_dir for delivery in self._pending]

    def next_due_at(self) -> Optional[float]:
        """When the waiting visits are due to leave; None when nothing waits."""
        if not self._pending:
            return None
        return max(self._retry_at, (self._last_sent_at or 0.0) + self._every_seconds)

    def flush(self, at_once: bool = False) -> int:
        """Deliver the waiting visits if a delivery is due, or `at_once` whether or not one
        is; returns how many went. A delivery that fails raises, keeps its visits waiting,
        and is tried again a few minutes later."""
        now = self._clock()
        # Visits whose photo has since been deleted from the archive have nothing to show.
        kept = [delivery for delivery in self._pending if delivery.photo.is_file()]
        if len(kept) != len(self._pending):
            log.warning(
                "outbox: %d waiting visits lost their photo", len(self._pending) - len(kept)
            )
            self._pending = kept
            self._save()
        if not self._pending or not (at_once or (now >= self._retry_at and self._due(now))):
            return 0
        batch = list(self._pending)
        shown: List[Path] = []
        if self._sifter is not None:
            batch, shown = self._sifted(batch, now)
            if not batch:
                return 0
        batch_dir = self._batch_dir(now)
        try:
            delivery = batch[0] if len(batch) == 1 else self._combined(batch, batch_dir)
            self._send(delivery)
        except Exception:
            self._retry_at = now + RETRY_SECONDS
            # The next attempt draws its picture afresh, perhaps of more visits.
            shutil.rmtree(batch_dir, ignore_errors=True)
            raise
        if len(batch) > 1:
            # With a record and this marker the archive's keep time applies to the directory.
            (delivery.visit_dir / DELIVERED_FILE).write_text(
                datetime.now().astimezone().isoformat() + "\n"
            )
        self._pending = self._pending[len(batch) :]
        self._last_sent_at = now
        self._posted.extend((now, photo) for photo in shown)
        self._save()
        return len(batch)

    def _sifted(self, batch: List[Delivery], now: float) -> Tuple[List[Delivery], List[Path]]:
        """The waiting visits that do not repeat what was delivered lately, and the photo
        each shows. The visits dropped stop waiting. A sifter that breaks drops nothing."""
        window = self._sifter.window_seconds()
        self._posted = [(at, photo) for at, photo in self._posted if 0 <= now - at < window]
        try:
            sifted = self._sifter.sift(batch, [photo for _, photo in self._posted])
        except Exception:
            log.exception("outbox: checking for repeated photos failed; delivering them all")
            return batch, []
        if len(sifted.kept) != len(batch):
            self._pending = list(sifted.kept)
            self._save()
        return sifted.kept, sifted.shown

    def _due(self, now: float) -> bool:
        if self._last_sent_at is None:
            return True
        # A last delivery in the future means the clock was set back; do not wait on it.
        return not 0 <= now - self._last_sent_at < self._every_seconds

    def _batch_dir(self, now: float) -> Path:
        stamp = datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S")
        return self._archive_dir / f"batch-{stamp}"

    def _combined(self, batch: Sequence[Delivery], batch_dir: Path) -> Delivery:
        pictured = batch[-MOST_PHOTOS:]
        batch_dir.mkdir(parents=True, exist_ok=True)
        photo = contact_sheet(
            [delivery.photo for delivery in pictured],
            batch_dir / f"{pictured[0].photo.stem}_and_{len(pictured) - 1}_more.jpg",
        )
        records = []
        for delivery in batch:
            try:
                records.append(json.loads(delivery.visit_json.read_text()))
            except (OSError, ValueError):
                continue
        (batch_dir / "visit.json").write_text(json.dumps({"visits": records}, indent=1))
        if len(pictured) < len(batch):
            heading = f"{len(batch)} visits; the latest {len(pictured)} are pictured, oldest first:"
        else:
            heading = f"{len(batch)} visits, oldest first:"
        return Delivery(
            caption="\n".join([heading, *(delivery.caption for delivery in batch)]),
            photo=photo,
            visit_dir=batch_dir,
            clips=[clip for delivery in batch for clip in delivery.clips],
        )

    def _load(self) -> None:
        try:
            state = json.loads(self._path.read_text())
            self._last_sent_at = state["last_sent_at"]
            self._posted = [
                (float(entry["at"]), Path(entry["photo"])) for entry in state.get("posted", [])
            ]
            self._pending = [
                Delivery(
                    caption=entry["caption"],
                    photo=Path(entry["photo"]),
                    visit_dir=Path(entry["visit_dir"]),
                    clips=[Path(clip) for clip in entry["clips"]],
                )
                for entry in state["pending"]
            ]
        except FileNotFoundError:
            return
        except (OSError, ValueError, KeyError, TypeError) as error:
            # Kept beside it for a look later; the next save would overwrite it.
            aside = self._path.with_name(self._path.name + ".unreadable")
            log.error("outbox: %s is unreadable (%s); moved to %s", self._path, error, aside)
            self._last_sent_at, self._posted, self._pending = None, [], []
            try:
                self._path.replace(aside)
            except OSError:
                pass

    def _save(self) -> None:
        state = {
            "last_sent_at": self._last_sent_at,
            "posted": [{"at": at, "photo": str(photo)} for at, photo in self._posted],
            "pending": [
                {
                    "caption": delivery.caption,
                    "photo": str(delivery.photo),
                    "visit_dir": str(delivery.visit_dir),
                    "clips": [str(clip) for clip in delivery.clips],
                }
                for delivery in self._pending
            ],
        }
        write_whole(self._path, json.dumps(state, indent=1))

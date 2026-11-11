# ABOUTME: Keeps the collector's archive from growing without bound: deletes each archived visit
# ABOUTME: (clips, frames, photo and record) a set number of days after it was collected.

import logging
import shutil
from pathlib import Path
from typing import Collection, Tuple

from collector.process import SOURCE_FILE
from collector.repeats import DELIVERED_FILE

log = logging.getLogger("collector")

SECONDS_PER_DAY = 86400.0


def _finished_with(visit_dir: Path) -> bool:
    """Whether the collector is done with this visit, so nothing in its directory will be wanted
    for another attempt. A directory written before delivery markers existed has a record and
    no source."""
    if not (visit_dir / "visit.json").is_file():
        return False
    return (visit_dir / DELIVERED_FILE).is_file() or not (visit_dir / SOURCE_FILE).is_file()


def prune_visits(
    archive_dir: Path, keep_days: float, now: float, keep: Collection[Path] = ()
) -> Tuple[int, int]:
    """Delete every archived visit collected more than `keep_days` before `now`; returns how
    many visits went and how many bytes that freed.

    A visit's age is counted from when it was collected (when visit.json was written), not from
    the camera's time for it, which is wrong for visits recorded before the camera's clock was
    set. Visits still being collected, the directories in `keep` (visits waiting in the
    outbox for their delivery), and anything that is not a visit, are left.
    """
    pruned, freed = 0, 0
    if not archive_dir.is_dir():
        return pruned, freed
    for visit_dir in sorted(archive_dir.iterdir()):
        if visit_dir in keep or not visit_dir.is_dir() or not _finished_with(visit_dir):
            continue
        try:
            collected_at = (visit_dir / "visit.json").stat().st_mtime
            if now - collected_at <= keep_days * SECONDS_PER_DAY:
                continue
            size = sum(path.stat().st_size for path in visit_dir.rglob("*") if path.is_file())
            shutil.rmtree(visit_dir)
        except OSError as error:
            log.warning("could not delete archived visit %s: %s", visit_dir.name, error)
            continue
        pruned += 1
        freed += size
    return pruned, freed

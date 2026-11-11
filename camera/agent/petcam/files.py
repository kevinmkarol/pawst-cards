# ABOUTME: Saves and removes the agent's state files so that a power cut part way leaves the old one:
# ABOUTME: the text goes to a new file, is forced to storage, and then takes the old one's place.

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def save_whole(path: Path, text: str) -> bool:
    """Replace `path` with `text`. A failure, most likely a full disk, is logged and answered
    with False: what was to be saved is still held in memory, and the agent carries on."""
    scratch = path.with_name(path.name + ".new")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with scratch.open("w") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(scratch, path)
    except OSError as error:
        log.error("could not save %s: %s", path, error)
        return False
    return True


def remove(path: Path) -> None:
    """Delete `path` if it is there. A failure is logged, and the agent carries on."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as error:
        log.error("could not remove %s: %s", path, error)

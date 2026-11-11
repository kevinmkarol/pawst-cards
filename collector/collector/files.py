# ABOUTME: Writes a file so that a power cut or a kill part way leaves the previous version:
# ABOUTME: the text goes to a scratch file beside it, which then takes the file's place.

import os
from pathlib import Path


def write_whole(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".new")
    with open(scratch, "w") as out:
        out.write(text)
        # Forced to storage first: otherwise a power cut can leave the new name on an empty file.
        out.flush()
        os.fsync(out.fileno())
    os.replace(scratch, path)

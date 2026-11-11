# ABOUTME: Delivery sinks for a collected visit: run a command template with the photo and caption
# ABOUTME: as arguments, or POST them as multipart form data to a webhook URL.

import logging
import mimetypes
import re
import shlex
import shutil
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple


@dataclass(frozen=True)
class Delivery:
    caption: str
    photo: Path
    visit_dir: Path
    clips: List[Path]

    @property
    def visit_json(self) -> Path:
        return self.visit_dir / "visit.json"


Sender = Callable[[Delivery], None]

PLACEHOLDERS = ("caption", "photo", "visit_dir")
# Longest a delivery command may run. One that hangs would otherwise stop every later visit.
COMMAND_TIMEOUT_SECONDS = 300
# A staged copy of a photo is removed once it is this old; the archive keeps the original.
KEEP_STAGED_SECONDS = 7 * 86400
# The names process.photo_name gives photos, and outbox.py the picture of several; nothing
# else in the stage directory is touched.
STAGED_NAME = re.compile(r".+_\d{4}-\d{2}-\d{2}_\d{1,2}-\d{2}[ap]m(_and_\d+_more)?\.jpg")

log = logging.getLogger(__name__)


def failure(error: Exception) -> str:
    """What went wrong, for a log line: of a program that failed, its name and what it said."""
    if isinstance(error, subprocess.CalledProcessError):
        return f"{error.cmd[0]} failed: {str(error.stderr or error.output).strip()}"
    return str(error)


def clear_staged(stage_dir: Path, now: float) -> None:
    """Remove the copies of photos staged more than KEEP_STAGED_SECONDS ago."""
    try:
        for staged in stage_dir.iterdir():
            if (
                STAGED_NAME.fullmatch(staged.name)
                and staged.is_file()
                and now - staged.stat().st_mtime > KEEP_STAGED_SECONDS
            ):
                staged.unlink()
    except OSError as error:
        log.warning("could not clear old staged photos from %s: %s", stage_dir, error)


def command_sender(template: str, stage_dir: Optional[Path] = None) -> Sender:
    """A sender that runs `template`, a shell-style command line in which `{caption}`, `{photo}`
    and `{visit_dir}` each expand to one whole argument. With `stage_dir`, the photo is copied
    there first and the copy's path is what `{photo}` expands to, for tools that only read files
    from their own directories; copies older than a week are removed as new ones are made."""
    tokens = shlex.split(template)
    probe = {name: "" for name in PLACEHOLDERS}
    for token in tokens:
        try:
            token.format_map(probe)
        except KeyError as error:
            raise ValueError(f"unknown placeholder {error} in delivery command") from None

    def send(delivery: Delivery) -> None:
        photo = delivery.photo
        if stage_dir is not None:
            stage_dir.mkdir(parents=True, exist_ok=True)
            photo = stage_dir / delivery.photo.name
            shutil.copyfile(delivery.photo, photo)
            clear_staged(stage_dir, time.time())
        values = {
            "caption": delivery.caption,
            "photo": str(photo),
            "visit_dir": str(delivery.visit_dir),
        }
        command = [token.format_map(values) for token in tokens]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            # Named by its program alone: the error is logged and shown on the page, and the
            # arguments may hold a token.
            raise TimeoutError(
                f"{command[0]} did not finish in {COMMAND_TIMEOUT_SECONDS} s"
            ) from None

    return send


def webhook_sender(url: str, include_clips: bool = False) -> Sender:
    """A sender that POSTs the caption, the visit's JSON record and the photo to `url` as
    multipart/form-data fields `caption`, `visit` and `photo`; with `include_clips`, each clip
    is added as a `clip` file field as well."""

    def send(delivery: Delivery) -> None:
        files = [("photo", delivery.photo)]
        if include_clips:
            files.extend(("clip", clip) for clip in delivery.clips)
        body, content_type = multipart(
            fields=[("caption", delivery.caption.encode())],
            json_fields=[("visit", delivery.visit_json.read_bytes())],
            files=files,
        )
        request = urllib.request.Request(
            url, data=body, method="POST", headers={"Content-Type": content_type}
        )
        with urllib.request.urlopen(request, timeout=120):
            pass

    return send


def multipart(
    fields: List[Tuple[str, bytes]],
    json_fields: List[Tuple[str, bytes]],
    files: List[Tuple[str, Path]],
) -> Tuple[bytes, str]:
    """Encodes text fields, JSON fields and files as a multipart/form-data body; returns the body
    and the Content-Type header value carrying its boundary."""
    boundary = uuid.uuid4().hex
    parts: List[bytes] = []
    for name, value in fields:
        parts.append(
            _part(boundary, f'form-data; name="{name}"', "text/plain; charset=utf-8", value)
        )
    for name, value in json_fields:
        parts.append(_part(boundary, f'form-data; name="{name}"', "application/json", value))
    for name, path in files:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        disposition = f'form-data; name="{name}"; filename="{path.name}"'
        parts.append(_part(boundary, disposition, content_type, path.read_bytes()))
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _part(boundary: str, disposition: str, content_type: str, payload: bytes) -> bytes:
    headers = (
        f"--{boundary}\r\n"
        f"Content-Disposition: {disposition}\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    )
    return headers.encode() + payload + b"\r\n"

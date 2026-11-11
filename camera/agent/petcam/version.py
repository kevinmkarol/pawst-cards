# ABOUTME: A fingerprint of the agent's own code, so the collector's side can tell whether
# ABOUTME: the camera runs the same code as the copy of the repository it has.

import hashlib
from pathlib import Path


def fingerprint(package: Path = Path(__file__).parent) -> str:
    """A short hash of the names and contents of the Python files in `package`."""
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]

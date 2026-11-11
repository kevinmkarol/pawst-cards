# ABOUTME: Tests the fingerprint the agent reports of its own code.

from pathlib import Path

from petcam.version import fingerprint


def test_the_fingerprint_follows_the_code_and_nothing_else(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    first = fingerprint(tmp_path)
    assert len(first) == 16
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "notes.txt").write_text("not code")
    assert fingerprint(tmp_path) == first
    (tmp_path / "b.py").write_text("y = 3\n")
    assert fingerprint(tmp_path) != first
    (tmp_path / "b.py").write_text("y = 2\n")
    (tmp_path / "b.py").rename(tmp_path / "c.py")
    assert fingerprint(tmp_path) != first

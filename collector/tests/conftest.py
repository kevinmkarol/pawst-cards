# ABOUTME: Makes the camera agent package importable so collector tests can run against a real
# ABOUTME: in-process copy of the agent's HTTP API instead of a mock.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "camera" / "agent"))

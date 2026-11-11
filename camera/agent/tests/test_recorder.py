# ABOUTME: Tests the recorder adapter that switches the camera's save node on and off over MQTT.
# ABOUTME: Captures the published messages instead of needing the camera.

import json
from typing import List, Tuple

from petcam.recorder import SaveNodeRecorder
from petcam.sscma import SAVE_NODE_ID, in_topic


def test_set_recording_publishes_enabled_messages() -> None:
    published: List[Tuple[str, bytes]] = []
    recorder = SaveNodeRecorder(publish=lambda topic, payload: published.append((topic, payload)))
    recorder.set_recording(True)
    recorder.set_recording(False)
    assert [topic for topic, _ in published] == [in_topic(SAVE_NODE_ID)] * 2
    assert [json.loads(payload)["data"] for _, payload in published] == [True, False]

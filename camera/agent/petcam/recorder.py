# ABOUTME: Switches the camera's save node on and off so clips are written only during a pet visit.
# ABOUTME: Publishes through whatever MQTT publish function it is given.

from typing import Callable

from petcam.sscma import SAVE_NODE_ID, enabled_message, in_topic

Publish = Callable[[str, bytes], None]


class SaveNodeRecorder:
    def __init__(self, publish: Publish) -> None:
        self.publish = publish

    def set_recording(self, enabled: bool) -> None:
        self.publish(in_topic(SAVE_NODE_ID), enabled_message(enabled).encode())

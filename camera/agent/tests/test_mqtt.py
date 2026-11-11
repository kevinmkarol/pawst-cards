# ABOUTME: Tests the stdlib MQTT client against the camera's real mosquitto broker.
# ABOUTME: Skipped with a reason unless PETCAM_BROKER names a camera reachable from this machine.

import os
import socket
import time
import uuid

import pytest

from petcam.mqtt import MqttClient

BROKER = os.environ.get("PETCAM_BROKER")


def broker_reachable() -> bool:
    if not BROKER:
        return False
    try:
        with socket.create_connection((BROKER, 1883), timeout=3):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not broker_reachable(),
    reason=(
        f"camera broker {BROKER}:1883 not reachable"
        if BROKER
        else "PETCAM_BROKER not set; no camera broker to test against"
    ),
)


def test_publish_is_received_on_subscribed_topic() -> None:
    topic = f"petcam/test/{uuid.uuid4()}"
    received = []
    client = MqttClient(BROKER, 1883, on_message=lambda t, p: received.append((t, p)))
    client.connect()
    try:
        client.subscribe(topic)
        client.publish(topic, b'{"hello": 1}')
        deadline = time.time() + 5
        while not received and time.time() < deadline:
            time.sleep(0.05)
    finally:
        client.close()
    assert received == [(topic, b'{"hello": 1}')]


def test_messages_on_other_topics_are_not_delivered() -> None:
    mine = f"petcam/test/{uuid.uuid4()}"
    other = f"petcam/test/{uuid.uuid4()}"
    received = []
    client = MqttClient(BROKER, 1883, on_message=lambda t, p: received.append((t, p)))
    client.connect()
    try:
        client.subscribe(mine)
        client.publish(other, b"x")
        client.publish(mine, b"y")
        deadline = time.time() + 5
        while not received and time.time() < deadline:
            time.sleep(0.05)
    finally:
        client.close()
    assert received == [(mine, b"y")]


def test_large_payload_round_trips() -> None:
    # sscma detection messages with a debug image exceed 127 bytes, exercising the multi-byte
    # length.
    topic = f"petcam/test/{uuid.uuid4()}"
    payload = b"x" * 5000
    received = []
    client = MqttClient(BROKER, 1883, on_message=lambda t, p: received.append(p))
    client.connect()
    try:
        client.subscribe(topic)
        client.publish(topic, payload)
        deadline = time.time() + 5
        while not received and time.time() < deadline:
            time.sleep(0.05)
    finally:
        client.close()
    assert received == [payload]

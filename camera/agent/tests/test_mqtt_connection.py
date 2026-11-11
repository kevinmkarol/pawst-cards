# ABOUTME: Tests how the MQTT client copes with trouble, against a real socket that speaks just
# ABOUTME: enough of the protocol: a handler that raises, and a broker that goes away.

import socket
import struct
import threading
import time
from typing import Callable, List, Tuple

from petcam.mqtt import CONNACK, PUBLISH, MqttClient


def publish_packet(topic: str, payload: bytes) -> bytes:
    body = struct.pack(">H", len(topic)) + topic.encode() + payload
    return bytes([PUBLISH, len(body)]) + body


def broker(script: Callable[[socket.socket], None]) -> Tuple[int, threading.Thread]:
    """Listens on a free port, accepts one client, acknowledges its CONNECT and runs `script`
    with the connection before closing it."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve() -> None:
        connection, _ = listener.accept()
        with connection, listener:
            connection.recv(1024)
            connection.sendall(bytes([CONNACK, 2, 0, 0]))
            script(connection)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return listener.getsockname()[1], thread


def wait_for(condition: Callable[[], bool], seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_a_handler_that_raises_does_not_stop_later_messages() -> None:
    hold = threading.Event()

    def script(connection: socket.socket) -> None:
        connection.sendall(publish_packet("t", b"bad") + publish_packet("t", b"good"))
        hold.wait(5)

    received: List[bytes] = []

    def handler(topic: str, payload: bytes) -> None:
        if payload == b"bad":
            raise ValueError("cannot handle this one")
        received.append(payload)

    port, _ = broker(script)
    client = MqttClient("127.0.0.1", port, on_message=handler)
    client.connect()
    try:
        assert wait_for(lambda: received == [b"good"])
        assert not client.lost.is_set()
    finally:
        hold.set()
        client.close()


def test_a_broker_that_goes_away_is_reported_as_lost() -> None:
    port, thread = broker(lambda connection: None)
    client = MqttClient("127.0.0.1", port)
    client.connect()
    try:
        thread.join(5)
        assert wait_for(client.lost.is_set)
    finally:
        client.close()


def test_closing_the_client_is_not_a_lost_connection() -> None:
    hold = threading.Event()
    port, _ = broker(lambda connection: hold.wait(5))
    client = MqttClient("127.0.0.1", port)
    client.connect()
    client.close()
    hold.set()
    time.sleep(0.2)
    assert not client.lost.is_set()


def test_a_silent_broker_is_reported_as_lost_after_the_keepalive() -> None:
    hold = threading.Event()
    port, _ = broker(lambda connection: hold.wait(10))
    client = MqttClient("127.0.0.1", port, keepalive_seconds=1)
    client.connect()
    try:
        assert wait_for(client.lost.is_set, seconds=5)
    finally:
        hold.set()
        client.close()

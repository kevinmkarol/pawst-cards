# ABOUTME: Minimal MQTT 3.1.1 client (QoS 0 only) on the standard library, enough to talk to the
# ABOUTME: camera's local mosquitto broker: connect, subscribe, publish, and receive on a thread.

import logging
import socket
import struct
import threading
import uuid
from typing import Callable, Optional, Tuple

CONNECT, CONNACK, PUBLISH, SUBSCRIBE, SUBACK, PINGREQ, DISCONNECT = (
    0x10,
    0x20,
    0x30,
    0x80,
    0x90,
    0xC0,
    0xE0,
)

MessageHandler = Callable[[str, bytes], None]

log = logging.getLogger(__name__)


def _encode_length(length: int) -> bytes:
    out = bytearray()
    while True:
        byte, length = length % 128, length // 128
        if length:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode_string(text: str) -> bytes:
    data = text.encode()
    return struct.pack(">H", len(data)) + data


def _packet(packet_type: int, body: bytes) -> bytes:
    return bytes([packet_type]) + _encode_length(len(body)) + body


class MqttClient:
    def __init__(
        self,
        host: str,
        port: int = 1883,
        on_message: Optional[MessageHandler] = None,
        keepalive_seconds: int = 60,
    ) -> None:
        self.host = host
        self.port = port
        self.on_message = on_message
        self.keepalive_seconds = keepalive_seconds
        self.sock: Optional[socket.socket] = None
        self._packet_id = 0
        self._send_lock = threading.Lock()
        self._closed = threading.Event()
        # Set when the connection to the broker ends without close() being called.
        self.lost = threading.Event()

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=10)
        client_id = f"petcam-{uuid.uuid4().hex[:8]}"
        body = _encode_string("MQTT") + bytes([4, 0x02]) + struct.pack(">H", self.keepalive_seconds)
        self._send(_packet(CONNECT, body + _encode_string(client_id)))
        packet_type, payload = self._read_packet()
        if packet_type != CONNACK or payload[1] != 0:
            raise ConnectionError(f"broker refused connection: {payload!r}")
        # The pinger draws an answer from the broker every half keepalive, so a longer silence
        # than this means the connection is dead even if the socket has not noticed.
        self.sock.settimeout(self.keepalive_seconds * 1.5)
        threading.Thread(target=self._reader, name="mqtt-reader", daemon=True).start()
        threading.Thread(target=self._pinger, name="mqtt-pinger", daemon=True).start()

    def subscribe(self, topic: str) -> None:
        self._packet_id = self._packet_id % 65535 + 1
        body = struct.pack(">H", self._packet_id) + _encode_string(topic) + bytes([0])
        self._send(_packet(SUBSCRIBE | 0x02, body))

    def publish(self, topic: str, payload: bytes) -> None:
        self._send(_packet(PUBLISH, _encode_string(topic) + payload))

    def close(self) -> None:
        self._closed.set()
        if self.sock is not None:
            try:
                self._send(_packet(DISCONNECT, b""))
            except OSError:
                pass
            self.sock.close()

    def _send(self, data: bytes) -> None:
        with self._send_lock:
            self.sock.sendall(data)

    def _read_exact(self, count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            chunk = self.sock.recv(count - len(chunks))
            if not chunk:
                raise ConnectionError("broker closed the connection")
            chunks.extend(chunk)
        return bytes(chunks)

    def _read_packet(self) -> Tuple[int, bytes]:
        header = self._read_exact(1)[0]
        length, multiplier = 0, 1
        while True:
            byte = self._read_exact(1)[0]
            length += (byte & 0x7F) * multiplier
            if not byte & 0x80:
                break
            multiplier *= 128
        return header & 0xF0, self._read_exact(length)

    def _reader(self) -> None:
        while not self._closed.is_set():
            try:
                packet_type, payload = self._read_packet()
            except (OSError, ConnectionError) as error:
                if not self._closed.is_set():
                    log.error("connection to the broker lost: %s", error)
                    self.lost.set()
                return
            if packet_type == PUBLISH and self.on_message is not None:
                topic = "?"
                try:
                    topic_length = struct.unpack_from(">H", payload)[0]
                    topic = payload[2 : 2 + topic_length].decode(errors="replace")
                    self.on_message(topic, payload[2 + topic_length :])
                except Exception:
                    # One message that cannot be handled must not end the thread that receives
                    # all the others.
                    log.exception("handling a message on %s failed", topic)

    def _pinger(self) -> None:
        while not self._closed.wait(self.keepalive_seconds / 2):
            try:
                self._send(_packet(PINGREQ, b""))
            except OSError:
                return

# ABOUTME: Drives the gimbal's yaw and pitch motors over SocketCAN with the motors' 8-byte protocol.
# ABOUTME: Frame encoding and reply decoding are pure functions; Gimbal owns the CAN socket.

import logging
import socket
import struct
import time
from typing import Callable, Optional

YAW_ID = 0x141
PITCH_ID = 0x142

CMD_RELATIVE_MOVE = 0xA8
CMD_ABSOLUTE_MOVE = 0xA4
CMD_STATUS_QUERY = 0x94

MAX_SPEED = 65535

# Seconds between log lines about a CAN bus that refuses frames.
SEND_ERROR_LOG_INTERVAL = 60.0
# Seconds between attempts to open a CAN interface that was not there.
OPEN_EVERY_SECONDS = 30.0

log = logging.getLogger(__name__)

# SocketCAN classic frame: can_id (u32), dlc (u8), 3 pad bytes, 8 data bytes.
_CAN_FRAME = struct.Struct("=IB3x8s")


def _move_frame(command: int, degrees: float, speed: int) -> bytes:
    if not 0 <= speed <= MAX_SPEED:
        raise ValueError(f"speed {speed} outside 0..{MAX_SPEED}")
    return struct.pack("<BBHi", command, 0x00, speed, int(round(degrees * 100)))


def relative_move_frame(degrees: float, speed: int) -> bytes:
    """Data bytes asking a motor to turn `degrees` from where it is at `speed` (motor units)."""
    return _move_frame(CMD_RELATIVE_MOVE, degrees, speed)


def absolute_move_frame(degrees: float, speed: int) -> bytes:
    """Data bytes asking a motor to go to the absolute angle `degrees` at `speed` (motor units)."""
    return _move_frame(CMD_ABSOLUTE_MOVE, degrees, speed)


def status_query_frame() -> bytes:
    return bytes([CMD_STATUS_QUERY, 0, 0, 0, 0, 0, 0, 0])


def decode_angle_reply(data: bytes) -> Optional[float]:
    """Motor angle in degrees from a status frame, or None if the frame is not one. A motor at
    exactly 0 degrees answers with the same bytes as the question it was asked."""
    if len(data) != 8 or data[0] != CMD_STATUS_QUERY:
        return None
    return struct.unpack_from("<H", data, 4)[0] / 100


class Gimbal:
    """The CAN interface is opened when it is there. Without it nothing moves and no angle is
    read, and opening it is tried again now and then: recording visits does not depend on it."""

    def __init__(
        self,
        interface: str = "can0",
        reply_timeout: float = 0.5,
        opener: Optional[Callable[[str], socket.socket]] = None,
    ) -> None:
        self.interface = interface
        self.reply_timeout = reply_timeout
        self._opener = opener or _open_can
        self.sock: Optional[socket.socket] = None
        self._open_tried_at: Optional[float] = None
        self._send_error_logged_at: Optional[float] = None
        self._ready()

    def _ready(self) -> bool:
        """Whether the CAN socket is open, opening it if it is time to try."""
        if self.sock is not None:
            return True
        now = time.monotonic()
        if self._open_tried_at is not None and now - self._open_tried_at < OPEN_EVERY_SECONDS:
            return False
        first = self._open_tried_at is None
        self._open_tried_at = now
        try:
            self.sock = self._opener(self.interface)
            self.sock.settimeout(self.reply_timeout)
        except OSError as error:
            self.sock = None
            if first:
                log.error(
                    "CAN interface %s could not be opened (%s); running without the gimbal",
                    self.interface,
                    error,
                )
            return False
        if not first:
            log.info("CAN interface %s is open; the gimbal is in use again", self.interface)
        return True

    def send(self, motor_id: int, data: bytes) -> None:
        """Send one frame to a motor. A bus that refuses it (motors unpowered, interface down)
        is logged and otherwise ignored: recording visits does not depend on the gimbal."""
        if not self._ready():
            return
        try:
            self.sock.send(_CAN_FRAME.pack(motor_id, len(data), data))
        except OSError as error:
            now = time.monotonic()
            if (
                self._send_error_logged_at is None
                or now - self._send_error_logged_at >= SEND_ERROR_LOG_INTERVAL
            ):
                self._send_error_logged_at = now
                log.warning("gimbal did not take a command: %s", error)

    def move(
        self, yaw_degrees: float, yaw_speed: int, pitch_degrees: float, pitch_speed: int
    ) -> None:
        if yaw_degrees:
            self.send(YAW_ID, relative_move_frame(yaw_degrees, yaw_speed))
        if pitch_degrees:
            self.send(PITCH_ID, relative_move_frame(pitch_degrees, pitch_speed))

    def move_to(self, yaw_degrees: float, pitch_degrees: float, speed: int) -> None:
        self.send(YAW_ID, absolute_move_frame(yaw_degrees, speed))
        self.send(PITCH_ID, absolute_move_frame(pitch_degrees, speed))

    def read_angle(self, motor_id: int) -> Optional[float]:
        """Ask one motor where it is; None if it does not answer within the reply timeout."""
        if not self._ready():
            return None
        self.send(motor_id, status_query_frame())
        # A frame of zeros is either the motor saying it is at 0 degrees or another program on
        # the camera asking the same question. The kernel marks frames sent from this computer,
        # and those are passed over; an unmarked one still waits out the timeout for a real
        # answer before it is believed, in case the mark is missing.
        at_zero = False
        deadline = time.monotonic() + self.reply_timeout
        try:
            while True:
                # Other traffic on the bus must not put the answer's deadline off.
                self.sock.settimeout(max(0.001, deadline - time.monotonic()))
                frame, _, flags, _ = self.sock.recvmsg(16)
                if flags & socket.MSG_DONTROUTE:
                    continue
                can_id, dlc, data = _CAN_FRAME.unpack(frame)
                angle = decode_angle_reply(data[:dlc])
                if can_id != motor_id or angle is None:
                    continue
                if angle:
                    return angle
                at_zero = True
        except socket.timeout:
            return 0.0 if at_zero else None

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()


def _open_can(interface: str) -> socket.socket:
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        sock.bind((interface,))
    except OSError:
        sock.close()
        raise
    return sock

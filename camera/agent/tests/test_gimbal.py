# ABOUTME: Tests for encoding gimbal motor commands as CAN frames and decoding status replies.
# ABOUTME: Byte layouts come from the stock reCamera gimbal nodes and a live yaw motor reply.

import socket
import struct
from typing import List, Tuple

import pytest

from petcam.gimbal import (
    Gimbal,
    absolute_move_frame,
    PITCH_ID,
    YAW_ID,
    decode_angle_reply,
    relative_move_frame,
    status_query_frame,
)


def test_relative_move_frame_layout() -> None:
    # 0xA8, direction 0, speed little-endian, offset in 0.01 degree little-endian int32.
    data = relative_move_frame(degrees=12.34, speed=90)
    assert data == bytes([0xA8, 0x00, 0x5A, 0x00, 0xD2, 0x04, 0x00, 0x00])


def test_relative_move_frame_negative_offset_is_twos_complement() -> None:
    data = relative_move_frame(degrees=-1.0, speed=2000)
    assert data == bytes([0xA8, 0x00, 0xD0, 0x07, 0x9C, 0xFF, 0xFF, 0xFF])


def test_relative_move_frame_rejects_speed_out_of_range() -> None:
    with pytest.raises(ValueError):
        relative_move_frame(degrees=1.0, speed=70000)


def test_status_query_frame() -> None:
    assert status_query_frame() == bytes([0x94, 0, 0, 0, 0, 0, 0, 0])


def test_decode_angle_reply_from_live_yaw_motor() -> None:
    # Captured from the camera: 141 94 00 00 00 9D 8C 00 00 while the yaw motor sat at 359.97
    # degrees.
    assert decode_angle_reply(bytes([0x94, 0, 0, 0, 0x9D, 0x8C, 0, 0])) == pytest.approx(359.97)


def test_decode_angle_reply_ignores_other_commands() -> None:
    assert decode_angle_reply(bytes([0xA8, 0, 0, 0, 0x9D, 0x8C, 0, 0])) is None


def test_motor_ids() -> None:
    assert YAW_ID == 0x141
    assert PITCH_ID == 0x142


def test_absolute_move_frame_layout() -> None:
    # 0xA4, pad, speed 150 little-endian, 225.00 degrees = 22500 hundredths little-endian.
    assert absolute_move_frame(225.0, speed=150) == bytes([0xA4, 0, 0x96, 0x00, 0xE4, 0x57, 0, 0])


def test_absolute_move_frame_rejects_speed_out_of_range() -> None:
    with pytest.raises(ValueError):
        absolute_move_frame(10.0, speed=70000)


def test_move_to_sends_an_absolute_frame_to_each_motor() -> None:
    gimbal = Gimbal.__new__(Gimbal)
    sent = []
    gimbal.send = lambda motor_id, data: sent.append((motor_id, data))  # type: ignore[method-assign]
    gimbal.move_to(yaw_degrees=225.0, pitch_degrees=120.0, speed=90)
    assert sent == [
        (YAW_ID, absolute_move_frame(225.0, speed=90)),
        (PITCH_ID, absolute_move_frame(120.0, speed=90)),
    ]


def test_decode_angle_reply_reads_zero_degrees() -> None:
    assert decode_angle_reply(bytes([0x94, 0, 0, 0, 0, 0, 0, 0])) == 0.0


class FakeCan:
    """A CAN socket that hands out the frames it was given, then times out."""

    def __init__(self, frames: List[Tuple[int, bytes, int]]) -> None:
        self.frames = list(frames)
        self.sent: List[bytes] = []

    def settimeout(self, seconds: float) -> None:
        pass

    def send(self, frame: bytes) -> None:
        self.sent.append(frame)

    def recvmsg(self, size: int) -> Tuple[bytes, list, int, None]:
        if not self.frames:
            raise socket.timeout()
        can_id, data, flags = self.frames.pop(0)
        return struct.pack("=IB3x8s", can_id, len(data), data), [], flags, None

    def close(self) -> None:
        pass


ZERO = bytes([0x94, 0, 0, 0, 0, 0, 0, 0])
AT_90 = bytes([0x94, 0, 0, 0, 0x28, 0x23, 0, 0])


def gimbal_hearing(frames: List[Tuple[int, bytes, int]]) -> Gimbal:
    return Gimbal(opener=lambda interface: FakeCan(frames))  # type: ignore[arg-type,return-value]


def test_a_motor_at_zero_degrees_is_read_as_zero() -> None:
    assert gimbal_hearing([(YAW_ID, ZERO, 0)]).read_angle(YAW_ID) == 0.0


def test_another_program_asking_the_same_question_is_not_an_answer() -> None:
    asked_here = (YAW_ID, ZERO, socket.MSG_DONTROUTE)
    assert gimbal_hearing([asked_here]).read_angle(YAW_ID) is None
    assert gimbal_hearing([asked_here, (YAW_ID, AT_90, 0)]).read_angle(YAW_ID) == 90.0
    # Even unmarked, a frame of zeros gives way to the answer that follows it.
    assert gimbal_hearing([(YAW_ID, ZERO, 0), (YAW_ID, AT_90, 0)]).read_angle(YAW_ID) == 90.0


def test_the_other_motors_answer_is_passed_over() -> None:
    assert gimbal_hearing([(PITCH_ID, AT_90, 0)]).read_angle(YAW_ID) is None


def test_without_a_can_interface_the_gimbal_does_nothing_until_there_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: List[FakeCan] = []
    there = False

    def opener(interface: str) -> FakeCan:
        if not there:
            raise OSError(19, "No such device")
        opened.append(FakeCan([(YAW_ID, AT_90, 0)]))
        return opened[-1]

    clock = [1000.0]
    monkeypatch.setattr("petcam.gimbal.time.monotonic", lambda: clock[0])
    gimbal = Gimbal(opener=opener)  # type: ignore[arg-type]
    gimbal.move_to(10.0, 20.0, speed=90)
    assert gimbal.read_angle(YAW_ID) is None
    there = True
    assert gimbal.read_angle(YAW_ID) is None and not opened
    clock[0] += 31.0
    assert gimbal.read_angle(YAW_ID) == 90.0
    gimbal.close()

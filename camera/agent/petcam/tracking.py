# ABOUTME: Converts where the pet sits in the frame into relative gimbal moves.
# ABOUTME: Proportional control with a dead zone, a per-move angle clamp, and an offset-based speed.

import math
from dataclasses import dataclass
from typing import Tuple

from petcam.detections import Box

# Field of view of the camera module, in degrees, as used by the stock gimbal dashboard.
HORIZONTAL_FOV_DEGREES = 60.0
VERTICAL_FOV_DEGREES = 40.0

# Pixel offsets (in model-input pixels) below the dead zone are treated as centered.
DEAD_ZONE_PX = 5
# Only a fraction of the geometric angle is commanded per update so the gimbal eases onto the
# target.
ANGLE_SCALE = 0.4

MIN_SPEED = 10
MAX_SPEED = 2000


@dataclass(frozen=True)
class Move:
    yaw_degrees: float
    pitch_degrees: float
    yaw_speed: int
    pitch_speed: int

    @property
    def is_motion(self) -> bool:
        return self.yaw_degrees != 0.0 or self.pitch_degrees != 0.0


def pixel_offset_to_degrees(offset_px: float, fov_degrees: float, resolution_px: int) -> float:
    """Angle the gimbal must turn so a point `offset_px` from center lands on center, eased by
    ANGLE_SCALE."""
    if abs(offset_px) <= DEAD_ZONE_PX:
        return 0.0
    half_fov = math.radians(fov_degrees / 2)
    angle = math.degrees(math.atan(offset_px * 2 * math.tan(half_fov) / resolution_px))
    return angle * ANGLE_SCALE


def degrees_to_pixels(degrees: float, fov_degrees: float, resolution_px: int) -> float:
    """How far the picture shifts, in pixels, when the gimbal turns by `degrees`."""
    half_fov = math.radians(fov_degrees / 2)
    return math.tan(math.radians(degrees)) * resolution_px / (2 * math.tan(half_fov))


def speed_for_offset(offset_px: float, max_offset_px: float) -> int:
    """Motor speed that grows faster than linearly with distance, so small corrections stay
    gentle."""
    normalized = min(abs(offset_px) / max_offset_px, 1.0)
    speed = round(normalized**1.5 * (MAX_SPEED - MIN_SPEED) + MIN_SPEED)
    return max(MIN_SPEED, min(MAX_SPEED, speed))


class Tracker:
    def __init__(self, max_angle_degrees: float = 15.0) -> None:
        self.max_angle_degrees = max_angle_degrees

    def move_for(self, target: Box, resolution: Tuple[int, int]) -> Move:
        """Relative yaw/pitch that brings `target` toward the frame center.

        Measured on the gimbal against the sensor's picture, which is horizontally mirrored:
        a positive yaw command brings a target on the right of the frame toward the center and
        a positive pitch command brings one below the center up, so both share the sign of the
        pixel offset.
        """
        width, height = resolution
        offset_x = target.cx - width / 2
        offset_y = target.cy - height / 2
        yaw = self._clamp(pixel_offset_to_degrees(offset_x, HORIZONTAL_FOV_DEGREES, width))
        pitch = self._clamp(pixel_offset_to_degrees(offset_y, VERTICAL_FOV_DEGREES, height))
        return Move(
            yaw_degrees=round(yaw, 2),
            pitch_degrees=round(pitch, 2),
            yaw_speed=speed_for_offset(offset_x, width / 2),
            pitch_speed=speed_for_offset(offset_y, height / 2),
        )

    def _clamp(self, degrees: float) -> float:
        return max(-self.max_angle_degrees, min(self.max_angle_degrees, degrees))

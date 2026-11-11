# ABOUTME: Parses detection messages from the sscma model node and picks the pet to follow.
# ABOUTME: Boxes arrive as [cx, cy, w, h, score, class_id] in the model's input resolution.

import json
from dataclasses import dataclass
from typing import Collection, List, Optional, Sequence, Tuple, Union

# What the YOLO11n model shipped on the camera can see: the COCO classes, in the order of the
# class numbers it reports.
# fmt: off
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
# fmt: on
# The animals followed unless the agent is told otherwise.
DEFAULT_PETS = ("cat", "dog")


def class_ids(pets: Sequence[Union[str, int]]) -> Tuple[int, ...]:
    """The class numbers of `pets`, each given as a COCO name or as its number; ValueError
    naming the first that is neither, or when there are none."""
    ids = []
    for pet in pets:
        if isinstance(pet, bool) or not isinstance(pet, (str, int)):
            raise ValueError("a pet is the name of an animal or its class number")
        word = str(pet).strip().lower()
        if word in COCO_NAMES:
            number = COCO_NAMES.index(word)
        elif word.isdigit() and int(word) < len(COCO_NAMES):
            number = int(word)
        else:
            raise ValueError(
                f"the camera's model does not know {pet!r}; it knows: {', '.join(COCO_NAMES)}"
            )
        if number not in ids:
            ids.append(number)
    if not ids:
        raise ValueError("name at least one animal to follow")
    return tuple(ids)


def class_names(ids: Collection[int]) -> List[str]:
    return [COCO_NAMES[number] for number in ids]


@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    w: float
    h: float
    score: float
    class_id: int

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass(frozen=True)
class Frame:
    boxes: List[Box]
    resolution: Tuple[int, int]
    timestamp_ms: int


def parse_invoke(payload: str) -> Optional[Frame]:
    """Return the detections in one model-node "invoke" message, or None for anything else."""
    try:
        message = json.loads(payload)
        if message.get("name") != "invoke":
            return None
        data = message["data"]
        # The model node reports confidence as a percentage; keep scores as fractions here.
        boxes = [
            Box(cx=b[0], cy=b[1], w=b[2], h=b[3], score=float(b[4]) / 100, class_id=int(b[5]))
            for b in data["boxes"]
        ]
        width, height = data["resolution"]
        return Frame(
            boxes=boxes,
            resolution=(int(width), int(height)),
            timestamp_ms=int(data["timestamp"]),
        )
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return None


def select_target(boxes: List[Box], wanted: Collection[int]) -> Optional[Box]:
    """The largest box of the `wanted` classes, which is the nearest animal when several are in
    view."""
    candidates = [box for box in boxes if box.class_id in wanted]
    if not candidates:
        return None
    return max(candidates, key=lambda box: box.area)

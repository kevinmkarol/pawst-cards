# ABOUTME: Tests for parsing sscma model-node messages and choosing the pet to follow.
# ABOUTME: Uses real message shapes captured from the camera's MQTT broker.

import json

import pytest

from petcam.detections import Box, class_ids, class_names, parse_invoke, select_target

INVOKE_WITH_PET = json.dumps(
    {
        "code": 0,
        "name": "invoke",
        "type": 1,
        "data": {
            "boxes": [
                [100, 120, 50, 40, 91, 0],
                [400, 300, 120, 90, 72, 15],
                [200, 500, 60, 60, 55, 15],
            ],
            "count": 7,
            "image": "",
            "labels": [],
            "perf": [[0, 40, 16]],
            "resolution": [640, 640],
            "timestamp": 1793507767092,
        },
    }
)

CREATE_REPLY = json.dumps({"code": 0, "name": "create", "type": 0, "data": {"classes": ["person"]}})


def test_parse_invoke_scales_percent_scores_to_fractions() -> None:
    # The camera's model node reports scores as percentages (a live pet read 52.0).
    frame = parse_invoke(INVOKE_WITH_PET)
    assert [box.score for box in frame.boxes] == [0.91, 0.72, 0.55]


def test_parse_invoke_returns_boxes_and_resolution() -> None:
    frame = parse_invoke(INVOKE_WITH_PET)
    assert frame is not None
    assert frame.resolution == (640, 640)
    assert frame.timestamp_ms == 1793507767092
    assert frame.boxes[1] == Box(cx=400, cy=300, w=120, h=90, score=0.72, class_id=15)


def test_parse_invoke_ignores_non_invoke_messages() -> None:
    assert parse_invoke(CREATE_REPLY) is None


def test_parse_invoke_ignores_malformed_payload() -> None:
    assert parse_invoke("not json") is None
    assert parse_invoke(json.dumps({"name": "invoke", "data": {"boxes": "nope"}})) is None


def test_select_target_picks_largest_pet() -> None:
    frame = parse_invoke(INVOKE_WITH_PET)
    target = select_target(frame.boxes, [15])
    assert target == Box(cx=400, cy=300, w=120, h=90, score=0.72, class_id=15)


def test_select_target_none_when_no_pet() -> None:
    boxes = [Box(cx=1, cy=1, w=1, h=1, score=0.9, class_id=0)]
    assert select_target(boxes, [15]) is None


def test_select_target_picks_the_largest_of_any_wanted_class() -> None:
    cat = Box(cx=1, cy=1, w=10, h=10, score=0.9, class_id=15)
    dog = Box(cx=1, cy=1, w=30, h=30, score=0.9, class_id=16)
    person = Box(cx=1, cy=1, w=90, h=90, score=0.9, class_id=0)
    assert select_target([cat, dog, person], [15, 16]) == dog


def test_pets_are_named_or_numbered() -> None:
    assert class_ids(["cat", "Dog", "14", 17]) == (15, 16, 14, 17)
    assert class_ids(["cat", "15"]) == (15,)
    assert class_names((15, 16)) == ["cat", "dog"]
    assert class_ids(["teddy bear"]) == (77,)


def test_unknown_or_missing_pets_are_refused() -> None:
    for wrong in (["rabbit"], ["80"], ["-1"], [True], [1.5], []):
        with pytest.raises(ValueError):
            class_ids(wrong)
    with pytest.raises(ValueError, match="rabbit.*it knows: person"):
        class_ids(["rabbit"])

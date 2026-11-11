# ABOUTME: Tests for building sscma-node control messages and topics for the camera pipeline nodes.
# ABOUTME: Message shapes match what the camera accepted over MQTT during manual testing.

import json

from petcam.sscma import (
    created_by_request,
    CAMERA_NODE_ID,
    MODEL_NODE_ID,
    SAVE_NODE_ID,
    STREAM_NODE_ID,
    create_model_message,
    create_save_message,
    enabled_message,
    in_topic,
    out_topic,
)


def test_topics() -> None:
    assert in_topic("abc") == "sscma/v0/recamera/node/in/abc"
    assert out_topic("abc") == "sscma/v0/recamera/node/out/abc"


def test_create_model_message_depends_on_camera() -> None:
    message = json.loads(create_model_message(score_threshold=0.45, iou_threshold=0.25))
    assert message["name"] == "create"
    assert message["type"] == 3
    assert message["data"]["type"] == "model"
    assert message["data"]["config"] == {
        "uri": "",
        "tscore": 0.45,
        "tiou": 0.25,
        "debug": False,
        "trace": False,
        "counting": False,
        "splitter": "0,0,0,0",
    }
    assert message["data"]["dependencies"] == [CAMERA_NODE_ID]
    assert message["data"]["dependents"] == []


def test_create_save_message_starts_disabled_with_slices() -> None:
    message = json.loads(create_save_message(slice_seconds=10))
    assert message["data"]["type"] == "save"
    assert message["data"]["config"] == {
        "storage": "local",
        "slice": 10,
        "duration": 0,
        "enabled": False,
    }
    assert message["data"]["dependencies"] == [CAMERA_NODE_ID]


def test_enabled_message() -> None:
    assert json.loads(enabled_message(True)) == {
        "name": "enabled",
        "type": 3,
        "data": True,
    }
    assert json.loads(enabled_message(False)) == {
        "name": "enabled",
        "type": 3,
        "data": False,
    }


def test_node_ids_are_distinct() -> None:
    assert len({CAMERA_NODE_ID, MODEL_NODE_ID, SAVE_NODE_ID, STREAM_NODE_ID}) == 4


def test_only_a_successful_create_answer_counts_as_the_node_being_created() -> None:
    assert created_by_request(
        b'{"code":0,"data":["width",1920,"height",1080,"fps",15],"name":"create","type":0}'
    )
    assert not created_by_request(
        b'{"code":11,"data":"Node already exists: a914565b1ca6c076","name":"create","type":0}'
    )
    assert not created_by_request(
        b'{"code":7,"data":"resolution mode changed, restarting sscma-node","name":"create"}'
    )
    assert not created_by_request(b'{"code":0,"data":"","name":"health","type":0}')
    assert not created_by_request(b"not json")
    assert not created_by_request(b"[1, 2]")

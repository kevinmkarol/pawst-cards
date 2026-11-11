# ABOUTME: Builds the MQTT control messages and topics that drive the camera's sscma-node daemon.
# ABOUTME: Node ids are fixed so the agent can find the nodes the boot pipeline created.

import json

TOPIC_PREFIX = "sscma/v0/recamera/node"

# The camera and stream nodes are created at boot by start_pipeline.sh; the agent adds the rest.
CAMERA_NODE_ID = "a914565b1ca6c076"
STREAM_NODE_ID = "petcam_rtsp_stream"
MODEL_NODE_ID = "petcam_model"
SAVE_NODE_ID = "petcam_save"


def in_topic(node_id: str) -> str:
    return f"{TOPIC_PREFIX}/in/{node_id}"


def out_topic(node_id: str) -> str:
    return f"{TOPIC_PREFIX}/out/{node_id}"


def _create(node_type: str, config: dict) -> str:
    return json.dumps(
        {
            "name": "create",
            "type": 3,
            "data": {
                "type": node_type,
                "config": config,
                "dependencies": [CAMERA_NODE_ID],
                "dependents": [],
            },
        }
    )


def create_model_message(score_threshold: float, iou_threshold: float) -> str:
    """Create the YOLO detection node; an empty uri loads the model installed in /userdata/MODEL."""
    return _create(
        "model",
        {
            "uri": "",
            "tscore": score_threshold,
            "tiou": iou_threshold,
            "debug": False,
            "trace": False,
            "counting": False,
            "splitter": "0,0,0,0",
        },
    )


def create_save_message(slice_seconds: int) -> str:
    """Create the recorder, idle until enabled; duration 0 records for as long as it stays
    enabled."""
    return _create(
        "save",
        {"storage": "local", "slice": slice_seconds, "duration": 0, "enabled": False},
    )


def created_by_request(payload: bytes) -> bool:
    """Whether a message on a node's out topic says the node has just been created, as opposed
    to "already exists" or any other answer."""
    try:
        message = json.loads(payload)
        return message.get("name") == "create" and message.get("code") == 0
    except (ValueError, AttributeError):
        return False


def enabled_message(enabled: bool) -> str:
    return json.dumps({"name": "enabled", "type": 3, "data": enabled})

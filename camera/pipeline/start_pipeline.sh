#!/bin/sh
# ABOUTME: Builds the reCamera capture + RTSP pipeline by sending create requests to the
# ABOUTME: sscma-node daemon over the camera's local MQTT broker. Idempotent; run at boot.

TOPIC="sscma/v0/recamera/node"
CAMERA_ID="a914565b1ca6c076"
STREAM_ID="petcam_rtsp_stream"
# 0 = 1920x1080@30 (1 Mbit/s), 1 = 1280x720@30, 2 = 640x480@30, 3 = sensor max 2592x1944@15 (4 Mbit/s)
CAMERA_RESOLUTION_OPTION=3
# User name and password of the RTSP stream: the factory's, unless install.sh left others in
# RTSP_LOGIN from PETCAM_RTSP_USER and PETCAM_RTSP_PASSWORD in petcam.env. sscma-node keeps the
# ones a stream was created with, so new ones take effect when the camera next starts.
RTSP_LOGIN=/etc/petcam_rtsp.conf
RTSP_USER=admin
RTSP_PASSWORD=admin
if [ -r "$RTSP_LOGIN" ]; then
    . "$RTSP_LOGIN"
fi
# How many times each request is sent, one to three seconds apart, before giving up on sscma-node.
ATTEMPTS=40

# The red (heartbeat) and blue (storage activity) status LEDs blink constantly; keep them dark.
for led in red blue; do
    echo none > /sys/class/leds/$led/trigger
    echo 0 > /sys/class/leds/$led/brightness
done

# Sends one create request and waits for sscma-node to say the node exists, sending it again
# until it does. Two things make one request not enough:
# - sscma-node drops requests that arrive before it has subscribed to the broker, which is the
#   case for the first seconds after boot and after every restart of it, and nothing else says
#   when it is ready.
# - Asked for a camera resolution other than the one it last ran with, sscma-node answers
#   "resolution mode changed, restarting sscma-node" (code 7), restarts itself and forgets every
#   node. The factory Node-RED flow asks for 1080p whenever sscma-node comes up, so this
#   happens after it has had its way.
# "Node already exists" (code 11) counts as done, so running this script twice is safe.
create() {
    attempt=0
    while [ "$attempt" -lt "$ATTEMPTS" ]; do
        attempt=$((attempt + 1))
        reply="$(
            (sleep 1; printf '%s' "$2" | mosquitto_pub -h localhost -t "$TOPIC/in/$1" -s) &
            mosquitto_sub -h localhost -t "$TOPIC/out/$1" -W 3 2>/dev/null | grep -m 1 '"name":"create"'
        )"
        case "$reply" in
            *'"code":0,'*|*'"code":11,'*)
                echo "$1: $reply"
                return 0
                ;;
            ?*)
                echo "$1: $reply; asking again"
                ;;
        esac
    done
    echo "$1: sscma-node did not create it in $ATTEMPTS attempts" >&2
    return 1
}

# The camera node first: the stream depends on it, and so do the nodes the agent adds.
create "$CAMERA_ID" '{"name":"create","type":3,"data":{"type":"camera","config":{"option":'"$CAMERA_RESOLUTION_OPTION"',"preview":false,"audio":80,"mirror":true},"dependencies":[],"dependents":["'"$STREAM_ID"'"]}}' || exit 1
create "$STREAM_ID" '{"name":"create","type":3,"data":{"type":"stream","config":{"protocol":0,"port":554,"session":"live","username":"'"$RTSP_USER"'","password":"'"$RTSP_PASSWORD"'","audio":true},"dependencies":["'"$CAMERA_ID"'"],"dependents":[]}}' || exit 1

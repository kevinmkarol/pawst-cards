#!/bin/sh
# ABOUTME: Keeps an SSH tunnel open from local port 8555 to the reCamera's RTSP port 554,
# ABOUTME: reconnecting whenever the camera reboots or the link drops.
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../config.sh"
while true; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes -N -L 8555:127.0.0.1:554 "$PETCAM_CAMERA_HOST"
    sleep 5
done

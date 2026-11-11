#!/bin/sh
# ABOUTME: Installs the agent's boot script on the camera and (re)starts the agent from the
# ABOUTME: deployed package with PETCAM_PETS and PETCAM_AGENT_ARGS. Run deploy.sh first; needs sudo there.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../../config.sh"
CAMERA="$PETCAM_CAMERA_HOST"

ARGS="$(mktemp)"
trap 'rm -f "$ARGS"' EXIT
# The animals to follow come first, so a --pets among the flags still wins.
case "$PETCAM_PETS" in
    *[!A-Za-z0-9\ ]*) echo "PETCAM_PETS may hold only animal names or class numbers, separated by spaces" >&2; exit 1 ;;
esac
printf '%s\n' "${PETCAM_PETS:+--pets $PETCAM_PETS }$PETCAM_AGENT_ARGS" > "$ARGS"
scp -q "$ARGS" "$CAMERA:/userdata/petcam/agent.args"
scp -q "$HERE/S96petcam_agent" "$CAMERA:/tmp/"
# The capture pipeline is built before the agent starts, so its status is asked for up to
# 30 seconds after the restart. It is asked on the camera itself at the usual port: with
# --api-host or --api-port among the flags the agent may be running and still not answer here.
ssh "$CAMERA" 'set -e
    chmod 644 /userdata/petcam/agent.args
    sudo cp /tmp/S96petcam_agent /etc/init.d/S96petcam_agent
    sudo chmod 755 /etc/init.d/S96petcam_agent
    rm /tmp/S96petcam_agent
    sudo /etc/init.d/S96petcam_agent restart
    answered=no
    for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        sleep 2
        if wget -qO- http://127.0.0.1:8765/status; then answered=yes; break; fi
    done
    echo
    tail -n 3 /userdata/petcam/agent.log
    [ "$answered" = yes ] || { echo "the agent did not answer within 30 seconds; see /userdata/petcam/agent.log" >&2; exit 1; }'
echo

#!/bin/sh
# ABOUTME: Copies the agent package to the camera under /userdata/petcam so it can run there
# ABOUTME: with the camera's own Python 3.9 and standard library.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../../config.sh"
CAMERA="$PETCAM_CAMERA_HOST"
DEST=/userdata/petcam

ssh "$CAMERA" "sudo mkdir -p $DEST && sudo chown -R recamera $DEST && rm -rf $DEST/petcam"
scp -rq "$HERE/petcam" "$CAMERA:$DEST/petcam"
ssh "$CAMERA" "find $DEST -name __pycache__ -type d -exec rm -rf {} +; ls $DEST/petcam"
echo "deployed to $CAMERA:$DEST"

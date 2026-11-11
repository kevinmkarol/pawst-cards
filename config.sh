#!/bin/sh
# ABOUTME: Loads petcam.env from the repository root into the environment for the deploy and
# ABOUTME: helper scripts. Sourced, not run: `. "$REPO/config.sh"`.
# The repository root is the nearest directory above the calling script that holds the example
# file, so this works in a copy downloaded without git as well as in a clone.
PETCAM_REPO="$(cd "$(dirname "$0")" && pwd)"
while [ ! -f "$PETCAM_REPO/petcam.env.example" ] && [ "$PETCAM_REPO" != "/" ]; do
    PETCAM_REPO="$(dirname "$PETCAM_REPO")"
done
PETCAM_ENV="$PETCAM_REPO/petcam.env"
if [ ! -f "$PETCAM_ENV" ]; then
    echo "missing $PETCAM_ENV; copy petcam.env.example to petcam.env and fill it in" >&2
    exit 1
fi
set -a
. "$PETCAM_ENV"
set +a

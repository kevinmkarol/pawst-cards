#!/bin/sh
# ABOUTME: Starts mediamtx for the live view with the camera stream's user name and password
# ABOUTME: from petcam.env put into a private copy of mediamtx.yml. Run tunnel.sh beside it.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../config.sh"
# The same check camera/pipeline/install.sh makes, so neither value can break the address or sed.
for value in "${PETCAM_RTSP_USER:=admin}" "${PETCAM_RTSP_PASSWORD:=admin}"; do
    case "$value" in
        ""|*[!A-Za-z0-9._~-]*)
            echo "PETCAM_RTSP_USER and PETCAM_RTSP_PASSWORD may hold only letters, digits and . _ ~ -" >&2
            exit 1
            ;;
    esac
done
umask 077
CONFIG="$(mktemp)"
trap 'rm -f "$CONFIG"' EXIT INT TERM
sed "s|rtsp://admin:admin@|rtsp://$PETCAM_RTSP_USER:$PETCAM_RTSP_PASSWORD@|" "$HERE/mediamtx.yml" > "$CONFIG"
mediamtx "$CONFIG"

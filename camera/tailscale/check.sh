#!/bin/sh
# ABOUTME: Verifies the camera's Tailscale install: daemon running, CLI answers, node has a
# ABOUTME: tailnet IP, and the camera's RTSP port is reachable from this Mac over the tailnet.
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../../config.sh"
CAMERA="$PETCAM_CAMERA_HOST"
TS="sudo -n /userdata/tailscale/tailscale --socket=/var/run/tailscale/tailscaled.sock"
fail=0
ssh -o BatchMode=yes "$CAMERA" "pidof tailscaled >/dev/null" && echo "ok   tailscaled running" || { echo "FAIL tailscaled not running"; fail=1; }
IP=$(ssh -o BatchMode=yes "$CAMERA" "$TS ip -4" 2>/dev/null)
[ -n "$IP" ] && echo "ok   tailnet ip $IP" || { echo "FAIL no tailnet ip (not enrolled?)"; fail=1; }
if [ -n "$IP" ]; then
    nc -z -G 5 "$IP" 554 && echo "ok   rtsp :554 reachable over tailnet" || { echo "FAIL rtsp :554 not reachable via $IP"; fail=1; }
fi
exit $fail

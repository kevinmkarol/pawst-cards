#!/bin/sh
# ABOUTME: Copies the Tailscale static riscv64 build and its init script onto the reCamera,
# ABOUTME: starts tailscaled, and prints the enrolment command that needs an auth key.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../../config.sh"
CAMERA="$PETCAM_CAMERA_HOST"
VERSION=1.104.4
# SHA-256 of that version's riscv64 build, as pkgs.tailscale.com publishes it beside the
# download (the same address with .sha256 added). Change both together.
SHA256=6478e7b18e47ac5464e218fe6895e9581683ab8f8bbf008ee396570dc0dc9c7b
NAME="tailscale_${VERSION}_riscv64.tgz"
TARBALL="$HERE/../../tools/tailscale/$NAME"
TS_DIR=/userdata/tailscale

# The build is fetched once and kept in tools/tailscale (ignored by git). Whether it was fetched
# now or was already there, nothing goes to the camera, where it runs as root, unless it is
# the build named above.
if [ ! -f "$TARBALL" ]; then
    mkdir -p "$(dirname "$TARBALL")"
    echo "downloading $NAME"
    curl -fL --proto '=https' --tlsv1.2 -o "$TARBALL.part" "https://pkgs.tailscale.com/stable/$NAME"
    mv "$TARBALL.part" "$TARBALL"
fi
if command -v sha256sum > /dev/null; then
    found="$(sha256sum "$TARBALL" | cut -d ' ' -f 1)"
else
    found="$(shasum -a 256 "$TARBALL" | cut -d ' ' -f 1)"
fi
if [ "$found" != "$SHA256" ]; then
    echo "$TARBALL is not Tailscale $VERSION for riscv64:" >&2
    echo "  sha256 $found, expected $SHA256" >&2
    echo "delete it and run this again to download it afresh" >&2
    exit 1
fi

# Run again on a camera where tailscaled is already going, this leaves the running one alone:
# the boot script's start would add a second restart loop beside the first. The new binaries
# are used from the next boot.
# /tmp on the camera is a 91 MB RAM disk, too small for the tarball plus its contents,
# so the archive goes straight to /userdata (5 GB) and is unpacked in place.
ssh "$CAMERA" "sudo mkdir -p $TS_DIR && sudo chmod 755 $TS_DIR && sudo chown recamera $TS_DIR"
scp "$TARBALL" "$CAMERA:$TS_DIR/"
scp "$HERE/S60tailscale" "$CAMERA:/tmp/"
ssh "$CAMERA" "cd $TS_DIR && tar xzf tailscale_${VERSION}_riscv64.tgz --strip-components=1 tailscale_${VERSION}_riscv64/tailscale tailscale_${VERSION}_riscv64/tailscaled && rm tailscale_${VERSION}_riscv64.tgz && chmod 755 tailscale tailscaled && sudo chown -R root:root $TS_DIR && sudo cp /tmp/S60tailscale /etc/init.d/S60tailscale && sudo chmod 755 /etc/init.d/S60tailscale && rm /tmp/S60tailscale && { ps -o args | grep -q '[S]60tailscale start\|[t]ailscaled --tun' || sudo /etc/init.d/S60tailscale start; } && sleep 3 && ps -o args | grep -q '[t]ailscaled --tun' && $TS_DIR/tailscale --socket=/var/run/tailscale/tailscaled.sock version | head -1"

echo
echo "tailscaled is running. Enrol the camera with your auth key:"
echo "  ssh $CAMERA 'sudo $TS_DIR/tailscale --socket=/var/run/tailscale/tailscaled.sock up --authkey=tskey-auth-XXXX --hostname=recamera --operator=recamera'"

#!/bin/sh
# ABOUTME: Installs the pet cam pipeline start script and its boot init script onto the reCamera
# ABOUTME: over SSH, so the RTSP stream comes back after a power cycle, and sets the stream's login.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../../config.sh"
CAM="$PETCAM_CAMERA_HOST"
# Where start_pipeline.sh reads the stream's user name and password from (RTSP_LOGIN there).
RTSP_LOGIN=/etc/petcam_rtsp.conf

# The stream's own user name and password, when petcam.env sets them. They end up in a JSON
# request, in a file the camera's shell reads and in the stream's address, so they are held to
# characters that mean nothing in any of the three.
plain() {
    case "$2" in
        ""|*[!A-Za-z0-9._~-]*)
            echo "$1 may hold only letters, digits and . _ ~ -" >&2
            exit 1
            ;;
    esac
}
if [ -n "$PETCAM_RTSP_USER$PETCAM_RTSP_PASSWORD" ]; then
    plain PETCAM_RTSP_USER "${PETCAM_RTSP_USER:=admin}"
    plain PETCAM_RTSP_PASSWORD "$PETCAM_RTSP_PASSWORD"
    # Through a pipe into a file only its owner reads: never on a command line, or in a file
    # the camera's other users could see.
    printf 'RTSP_USER=%s\nRTSP_PASSWORD=%s\n' "$PETCAM_RTSP_USER" "$PETCAM_RTSP_PASSWORD" \
        | ssh "$CAM" "umask 077 && cat > /tmp/petcam_rtsp.conf && sudo mv /tmp/petcam_rtsp.conf $RTSP_LOGIN && sudo chown root:root $RTSP_LOGIN && sudo chmod 600 $RTSP_LOGIN"
    echo "stream login set from petcam.env; restart the camera for it to take effect"
else
    ssh "$CAM" "sudo rm -f $RTSP_LOGIN"
    echo "stream login left as the factory's (admin:admin); set PETCAM_RTSP_PASSWORD in petcam.env to change it"
fi

scp "$HERE/start_pipeline.sh" "$HERE/S95petcam_pipeline" "$CAM:/tmp/"
ssh "$CAM" 'sudo mkdir -p /usr/local/bin \
  && sudo cp /tmp/start_pipeline.sh /usr/local/bin/start_pipeline.sh \
  && sudo cp /tmp/S95petcam_pipeline /etc/init.d/S95petcam_pipeline \
  && sudo chmod 755 /usr/local/bin/start_pipeline.sh /etc/init.d/S95petcam_pipeline \
  && rm /tmp/start_pipeline.sh /tmp/S95petcam_pipeline \
  && ls -l /usr/local/bin/start_pipeline.sh /etc/init.d/S95petcam_pipeline'

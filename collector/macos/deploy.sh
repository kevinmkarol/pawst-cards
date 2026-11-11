#!/bin/sh
# ABOUTME: Installs the collector on an always-on Mac as a launchd job: syncs this repository to
# ABOUTME: ~/petcam/repo there, installs dependencies, and (re)loads the job. --local for this Mac.
# With PETCAM_PAGE_HOST set, the selector page is installed as a second job beside it.
set -e
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
. "$HERE/config.sh"
PLIST=com.petcam.collector.plist
PAGE_PLIST=com.petcam.page.plist
RENDERED="$(mktemp)"
RENDERED_PAGE="$(mktemp)"
trap 'rm -f "$RENDERED" "$RENDERED_PAGE"' EXIT

# By default the Mac is reached over SSH as PETCAM_COLLECTOR_HOST. With --local the collector is
# installed on the machine this script runs on, which needs no SSH access to itself and whose
# home directory is known.
if [ "$1" = "--local" ]; then
    DEST="$HOME/"
    COLLECTOR_HOME="$HOME"
    on_collector() { sh -c "$1"; }
else
    : "${PETCAM_COLLECTOR_HOST:?set it in petcam.env, or use --local}"
    : "${PETCAM_COLLECTOR_HOME:?set it in petcam.env, or use --local}"
    DEST="$PETCAM_COLLECTOR_HOST:"
    COLLECTOR_HOME="$PETCAM_COLLECTOR_HOME"
    on_collector() { ssh "$PETCAM_COLLECTOR_HOST" "$1"; }
fi
export COLLECTOR_HOME
# Where the selector keeps its data on the collector's computer; the collector's own default.
SELECTOR_DIR="${PETCAM_SELECTOR_DIR:-$COLLECTOR_HOME/petcam/selector}"
SETUP="export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.local/bin:\$PATH; set -e"

# The commands that replace a running launchd job with the plist just installed, and show
# whether it started.
reload() {
    job="gui/\$(id -u)/${1%.plist}"
    echo "install -m 600 ~/petcam/repo/collector/macos/$1 ~/Library/LaunchAgents/$1; \
        launchctl bootout $job 2>/dev/null || true; \
        for attempt in 1 2 3 4 5 6 7 8 9 10; do \
            launchctl print $job >/dev/null 2>&1 || break; sleep 1; \
        done; \
        launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/$1 \
            || { sleep 5; launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/$1; }; \
        sleep 3; launchctl print $job | grep -E 'state|pid' | head -3"
}

"$HERE/collector/macos/render_plist.sh" > "$RENDERED"

on_collector "mkdir -p ~/petcam/repo ~/petcam/logs ~/Library/LaunchAgents"
# Nothing of petcam.env or its backup copies is sent: the jobs get their values from the plists.
rsync -az --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
    --exclude .ruff_cache --exclude camera/factory_images --exclude camera/stock --exclude tools \
    --include petcam.env.example --exclude 'petcam.env*' \
    "$HERE/" "${DEST}petcam/repo/"
rsync -a "$RENDERED" "${DEST}petcam/repo/collector/macos/$PLIST"
if [ -n "$PETCAM_PAGE_HOST" ]; then
    # The page is given only what it uses: not the delivery command, which may hold a token.
    "$HERE/collector/macos/render_plist.sh" "$HERE/collector/macos/$PAGE_PLIST.template" \
        PETCAM_SELECTOR_DIR PETCAM_CAMERA_AGENT_URL PETCAM_CAMERA_STREAM_URL \
        PETCAM_RTSP_USER PETCAM_RTSP_PASSWORD PETCAM_PETS PETCAM_PAGE_PASSPHRASE > "$RENDERED_PAGE"
    rsync -a "$RENDERED_PAGE" "${DEST}petcam/repo/collector/macos/$PAGE_PLIST"
fi
# The selector's environment is installed once its tools have set up its data directory.
# A failed install stops here, before the running collector is replaced.
on_collector "$SETUP; cd ~/petcam/repo/collector && uv sync -q --python 3.12; \
    if [ -d '$SELECTOR_DIR/frames' ]; then (cd ~/petcam/repo/selector && uv sync -q --python 3.12); fi; \
    $(reload "$PLIST"); \
    tail -n 3 ~/petcam/logs/collector.log 2>/dev/null || true"

# The page: installed and kept running when petcam.env gives it an address to listen on. A
# page started by hand holds the same port, so it is stopped first.
if [ -n "$PETCAM_PAGE_HOST" ]; then
    on_collector "$SETUP; cd ~/petcam/repo/selector && uv sync -q --python 3.12; \
        pkill -f 'python -m selector .*annotate' 2>/dev/null || true; \
        $(reload "$PAGE_PLIST"); \
        echo 'selector page: http://$PETCAM_PAGE_HOST:${PETCAM_PAGE_PORT:-8798}/ (its first start takes about a minute)'"
fi

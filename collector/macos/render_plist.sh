#!/bin/sh
# ABOUTME: Renders a launchd plist from a template: fills in the collector home, time zone and page
# ABOUTME: address, and hands PETCAM_* values from petcam.env to the job's environment.
# Usage: render_plist.sh [template [NAME ...]]. Without a template, the collector's. With names,
# only those PETCAM_* values are handed on; without, all of them. COLLECTOR_HOME in the
# environment stands in for PETCAM_COLLECTOR_HOME, which is how deploy.sh --local names this Mac.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../../config.sh"
TEMPLATE="${1:-$HERE/com.petcam.collector.plist.template}"
[ $# -gt 0 ] && shift
ONLY=" $* "

xml_escape() {
    printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

while IFS= read -r line; do
    case "$line" in
        *@ENVIRONMENT@*)
            for name in $(env | sed -n 's/^\(PETCAM_[A-Z_]*\)=.*/\1/p' | sort); do
                case "$ONLY" in "  "|*" $name "*) ;; *) continue ;; esac
                eval "value=\$$name"
                printf '        <key>%s</key>\n        <string>%s</string>\n' \
                    "$name" "$(xml_escape "$value")"
            done
            ;;
        *)
            printf '%s\n' "$line" | sed -e "s|@COLLECTOR_HOME@|${COLLECTOR_HOME:-$PETCAM_COLLECTOR_HOME}|g" \
                -e "s|@TIMEZONE@|$PETCAM_TIMEZONE|g" \
                -e "s|@PAGE_HOST@|$PETCAM_PAGE_HOST|g" -e "s|@PAGE_PORT@|${PETCAM_PAGE_PORT:-8798}|g"
            ;;
    esac
done < "$TEMPLATE"

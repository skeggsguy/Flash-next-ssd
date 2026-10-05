#!/bin/bash
# Starts the Flash-Next server by itself at every login, with the settings in flashnext.conf.
#
#   install-autostart.sh [internal|two-ssds] [--no-load]
#
#   internal    the internal SSD only (the default)
#   two-ssds    two SSDs, as "Start Flash-Next (two SSDs).command"
#   --no-load   install for the next login, but don't start it now
#
# It fills in launchd/com.flashnext.server.plist.template and copies it to
# ~/Library/LaunchAgents/com.flashnext.server.plist (yours, outside git). The server's log goes
# to ~/Library/Logs/flashnext/server.log. Undo it with uninstall-autostart.sh.

set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
LABEL=com.flashnext.server
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist
LOGS=$HOME/Library/Logs/flashnext
DRIVES=internal
LOAD=1
for arg in "$@"; do
    case $arg in
        internal|two-ssds) DRIVES=$arg ;;
        --no-load) LOAD=0 ;;
        *) echo "usage: $0 [internal|two-ssds] [--no-load]" >&2; exit 2 ;;
    esac
done
case $REPO$HOME in
    *[\&\<\>\|]*) echo "The folder path has a character the launchd file can't hold (& < > |)." >&2; exit 1 ;;
esac

if pgrep -x llama-server >/dev/null; then
    LOAD=0
    echo "A llama-server is running now, so autostart won't start another one today."
elif ! "$HERE/start.sh" "$DRIVES" --dry-run; then
    echo
    echo "Fix the problems above, then run this again. Nothing was installed."
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$LOGS"
sed -e "s|__REPO__|$REPO|g" -e "s|__HOME__|$HOME|g" -e "s|__DRIVES__|$DRIVES|g" \
    "$HERE/launchd/$LABEL.plist.template" > "$PLIST"
plutil -lint "$PLIST" >/dev/null

echo
echo "Installed $PLIST ($DRIVES)."
if [ "$LOAD" = 1 ]; then
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "Started: it loads in the background. Watch it with: tail -f \"$LOGS/server.log\""
else
    echo "It will start at your next login. Its log: $LOGS/server.log"
fi
echo "It also starts at every login from now on, 10 minutes after the Mac starts."

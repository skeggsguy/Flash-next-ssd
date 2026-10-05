#!/bin/bash
# Stops starting the Flash-Next server at login. If the autostarted server is running, it stops too.
# Your flashnext.conf and the server's log (~/Library/Logs/flashnext) are left alone.

set -u
LABEL=com.flashnext.server
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
if [ -f "$PLIST" ]; then
    rm "$PLIST"
    echo "Removed $PLIST: the server no longer starts at login."
else
    echo "Autostart wasn't installed ($PLIST doesn't exist)."
fi

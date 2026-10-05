#!/bin/bash
# Double-click to start the Flash-Next server, reading the model from two SSDs at once:
# the internal one and the copy on a second drive (SECOND_COPY in flashnext.conf). See README.md.
exec "$(dirname "$0")/tools/flashnext/start.sh" two-ssds "$@"

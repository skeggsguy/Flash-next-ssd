#!/bin/bash
# Double-click to start the Flash-Next server, reading the model from the internal SSD.
# Settings: flashnext.conf (made from flashnext.conf.example on the first start). See README.md.
exec "$(dirname "$0")/tools/flashnext/start.sh" internal "$@"

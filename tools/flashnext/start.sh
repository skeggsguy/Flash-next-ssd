#!/bin/bash
# Starts the Flash-Next server with the settings in flashnext.conf, after checking that
# everything it needs is there. The "Start Flash-Next" files and the autostart run this.
#
#   start.sh internal|two-ssds [--dry-run] [--at-login]
#
#   internal    read the model from the internal SSD only
#   two-ssds    also read it from the copy on a second SSD (SECOND_COPY)
#   --dry-run   check and print the command, but start nothing (or FLASHNEXT_DRY_RUN=1)
#   --at-login  first wait until the Mac has been up 10 minutes (the autostart uses this)
#
# FLASHNEXT_CONF=path reads another settings file instead of flashnext.conf.

set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DRIVES=${1:-}
DRY_RUN=${FLASHNEXT_DRY_RUN:-0}
AT_LOGIN=0
case $DRIVES in
    internal|two-ssds) shift ;;
    *) echo "usage: $0 internal|two-ssds [--dry-run] [--at-login]" >&2; exit 2 ;;
esac
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
        --at-login) AT_LOGIN=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

PROBLEMS=0
# Says what is wrong and what to do, then stops. A dry run notes it and carries on.
stop() {
    printf '\n' >&2
    printf '%s\n' "$@" >&2
    PROBLEMS=$((PROBLEMS + 1))
    [ "$DRY_RUN" = 1 ] || exit 1
}
note() {
    printf '\nNote: %s\n' "$1"
    shift
    for line in "$@"; do printf '      %s\n' "$line"; done
}
home() { printf '%s' "${1/#\~/$HOME}"; }   # a quoted "~/..." from the settings file

# --- The settings: the example's values first, then yours on top ------------------------
EXAMPLE=$ROOT/flashnext.conf.example
CONF=${FLASHNEXT_CONF:-$ROOT/flashnext.conf}   # the file every message below names
SETTINGS=$CONF                                  # the file read: the example until CONF exists
if [ ! -f "$CONF" ]; then
    if [ "$DRY_RUN" = 1 ]; then
        echo "No $CONF yet: using the example's settings."
        echo "To change one, copy flashnext.conf.example to flashnext.conf and edit the copy (the first"
        echo "real start makes it too)."
        SETTINGS=$EXAMPLE
    else
        cp "$EXAMPLE" "$CONF" || exit 1
        echo "First start: made $CONF from the example. Edit it to change a setting."
    fi
fi
. "$EXAMPLE"
. "$SETTINGS"
MODEL=$(home "$MODEL")
DRAFT_HEAD=$(home "$DRAFT_HEAD")
SECOND_COPY=$(home "$SECOND_COPY")
BIN=$ROOT/build/bin/llama-server

# --- At login, give macOS its first 10 minutes ------------------------------------------
if [ "$AT_LOGIN" = 1 ]; then
    BOOTED=$(sysctl -n kern.boottime | sed 's/^{ sec = \([0-9]*\).*/\1/')
    WAIT=$((BOOTED + 600 - $(date +%s)))
    if [ "$WAIT" -gt 0 ]; then
        echo "The Mac started less than 10 minutes ago: waiting $WAIT s while macOS settles."
        [ "$DRY_RUN" = 1 ] || sleep "$WAIT"
    fi
fi

# --- Checks: each says what to do ---------------------------------------------------------
if pgrep -x llama-server >/dev/null; then
    stop "A llama-server is already running, so this one won't start: two at once would" \
         "fight over memory and push the Mac into swap. Use the one that is running, or stop" \
         "it first (Control-C in its window; if it started at login, see README.md," \
         "\"Start at login\")."
elif nc -z -G 2 127.0.0.1 "$PORT" >/dev/null 2>&1; then
    stop "Something is already answering on port $PORT. If it is a Flash-Next server, use it:" \
         "http://127.0.0.1:$PORT. Otherwise stop that program, or set another PORT in $CONF."
fi

if [ ! -x "$BIN" ]; then
    stop "The server isn't built yet ($BIN is missing). In Terminal, run:" \
         "  cd \"$ROOT\"" \
         "  cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
fi

if [ ! -f "$MODEL" ]; then
    stop "Can't find the model at $MODEL." \
         "Download it (README.md, \"Quick start\", step 2), or set MODEL in $CONF" \
         "to where its first file is."
fi

if [ -n "$DRAFT_HEAD" ] && [ ! -f "$DRAFT_HEAD" ]; then
    stop "Can't find the draft head at $DRAFT_HEAD." \
         "Download it (README.md, \"Quick start\", step 2), set DRAFT_HEAD in $CONF to where" \
         "it is, or set DRAFT_HEAD= (empty) to run without it."
fi

if [ "$DRIVES" = two-ssds ]; then
    DRIVE=$(printf '%s' "$SECOND_COPY" | sed -n 's|^\(/Volumes/[^/]*\).*|\1|p')
    if [ -z "$SECOND_COPY" ]; then
        stop "SECOND_COPY is empty in $CONF: set it to the model's first file on the second" \
             "drive, or double-click \"Start Flash-Next.command\" to use the internal SSD only."
    elif [ -n "$DRIVE" ] && [ ! -d "$DRIVE" ]; then
        stop "The second drive ($DRIVE) isn't connected. Plug it in and wait for it to appear" \
             "in Finder, then try again, or double-click \"Start Flash-Next.command\" to use" \
             "the internal SSD only."
    elif [ ! -f "$SECOND_COPY" ]; then
        stop "The second drive has no model copy at $SECOND_COPY." \
             "Copy the four model files there (docs/flashnext/getting-started.md, step 3)," \
             "or set SECOND_COPY in $CONF to where the copy is."
    fi
fi

# Advice only: these never stop the start.
if [ "$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)" = 0 ]; then
    note "the GPU's wired-memory limit is macOS's default (75% of RAM). The study raised it:" \
         "docs/flashnext/getting-started.md, step 4."
fi
PILE=$(vm_stat | awk '/Pages stored in compressor/ { print $NF + 0 }')
if [ "${PILE:-0}" -gt 90000 ]; then
    note "macOS holds a lot of compressed memory ($PILE pages), which makes the start's" \
         "memory rush riskier. If you see swap, restart the Mac (or run sudo purge) and start again" \
         "(docs/flashnext/memory-sizing.md, \"Opening the model safely\")."
fi

# --- The command: the guides' everyday settings ------------------------------------------
ENVS=(LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB=4096 GGML_METAL_RESIDENCY_KEEP_ALIVE_S=10000000)
ARGS=(-m "$MODEL" -ngl 99
      --moe-stream --moe-stream-cache "$CACHE_GIB" --moe-stream-io-threads 8 --moe-stream-direct)
if [ "$DRIVES" = two-ssds ]; then
    ARGS+=(--moe-stream-alt-path "$SECOND_COPY" --moe-stream-alt-split "$SECOND_SPLIT")
fi
if [ -n "$DRAFT_HEAD" ]; then
    ENVS=(LLAMA_SPEC_ADAPTIVE_RATE=1 "${ENVS[@]}")
    ARGS+=(-md "$DRAFT_HEAD" --spec-draft-ngl 99
           --spec-type draft-mtp-adaptive --spec-draft-n-max 5 --spec-draft-p-min 0.3
           --spec-max-prompt 0)
fi
ARGS+=(-c "$CONTEXT" -b 4096 -ub 4096 -cms 8192 -ctxcp 3 -np 1 -fa on
       --cache-reuse 0 --cache-ram 0 --jinja --reasoning-format deepseek)
if [ "${CHAT_WINDOW:-0}" != 0 ]; then
    ARGS+=(--chat-window "$CHAT_WINDOW")
fi
ARGS+=(--host 127.0.0.1 --port "$PORT")
CMD=(env "${ENVS[@]}" "$BIN" "${ARGS[@]}")

if [ "$DRY_RUN" = 1 ]; then
    printf '\nWould run:\n'
    printf '%q ' "${CMD[@]}"
    printf '\n'
    if [ "$PROBLEMS" -gt 0 ]; then
        printf '\n%s problem(s) above would stop a real start.\n' "$PROBLEMS"
        exit 1
    fi
    exit 0
fi

echo
echo "Starting Flash-Next ($DRIVES). It loads in seconds; the first answer is slow while the"
echo "expert cache fills. Open http://127.0.0.1:$PORT, or point a client at"
echo "http://127.0.0.1:$PORT/v1."
[ "$AT_LOGIN" = 1 ] || echo "To stop it, press Control-C or close this window."
echo
exec "${CMD[@]}"

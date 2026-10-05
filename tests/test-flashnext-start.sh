#!/bin/bash
# Tests the one-click start (tools/flashnext/start.sh, the two .command files) and the
# autostart install scripts. Nothing real starts: each case runs in a copy of the scripts
# beside a fake llama-server that only prints what it was given, with fake model paths, and
# pgrep, sysctl, vm_stat and launchctl stood in by small scripts on PATH. The port check
# uses a real listener.
#
#   bash tests/test-flashnext-start.sh

set -u
SRC=$(cd "$(dirname "$0")/.." && pwd)
T=$(cd "$(mktemp -d "${TMPDIR:-/tmp}/flashnext-start.XXXXXX")" && pwd)   # pwd: no "//"
LISTENER=
trap '[ -n "$LISTENER" ] && kill "$LISTENER" 2>/dev/null; rm -rf "$T"' EXIT
FAILS=0
PASSES=0

# --- A fake repo: the real scripts, a fake server, stand-ins for the system tools ---------
R="$T/repo dir"   # a space in the path, as a Finder folder may have
mkdir -p "$R/tools/flashnext/launchd" "$R/build/bin" "$T/stub" "$T/home" "$T/models"
cp "$SRC/tools/flashnext/"*.sh "$R/tools/flashnext/"
cp "$SRC/tools/flashnext/launchd/"* "$R/tools/flashnext/launchd/"
cp "$SRC/flashnext.conf.example" "$SRC/"*.command "$R/"
cat > "$R/build/bin/llama-server" <<'EOF'
#!/bin/bash
echo "FAKE-SERVER ADAPTIVE=${LLAMA_SPEC_ADAPTIVE_RATE:-unset} CHUNK=${LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB:-unset} $*"
EOF
cat > "$T/stub/pgrep" <<'EOF'
#!/bin/bash
[ "${STUB_RUNNING:-0}" = 1 ] && { echo 4242; exit 0; }
exit 1
EOF
cat > "$T/stub/sysctl" <<'EOF'
#!/bin/bash
case "$*" in
    *kern.boottime*) echo "{ sec = ${STUB_BOOT:-1000}, usec = 0 } Thu Jan  1 00:16:40 1970" ;;
    *iogpu.wired_limit_mb*) echo "${STUB_WIRED:-59392}" ;;
    *) exec /usr/sbin/sysctl "$@" ;;
esac
EOF
cat > "$T/stub/vm_stat" <<'EOF'
#!/bin/bash
echo "Pages stored in compressor:                   ${STUB_PILE:-1000}."
EOF
cat > "$T/stub/launchctl" <<'EOF'
#!/bin/bash
echo "launchctl $*" >> "$STUB_LOG"
EOF
chmod +x "$R/build/bin/llama-server" "$T/stub/"*
export PATH="$T/stub:$PATH" HOME="$T/home" STUB_LOG="$T/launchctl.log"
unset FLASHNEXT_DRY_RUN FLASHNEXT_CONF
free_port() {
    local p
    for p in $(jot 50 18100); do
        nc -z 127.0.0.1 "$p" >/dev/null 2>&1 || { echo "$p"; return; }
    done
}
FREE_PORT=$(free_port)
touch "$T/models/m1.gguf" "$T/models/head.gguf" "$T/models/copy1.gguf" "$T/home/tilde.gguf"

conf() {   # conf [extra lines...]: a settings file with every path present, plus overrides
    {
        echo "MODEL=$T/models/m1.gguf"
        echo "DRAFT_HEAD=$T/models/head.gguf"
        echo "SECOND_COPY=$T/models/copy1.gguf"
        echo "PORT=$FREE_PORT"
        printf '%s\n' "$@"
    } > "$T/test.conf"
    export FLASHNEXT_CONF="$T/test.conf"
}
run() {    # run ARGS...: the helper, output and exit status kept in OUT and CODE
    OUT=$("$R/tools/flashnext/start.sh" "$@" 2>&1)
    CODE=$?
}
pass() { PASSES=$((PASSES + 1)); }
fail() { FAILS=$((FAILS + 1)); printf 'FAIL %s\n---- output:\n%s\n----\n' "$1" "$OUT"; }
has() { case $OUT in *"$2"*) pass ;; *) fail "$1: missing '$2'" ;; esac; }
lacks() { case $OUT in *"$2"*) fail "$1: should not have '$2'" ;; *) pass ;; esac; }
code() { [ "$CODE" = "$2" ] && pass || fail "$1: exit $CODE, wanted $2"; }
OUT=

# The start files must stay executable, or a double-click opens them in a text editor.
[ -x "$SRC/Start Flash-Next.command" ] && [ -x "$SRC/Start Flash-Next (two SSDs).command" ] && pass || fail "the .command files are executable"
[ -x "$SRC/tools/flashnext/start.sh" ] && [ -x "$SRC/tools/flashnext/install-autostart.sh" ] && [ -x "$SRC/tools/flashnext/uninstall-autostart.sh" ] && pass || fail "the helper scripts are executable"

# The fake repo's example uses the free port too, so no case ever knocks on a real server.
grep -qx 'PORT=8080' "$R/flashnext.conf.example" && pass || fail "the example's port is 8080"
sed -i '' "s/^PORT=8080$/PORT=$FREE_PORT/" "$R/flashnext.conf.example"
Q() { printf '%q' "$1"; }  # how the dry run quotes a path
BIN=$(Q "$R/build/bin/llama-server")

# --- The command, internal SSD ------------------------------------------------------------
conf
run internal --dry-run
code "internal dry run" 0
has "internal command" "Would run:
env LLAMA_SPEC_ADAPTIVE_RATE=1 LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB=4096 GGML_METAL_RESIDENCY_KEEP_ALIVE_S=10000000 $BIN -m $T/models/m1.gguf -ngl 99 --moe-stream --moe-stream-cache 28 --moe-stream-io-threads 8 --moe-stream-direct -md $T/models/head.gguf --spec-draft-ngl 99 --spec-type draft-mtp-adaptive --spec-draft-n-max 5 --spec-draft-p-min 0.3 --spec-max-prompt 0 -c 200000 -b 4096 -ub 4096 -cms 8192 -ctxcp 3 -np 1 -fa on --cache-reuse 0 --cache-ram 0 --jinja --reasoning-format deepseek --chat-window 300 --host 127.0.0.1 --port $FREE_PORT "
lacks "internal leaves the second drive out" "--moe-stream-alt"

# --- The command, two SSDs ------------------------------------------------------------------
run two-ssds --dry-run
code "two-ssds dry run" 0
has "two-ssds command" "--moe-stream-direct --moe-stream-alt-path $T/models/copy1.gguf --moe-stream-alt-split 53 -md "

# --- The .command files pass the right switch -------------------------------------------
OUT=$("$R/Start Flash-Next.command" --dry-run 2>&1); CODE=$?
code "internal .command" 0
lacks "internal .command" "--moe-stream-alt"
OUT=$(FLASHNEXT_DRY_RUN=1 "$R/Start Flash-Next (two SSDs).command" 2>&1); CODE=$?
code "two-ssds .command" 0
has "two-ssds .command" "--moe-stream-alt-path $T/models/copy1.gguf"

# --- Settings ------------------------------------------------------------------------------
conf "CACHE_GIB=26" "CONTEXT=65536" "CHAT_WINDOW=600" "SECOND_SPLIT=60"
run two-ssds --dry-run
has "cache setting" "--moe-stream-cache 26 "
has "context setting" " -c 65536 "
has "chat window setting" " --chat-window 600 --host"
has "split setting" "--moe-stream-alt-split 60 "

conf "CHAT_WINDOW=0"
run internal --dry-run
lacks "chat window 0 is off" "--chat-window"

conf "DRAFT_HEAD="
run internal --dry-run
code "no draft head" 0
lacks "no draft head: no -md" " -md "
lacks "no draft head: no --spec" "--spec-"
lacks "no draft head: no measured depth" "LLAMA_SPEC_ADAPTIVE_RATE"

conf 'MODEL="~/tilde.gguf"'
run internal --dry-run
has "a quoted ~ is the home folder" " -m $T/home/tilde.gguf "

printf 'PORT=%s\n' "$FREE_PORT" > "$T/test.conf"   # every other line from the example
run internal --dry-run
has "a missing line takes the example's value" " -m $HOME/models/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf "
has "the example's draft head" " -md $HOME/models/flashnext/mtp-shared-Q4_K_M.gguf "

# --- Each stop says what to do, and a real start does not run the server -----------------
conf "MODEL=$T/models/missing.gguf"
run internal --dry-run
code "missing model, dry run" 1
has "missing model" "Can't find the model at $T/models/missing.gguf."
has "dry run still prints the command" "Would run:"
has "dry run counts problems" "1 problem(s) above would stop a real start."
run internal
code "missing model, real start" 1
lacks "missing model: no server" "FAKE-SERVER"

conf "DRAFT_HEAD=$T/models/nohead.gguf"
run internal
code "missing draft head" 1
has "missing draft head" "Can't find the draft head at $T/models/nohead.gguf."
has "missing draft head: how to go without" "set DRAFT_HEAD= (empty)"

DRIVE=/Volumes/NoSuchDrive-flashnext-test-$$
conf "SECOND_COPY=$DRIVE/flashnext/m1.gguf"
run two-ssds
code "drive not connected" 1
has "drive not connected" "The second drive ($DRIVE) isn't connected."
lacks "drive not connected: no server" "FAKE-SERVER"
run internal --dry-run
code "internal ignores the second drive" 0

conf "SECOND_COPY=$T/models/nocopy.gguf"
run two-ssds
code "no copy on the drive" 1
has "no copy on the drive" "The second drive has no model copy at $T/models/nocopy.gguf."

conf "SECOND_COPY="
run two-ssds
code "empty second copy" 1
has "empty second copy" "SECOND_COPY is empty"

conf
STUB_RUNNING=1 run internal
code "server already running" 1
has "server already running" "A llama-server is already running, so this one won't start"
lacks "server already running: no server" "FAKE-SERVER"

conf
nc -lk 127.0.0.1 "$FREE_PORT" >/dev/null 2>&1 &
LISTENER=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do nc -z 127.0.0.1 "$FREE_PORT" >/dev/null 2>&1 && break; sleep 0.2; done
run internal
code "port busy" 1
has "port busy" "Something is already answering on port $FREE_PORT."
lacks "port busy: no server" "FAKE-SERVER"
kill "$LISTENER" 2>/dev/null; wait "$LISTENER" 2>/dev/null; LISTENER=

mv "$R/build/bin/llama-server" "$T/llama-server.saved"
run internal
code "not built" 1
has "not built" "The server isn't built yet"
has "not built: how to build" "cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
mv "$T/llama-server.saved" "$R/build/bin/llama-server"

conf "MODEL=$T/models/missing.gguf" "DRAFT_HEAD=$T/models/nohead.gguf"
run internal --dry-run
has "dry run reports every problem" "2 problem(s) above would stop a real start."

# --- Advice never stops a start ---------------------------------------------------------
conf
STUB_PILE=200000 STUB_WIRED=0 run internal --dry-run
code "advice only" 0
has "big pile" "macOS holds a lot of compressed memory (200000 pages)"
has "default wired limit" "the GPU's wired-memory limit is macOS's default"
run internal --dry-run
lacks "no advice when all is well" "Note:"

# --- At login: wait for the Mac's first 10 minutes ----------------------------------------
STUB_BOOT=$(( $(date +%s) - 60 )) run internal --dry-run --at-login
code "at login, early" 0
has "at login, early" "waiting 5"   # 540 s, give or take a second
STUB_BOOT=$(( $(date +%s) - 3600 )) run internal --dry-run --at-login
lacks "at login, later" "waiting"

# --- A real start runs the server with the same command ---------------------------------
conf
run two-ssds
code "real start" 0
has "real start runs the server" "FAKE-SERVER ADAPTIVE=1 CHUNK=4096 -m $T/models/m1.gguf -ngl 99 "
has "real start, same flags" "--moe-stream-alt-split 53 -md $T/models/head.gguf"
has "real start, chat window" "--chat-window 300 --host 127.0.0.1 --port $FREE_PORT"

unset FLASHNEXT_CONF
run internal --dry-run
has "dry run without a settings file uses the example" "using the example's settings"
has "no settings file: how to make one" "copy flashnext.conf.example to flashnext.conf"
has "no settings file: a stop names flashnext.conf" "set MODEL in $R/flashnext.conf
to where its first file is."
lacks "no settings file: a stop never points at the example" "in $R/flashnext.conf.example"
[ ! -f "$R/flashnext.conf" ] && pass || fail "a dry run must not write flashnext.conf"
run internal
has "first start makes flashnext.conf" "First start: made $R/flashnext.conf from the example."
cmp -s "$R/flashnext.conf" "$R/flashnext.conf.example" && pass || fail "flashnext.conf is a copy of the example"

# --- Wrong use ---------------------------------------------------------------------------
run
code "no switch" 2
run both
code "unknown switch" 2
run internal --go-fast
code "unknown option" 2

# --- Autostart install -------------------------------------------------------------------
conf
cp "$T/test.conf" "$R/flashnext.conf"
unset FLASHNEXT_CONF
PLIST="$HOME/Library/LaunchAgents/com.flashnext.server.plist"
: > "$STUB_LOG"
OUT=$("$R/tools/flashnext/install-autostart.sh" two-ssds --no-load 2>&1); CODE=$?
code "install --no-load" 0
[ -f "$PLIST" ] && pass || fail "install writes the plist"
plutil -lint "$PLIST" >/dev/null && pass || fail "the filled plist passes plutil -lint"
OUT=$(cat "$PLIST")
lacks "every placeholder filled" "__"
has "plist runs the helper" "<string>$R/tools/flashnext/start.sh</string>"
has "plist passes the switch" "<string>two-ssds</string>"
has "plist waits at login" "<string>--at-login</string>"
has "plist logs to the user's Library" "<string>$HOME/Library/Logs/flashnext/server.log</string>"
has "plist restarts only after a failure" "<key>SuccessfulExit</key>
		<false/>"
has "plist waits 5 minutes between restarts" "<key>ThrottleInterval</key>
	<integer>300</integer>"
[ -d "$HOME/Library/Logs/flashnext" ] && pass || fail "install makes the log folder"
[ ! -s "$STUB_LOG" ] && pass || fail "--no-load must not call launchctl"

OUT=$("$R/tools/flashnext/install-autostart.sh" 2>&1); CODE=$?
code "install and load" 0
OUT=$(cat "$STUB_LOG")
has "install loads the agent" "launchctl bootstrap gui/$(id -u) $PLIST"
OUT=$(cat "$PLIST")
has "install defaults to the internal SSD" "<string>internal</string>"

: > "$STUB_LOG"
OUT=$(STUB_RUNNING=1 "$R/tools/flashnext/install-autostart.sh" 2>&1); CODE=$?
code "install while a server runs" 0
has "install while a server runs" "won't start another one today"
[ ! -s "$STUB_LOG" ] && pass || fail "install must not load while a server runs"

rm -f "$PLIST"
echo "MODEL=$T/models/missing.gguf" >> "$R/flashnext.conf"
OUT=$("$R/tools/flashnext/install-autostart.sh" 2>&1); CODE=$?
code "install with a problem" 1
has "install with a problem" "Nothing was installed."
[ ! -f "$PLIST" ] && pass || fail "install with a problem writes no plist"

cp "$T/test.conf" "$R/flashnext.conf"
"$R/tools/flashnext/install-autostart.sh" --no-load >/dev/null 2>&1
: > "$STUB_LOG"
OUT=$("$R/tools/flashnext/uninstall-autostart.sh" 2>&1)
has "uninstall" "the server no longer starts at login"
[ ! -f "$PLIST" ] && pass || fail "uninstall removes the plist"
OUT=$(cat "$STUB_LOG")
has "uninstall stops the agent" "launchctl bootout gui/$(id -u)/com.flashnext.server"

echo "$PASSES passed, $FAILS failed"
[ "$FAILS" = 0 ]

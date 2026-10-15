# Flash-Next SSD

**Run Qwen3.8-Flash-Next, a 111 GB model, on a 64 GB Mac.**

This is a [llama.cpp](https://github.com/ggml-org/llama.cpp) fork for Apple Silicon. The model is a
mixture of experts: 48 layers of 512 experts, and each token uses only 10 experts on each layer. So
the experts stay on the SSD. The ones a token needs are read in as it is computed, and the ones used
most stay in an expert cache in RAM. Everything else stays resident on the GPU.

It was tuned for a Mac mini M5 Pro with 64 GB and a second SSD over Thunderbolt 5. The tuning was
measured on real agent conversations, and any timing that swap touched was thrown out.

<p align="center"><img src="media/flashnext/how-it-works.svg" width="760" alt="How a 111 GB model runs on a 64 GB Mac: prompts stream through the reading room, answers come from the expert cache, and missing experts are fetched from two SSDs"></p>

## Headline numbers

Measured on a Mac mini M5 Pro (64 GB) with Unsloth's UD-Q4_K_XL quant, a 28 GiB expert cache, two
SSDs, the MTP draft head and both trained guessers on ([what each extra is worth](#what-each-extra-is-worth)).
Sources and method are in [Results](docs/flashnext/results.md).

The October numbers come from two sets of the author's own work, each replayed turn by turn: 24
coding-agent conversations at high thinking effort (`code`), and 24 chat conversations at medium
effort (`chat`).

| | |
|---|---|
| **Writing (decode), coding agent** | **23.6 tokens/s**: **+9.6% since the last update** (21.5 on the same conversations) |
| **Writing (decode), chat** | **25.0 tokens/s**: **+9.7% since the last update** (from 22.7) |
| **A new chat while a coding agent works** | **3.5-4.1 s** to the first token, from 14.2 s: the chat's opening comes back from RAM instead of being read in again, also when the agent had the server first |
| **Reading a prompt (prefill)** | **523 / 430 / 391 tokens/s** at 4K / 32K / 100K tokens, from cold |
| **Quality** | **18/20** on a fixed 20-task exam, the same score as the plain setup. None of the October changes alters a single output token |

## What it adds

### SSD expert streaming with an expert cache

`--moe-stream` keeps the routed experts on disk and holds a fixed-size expert cache in RAM
(`--moe-stream-cache 28` = 28 GiB). Reads skip macOS's file cache (`--moe-stream-direct`), so the
cache and the rest of the system keep their memory, and many reads are in flight at once. While a
token is being written, the next layer's likely experts are fetched ahead of time; about 1 in 4 of
those early fetches is used before the cache lets it go (the trained lookahead below does better).
When the cache is full, it lets go the expert that has gone longest without being used (below). At
28 GiB about 4 in 5 expert lookups find their expert already in RAM.

### Which expert the cache lets go (new in October)

When the expert cache is full and a new expert has to come in, another has to go. The old rule let
go the expert with the lowest recent use count. The new default lets go the one that has gone
longest without being used (least recently used, LRU). The rule itself was written by Marian
Mihailescu, in his GPU slot-resolution work.

It changes which experts are in RAM, never which are used, so the output is unchanged (byte-identical
word checks). It is on by default; `LLAMA_MOE_STREAM_LRU=0` restores the old rule.

### The trained lookahead: guessing one and two layers ahead (new in October)

Guessing one layer ahead leaves the SSD only one layer's compute time to fetch a missed expert, and
the engine's own guess is rough: only about 1 in 4 of its early fetches is used, and half are
evicted from the cache before their layer comes. Two small trained guessers on
[crystal-ball](https://huggingface.co/skeggsguy/crystal-ball) fix both. Each was trained on the
model's own routing while it wrote real replies.

Together: **about +12% writing on the coding set and +16% on chat**, against the engine's own guess
without the keep rule.

<p align="center"><img src="media/flashnext/lookahead.svg" width="760" alt="Before: guessing one layer ahead, the SSD has one layer's time, the GPU waits and only 1 in 4 guesses is used. After: the two-layer guesser fetches experts two layers ahead, the one-layer guesser tops up a layer later, and each guess is kept until its layer runs; 84% and 65-71% of guesses used. Writing about +12% coding and +16% chat, estimated from two runs"></p>

### The reading room: a carousel for long prompts

A long prompt needs nearly every expert on every layer. The reading room is a small ring buffer
(about 1.4 GiB) carved out of the expert cache. The SSDs fill it layer after layer, ahead of the
GPU, so prompt reading costs whichever of the two is slower, not both added together. The output
is bit-identical to having every expert in RAM. While writing, the buffer is lent back as extra
cache, which won back its cost (+5.5% writing). It is on by default.

<p align="center"><img src="media/flashnext/carousel.svg" width="760" alt="Before: the SSD and GPU take turns. After: the reading room keeps both busy, reading in +32%, time to first word −28%"></p>

### Two drives

Put a byte-identical copy of the model on a second SSD and the expert reads are split across both
(`--moe-stream-alt-path`, `--moe-stream-alt-split 53`). The internal SSD (6.5 GB/s) and a 990 PRO
over Thunderbolt 5 (5.7 GB/s) give 12.3 GB/s together. That made writing 15% faster and cut the time
to first token by 31%.

<p align="center"><img src="media/flashnext/two-ssds.svg" width="760" alt="Gain from a second SSD: reading in +40% before the reading room and +15% after; writing +15%; time to first word −31% to −33%"></p>

### The MTP draft head, at a depth it measures for itself

A small draft head guesses up to 5 tokens ahead, and the model checks them all in one pass
(`-md` with `--spec-type draft-mtp-adaptive`, which `--flashnext` sets). It picks its own guessing
depth from how often recent guesses were kept; that measured depth is the engine's default
(`LLAMA_SPEC_ADAPTIVE_RATE=0` goes back to the older rule, a depth that climbs and falls with
how many guesses were kept). On the agent conversations (tool calls,
code) it made writing 23% faster. On prose it breaks even, because every guess, right or wrong, needs
its own experts fetched.

### Union attention

The model's sparse attention picks the context blocks each query attends to. Union attention
shares one deduplicated pick list per batch while reading a prompt. At 160K context its working
memory drops from 5.6 GiB to 1.9 GiB, and prompt reading gets 5% faster. It sums in a different
order, so words can differ at near-ties; perplexity is unchanged. It is the default
(`LLAMA_QSA_UNION=1`).

### The chat window

One server, two users: you chatting, and a coding agent working in the background. A chat request
stops a running background request, which gets a 503 that its client retries. Background requests
then wait until 90 seconds after the last chat reply. Only the background client needs to send a
header (`X-Lane: code`). `--flashnext` turns it on at 90 s (`--chat-window 90`); with no such client
it changes nothing.

<p align="center"><img src="media/flashnext/chat-window.svg" width="760" alt="The chat window: a chat request cuts a running coding request off, its retry is held while the window is open, and coding resumes when the window lapses"></p>

### And smaller wins, all on by default

- The n-gram table's hot rows sit on a 128 MiB shelf instead of macOS's file cache (`--ple-shelf`, +3% writing).
- One prompt-cache checkpoint stays pinned at the start of the conversation, so a new chat after a
  long one resumes in 4 s instead of 14 s.
- Cheaper per-layer stops: less locking and lookup at each layer, the next layer's GPU work encoded
  ahead, and a tiny keep-awake ping so the GPU does not power down while it waits (+7% writing).
  
## Quick start

**What you need:** an Apple Silicon Mac with 64 GB, about 115 GB free on the internal SSD, Xcode's
command line tools, and CMake. A second fast SSD is optional but worth it.

**1. Build it**

```sh
git clone https://github.com/skeggsguy/Flash-next-ssd
cd Flash-next-ssd
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**2. Download the files.** That's the model (111.3 GB, 4 files), the draft head (1.9 GB), the two
trained guessers (121 MB and 123 MB) and the image encoder (908 MB). The model takes a while.

```sh
D=~/models/flashnext && mkdir -p $D
for i in 1 2 3 4; do
  curl -fL --retry 5 -C - -o $D/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000$i-of-00004.gguf \
    https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000$i-of-00004.gguf
done
curl -fL --retry 5 -C - -o $D/mtp-shared-Q4_K_M.gguf \
  https://huggingface.co/nitinpanj/qwen38-flash-next-v3/resolve/main/MTP/mtp-shared-Q4_K_M.gguf
curl -fL --retry 5 -C - -o $D/c1-guesser.gguf \
  https://huggingface.co/skeggsguy/crystal-ball/resolve/main/c1-guesser.gguf
shasum -a 256 $D/c1-guesser.gguf   # c22af709d65b99bb2bc64aaa5b8b2d47c04c86367b0f9ba0fe452ab47ec053c7
curl -fL --retry 5 -C - -o $D/c1-guesser-gap1.gguf \
  https://huggingface.co/skeggsguy/crystal-ball/resolve/main/c1-guesser-gap1.gguf
shasum -a 256 $D/c1-guesser-gap1.gguf   # 2df6a3d9c63f787286a387965aa1c89da3ee86c6b75829e3c9a1e2ca24690d94
curl -fL --retry 5 -C - -o $D/mmproj-BF16.gguf \
  https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-BF16.gguf
shasum -a 256 $D/mmproj-BF16.gguf   # 2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355
```

**3. Let the GPU use enough memory.** This resets on every reboot.
[Getting started](docs/flashnext/getting-started.md#4-the-gpus-wired-memory-limit) shows how to set
it at every boot.

```sh
sudo sysctl iogpu.wired_limit_mb=59392
```

**4. Run it**

```sh
LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER=$HOME/models/flashnext/c1-guesser.gguf \
LLAMA_MOE_STREAM_GUESSER1=$HOME/models/flashnext/c1-guesser-gap1.gguf \
./build/bin/llama-server --flashnext \
  -m ~/models/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md ~/models/flashnext/mtp-shared-Q4_K_M.gguf \
  --moe-stream-cache 28
```

That's it: your files, the cache size, and `--flashnext` for everything else. The guessers are
named by environment variables; the model and draft head by flags.

It loads in seconds. The first request is slower while the cache fills up. Then open
<http://127.0.0.1:8080>, or point any OpenAI-compatible client at `http://127.0.0.1:8080/v1` (the
Anthropic-style `/v1/messages` works too). `--host` and `--port` change the address.

**Optional extras**

- **A second SSD:** copy the four model files to it and add
  `--moe-stream-alt-path /Volumes/<ssd>/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --moe-stream-alt-split 53`.
- **Pictures:** add `--mmproj ~/models/flashnext/mmproj-BF16.gguf`.
- **A coding agent sharing the server:** have it send the header `X-Lane: code`. `--flashnext`
  already turns on the 90-second [chat window](#the-chat-window), so your chats go first.

With all three, you're running the study's everyday server. The only differences are where its
files sit and the statistics it records for the study.

**Watch your memory.** At a 28 GiB cache the Mac doesn't have much room to spare, so close
memory-hungry apps. If you see swap (`sysctl vm.swapusage`), use a smaller cache or context.
[Memory sizing](docs/flashnext/memory-sizing.md) helps you choose.

## Start it with one click

Once it's built and downloaded (steps 1-3), double-click one of these in Finder, in the repo's
folder:

- **`Start Flash-Next.command`**: the model on the internal SSD only.
- **`Start Flash-Next (two SSDs).command`**: the internal SSD plus a copy on a second SSD
  ([Getting started](docs/flashnext/getting-started.md#3-the-second-drive-optional) explains the copy).

A Terminal window opens and runs step 4's command, plus the image encoder. Keep the window open
while you use it. To stop the server, press Control-C or close the window.

Before it starts, it checks a few things: the server is built, the model files are where your
settings say, the second drive is plugged in (two SSDs only), and no other server is running. If
something's missing, it tells you what to do and stops. It never starts a second server, because
two at once would push the Mac into swap. To see the checks and the exact command without starting
anything, run it from Terminal with `--dry-run`:

```sh
./"Start Flash-Next (two SSDs).command" --dry-run
```

**Your settings** live in `flashnext.conf`, next to the start files. The first start makes it from
`flashnext.conf.example`; edit it in any text editor. Git ignores it, so pulling an update never
touches it. Delete a setting and it takes the example's value.

| Setting | Default | What it is |
|---|---|---|
| `MODEL` | `~/models/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | The model's first file; the other three sit beside it. |
| `DRAFT_HEAD` | `~/models/flashnext/mtp-shared-Q4_K_M.gguf` | The MTP draft head. Empty (`DRAFT_HEAD=`) runs without it. |
| `GUESSER` | `~/models/flashnext/c1-guesser.gguf` | The trained two-layer guesser ([the trained lookahead](#the-trained-lookahead-guessing-one-and-two-layers-ahead-new-in-october)). If the file is missing, the start says so and runs without it; empty (`GUESSER=`) turns it off. |
| `GUESSER1` | `~/models/flashnext/c1-guesser-gap1.gguf` | The trained one-layer guesser. If the file is missing, the start says so and uses the engine's own guess; empty (`GUESSER1=`) does that on purpose. |
| `VISION` | `~/models/flashnext/mmproj-BF16.gguf` | The image encoder, so the model can read pictures ([Images](docs/flashnext/settings.md#images)). If the file is missing, the start says so and runs text-only; empty (`VISION=`) turns it off. About 0.9 GB beside the cache. |
| `SECOND_COPY` | `/Volumes/ssd/flashnext/...-00001-of-00004.gguf` | Two SSDs only: the first file of the copy on the second drive. |
| `SECOND_SPLIT` | `53` | Two SSDs only: the percent of experts read from the internal copy ([Settings](docs/flashnext/settings.md#two-drives)). |
| `CACHE_GIB` | `28` | The expert cache in GiB, the biggest speed setting ([Memory sizing](docs/flashnext/memory-sizing.md)). |
| `CONTEXT` | `200000` | The context in tokens (65536 on a 48 GB Mac). |
| `CHAT_WINDOW` | `90` | Seconds chat keeps the server after its last reply, when a coding agent shares it ([The chat window](docs/flashnext/chat-window.md)); `0` = off. Without such an agent it changes nothing. |
| `PORT` | `8080` | The server's address is `http://127.0.0.1:PORT`. |

Everything else comes from the engine's defaults and `--flashnext`. To change one of those, run
`llama-server` yourself with the options in [Settings](docs/flashnext/settings.md). An option you
give always beats `--flashnext`.
## Start at login (suggested)

Want the server to start by itself whenever you log in? Install it as a launchd agent:

```sh
tools/flashnext/install-autostart.sh            # internal SSD only
tools/flashnext/install-autostart.sh two-ssds   # or with the second drive
```

It checks your settings first, then copies a filled-in
`tools/flashnext/launchd/com.flashnext.server.plist.template` to
`~/Library/LaunchAgents/com.flashnext.server.plist` and starts the server in the background. Add
`--no-load` to wait for the next login instead. From then on, at every login it:

- runs the same checks and the same command as the start files, with the same `flashnext.conf`;
- waits until the Mac has been up 10 minutes, because opening a 28 GiB cache while macOS is still
  busy after a restart can push it into swap ([Memory sizing](docs/flashnext/memory-sizing.md#opening-the-model-safely));
- if the server fails or a check stops it (say the second drive isn't connected yet), tries again
  5 minutes later, but not after a clean stop;
- writes the server's log to `~/Library/Logs/flashnext/server.log`.

To stop it until the next login, and to start it again:

```sh
launchctl bootout gui/$(id -u)/com.flashnext.server
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.flashnext.server.plist
```

`tools/flashnext/uninstall-autostart.sh` stops it and removes the agent.

**It needs you logged in.** The GPU (Metal) only works in a logged-in session, so the agent starts
at login, not at power-on. For a Mac that should come back by itself after a power cut, turn on
automatic login (System Settings > Users & Groups; macOS offers it only with FileVault off, since
FileVault waits for your password at the unlock screen) and `sudo pmset -a autorestart 1` (start up
after a power failure). Install the wired-limit LaunchDaemon from
[Getting started](docs/flashnext/getting-started.md#4-the-gpus-wired-memory-limit) too, so the
limit is set at every boot.
## Out of the box, and what you add

Once a change proves itself in the study, it becomes the engine's default, unless it needs a
download. So there are three layers: what the engine does with no settings, what `--flashnext`
adds, and what only you can add (the downloads, and choices that depend on your Mac). The study's
own everyday server is exactly these three layers. Nothing is hidden in extra switches.

### With no settings

The fork's own switches already run the study's proven values when you leave them unset. Each is an
environment variable. To go back to the old way, set it to the value in the last column.

| Switch | Unset (the default) | The old way |
|---|---|---|
| `LLAMA_MOE_STREAM_LRU` | A full cache lets go the expert unused longest ([above](#which-expert-the-cache-lets-go-new-in-october)). | `0`: the lowest recent use count |
| `LLAMA_MOE_STREAM_DESK_FIRST` | While a layer's missing experts are read, the GPU already works on the ones in the cache (and the shared expert); the rest run as soon as they land. Same output. | `0`: each layer waits for every missing expert before it starts |
| `LLAMA_RS_SKIP_SLOTS` | The draft head's check saves only the undo copies a rollback can reach. | `0`: every copy, every round |
| `LLAMA_SPEC_ADAPTIVE_RATE` | With `--spec-type draft-mtp-adaptive`, the draft head measures its own depth. | `0`: the older rule, a depth that climbs and falls with the guesses kept |
| `LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB` | The expert cache is claimed in 4 GiB steps with short pauses, so macOS isn't rushed at load. | `0`: all at once |
| `GGML_METAL_RESIDENCY_KEEP_ALIVE_S` | The model stays wired in RAM while the server is idle (10,000,000 s). | `0`: upstream's 180 s, after which the next request wires ~41 GiB back in one go |
| `GGML_METAL_STAFF_MV` | The check's matrix work runs on the route the study measured. | `1`: the M5's matrix units (below the study's bar) |
| The two-layer guesser's tuning: `LLAMA_MOE_STREAM_LOOKAHEAD_ALL`, `_DEPTH2`, `_KEEP`, `_CONF` | The everyday values (on, 4, on, 0.2) once `LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER` names the guesser's file; off without it. | Set any of them by hand |

These join the changes that were already on by default: the reading room, union attention, the
n-gram shelf, the pinned checkpoint, the cheaper per-layer stops and, with a second drive, the drive
choice by load. At start the server prints one line beginning `flashnext:` that lists the defaults
in force, and marks any you set yourself "(yours)".

The engine leaves llama.cpp's own options alone. Their upstream defaults (no streaming, small
batches, four conversations at once, an 8 GiB prompt store in RAM) don't suit this model, and
that's what `--flashnext` is for.

### With `--flashnext`

`--flashnext` sets llama.cpp's options to the everyday values. It only sets an option you haven't
set yourself (on the command line, as its `LLAMA_ARG_*` variable, or in a config file). Without it,
upstream's defaults are untouched.

| Option | What it's for |
|---|---|
| `-ngl 99` | Everything that isn't streamed goes on the GPU. |
| `--moe-stream --moe-stream-direct --moe-stream-io-threads 8` | Stream the experts from the SSD, past macOS's file cache, 8 reads at once. |
| `-c 200000` | A 200K-token context (on 48 GB, set `-c 65536` yourself). |
| `-b 4096 -ub 4096` | Big prompt batches, which the reading room needs. |
| `-cms 8192 -ctxcp 3` | Prompt-cache checkpoints every 8,192 tokens, at most 3 kept. |
| `-np 1` | One conversation at a time: each extra one costs memory the expert cache needs. |
| `-fa on` | Flash attention, which this model's sparse attention needs to read a prompt. |
| `--cache-reuse 0` | Off, as the study ran it. |
| `--cache-ram 400 --cache-ram-opening` | One RAM copy of chat's opening (~232 MiB), so a new chat doesn't read it in again ([the chat window](#the-chat-window)). Nothing else is kept in RAM: on 64 GB that memory belongs to the expert cache. |
| `--chat-window 90` | Chat goes first when a coding agent shares the server. Does nothing until a client sends `X-Lane: code`. |
| `--jinja --reasoning-format deepseek` | The model's chat template, with its thinking returned apart from the answer. |
| With a draft head (`-md`) only: `--spec-type draft-mtp-adaptive --spec-draft-n-max 5 --spec-draft-p-min 0.3 --spec-draft-ngl 99 --spec-max-prompt 0` | The draft head, up to 5 tokens ahead, on the GPU, for prompts of any length. |

The `flashnext:` line at start also says what the preset set and which of your own settings it kept.

### What you add

These are yours: the downloads, and the choices that depend on your Mac.

| What | How | Notes |
|---|---|---|
| The model (download, 111.3 GB) | `-m` and its first file | [Quick start](#quick-start), step 2. |
| The expert cache's size | `--moe-stream-cache 28` | Change the GiB to suit how much memory you want to keep spare; it's yours to adjust each time you start the server. 28 GiB on 64 GB with the draft head; 24 if you keep other apps open (about 6-8% slower, ~4 GiB back); start at 16 on 48 GB ([Memory sizing](docs/flashnext/memory-sizing.md)). |
| The draft head (download, 1.9 GB) | `-md` and its file | `--flashnext` then sets the draft options. |
| The two trained guessers (downloads, 121 and 123 MB) | `LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER=` and `LLAMA_MOE_STREAM_GUESSER1=`, each with its file | Environment variables, not flags. Naming the two-layer guesser also turns on its tuning and the keep rule. |
| The image encoder (download, 908 MB) | `--mmproj` and its file | Lets the model read pictures; about 0.9 GB beside the cache. |
| A second SSD | `--moe-stream-alt-path` with the copy's first file, `--moe-stream-alt-split 53` | +15% writing ([Two drives](#two-drives)). |
| The GPU's wired-memory limit | `sudo sysctl iogpu.wired_limit_mb=59392` | [Quick start](#quick-start), step 3. |

## What each extra is worth

> **The headline numbers are with all the extras on.** That means `--flashnext`, a 28 GiB expert cache,
> the MTP draft head, both trained guessers and a second SSD. Leave an extra off and you'll be
> slower. The table shows roughly how much each one is worth.

| Extra | What it's worth | Without it |
|---|---|---|
| `--flashnext` | The whole setup: streaming, big batches, the right context and cache settings | Upstream's defaults don't stream, so the model won't run well on 64 GB |
| A 28 GiB expert cache (`--moe-stream-cache 28`) | The biggest single speed setting | Set the GiB to fit how much memory you want spare: 24 GiB gives about 4 GiB back and writes about 6-8% slower |
| The MTP draft head (download, 1.9 GB) | **+23%** writing on agent work (tool calls, code); about even on plain prose | Plain one-token-at-a-time writing |
| The two trained guessers (downloads, 121 + 123 MB) | About **+12%** coding, **+16%** chat (an estimate from two runs) | The engine's own rougher guess |
| A second SSD | **+15%** writing, **-31%** time to first token | One drive does all the reading |

These were measured one at a time, months apart, so don't just add them up. Each one counts, though.

## Guides

- [Getting started](docs/flashnext/getting-started.md): the steps above in detail, the second drive, and what a healthy start looks like.
- [Settings](docs/flashnext/settings.md): every option for this model, with its default, when to change it, and how to set it on the command line, as an environment variable or in an INI preset file.
- [Memory sizing](docs/flashnext/memory-sizing.md): choosing the expert cache and context on 64 GB and 48 GB Macs, and how to watch for swap.
- [The chat window](docs/flashnext/chat-window.md): sharing one server between chat and a coding agent.
- [Results](docs/flashnext/results.md): the measurements behind this README, and how they were taken.

The rest of llama.cpp works as upstream documents it: see [docs/](docs/) and
[the upstream README](https://github.com/ggml-org/llama.cpp#readme). npanj's notes on the model's
architecture and kernels are in [docs/Qwen3.8-Flash-Next.md](docs/Qwen3.8-Flash-Next.md), and the
README of his fork is kept as [docs/npanj-fork-README.md](docs/npanj-fork-README.md).

## Credits

This is a fork of a fork of a fork, and it stands on the work underneath it:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml, by Georgi Gerganov and the
  llama.cpp contributors.
- [mihailescu2m/llama.cpp](https://github.com/mihailescu2m/llama.cpp) by Marian Mihailescu, the
  origin of MoE expert streaming (`--moe-stream`), phase-aware batching, the SSD context cache and
  MTP rejection sampling. Without that work none of this runs.
- [npanj/llama.cpp](https://github.com/npanj/llama.cpp), which brought streaming to
  Qwen3.8-Flash-Next on Apple Silicon: the model's graph and indexer work, the native MTP head, the
  Metal kernels, and the shared MTP draft head used here
  ([nitinpanj/qwen38-flash-next-v3](https://huggingface.co/nitinpanj/qwen38-flash-next-v3)).
- [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) for the UD-Q4_K_XL quant, and
  the Qwen team for the model.

This fork adds two-drive striping, the reading room, the measured-depth draft head and its token
list, union attention, the n-gram shelf, the pinned checkpoint, the cheaper per-layer stops, the
chat window, the trained two-layer lookahead and its keep rule, the trained one-layer guess, the
drive choice by load, fewer undo copies, chat's opening copy and the pin that keeps it whichever
request came first, the `--flashnext` preset, and the study that measured them all, Marian's
LRU rule for the expert cache included.

## Licence

MIT, as upstream: see [LICENSE](LICENSE). Bundled third-party code keeps its own licences, listed
in [licenses/](licenses/).

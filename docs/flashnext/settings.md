# Settings

Every option you would set to run Qwen3.8-Flash-Next with this fork: what it does, its default,
when to change it, and how to set it. The recommended values are the study's everyday setup
([Results](results.md)). For memory sizing (the expert cache and the context), see
[Memory sizing](memory-sizing.md).

- [Three ways to set an option](#three-ways-to-set-an-option)
- [Defaults and `--flashnext`](#defaults-and---flashnext)
- [Expert streaming and the expert cache](#expert-streaming-and-the-expert-cache)
- [Two drives](#two-drives)
- [The reading room (prompt reading)](#the-reading-room-prompt-reading)
- [The n-gram shelf](#the-n-gram-shelf)
- [The MTP draft head](#the-mtp-draft-head)
- [Images](#images)
- [Context, attention and slots](#context-attention-and-slots)
- [The prompt cache](#the-prompt-cache)
- [The chat window](#the-chat-window)
- [Thinking](#thinking)
- [Switches for A/B testing](#switches-for-ab-testing)
- [A complete example in all three forms](#a-complete-example-in-all-three-forms)

## Three ways to set an option

Most options are ordinary `llama-server` flags. Each one also has an environment variable
(`LLAMA_ARG_...`) and an INI key (the long flag name without its dashes):

| Form | Example |
|---|---|
| Command line | `--moe-stream-cache 28` |
| Environment variable | `LLAMA_ARG_MOE_STREAM_CACHE=28` |
| INI key | `moe-stream-cache = 28` |

A flag that takes no value is written `= true` in an INI file and `=1` as a variable
(`moe-stream = true`, `LLAMA_ARG_MOE_STREAM=1`). The command line wins over the environment.

INI keys work in two files:

- **A model preset file**, `llama-server --models-preset flashnext.ini`. This is llama-server's
  router mode: the router starts a child server for the model with the section's settings
  ([upstream docs](../../tools/server/README.md#model-presets)). Start the router with
  `--models-max 1` so only one model is ever loaded. The study itself ran plain single-model servers, so
  router mode with this model is untested here.
- **A config file** read by every llama.cpp program at start:
  `~/.config/llama.cpp/config.ini`, `[*]` section only ([docs/preset.md](../preset.md)).

Some switches are **environment variables only**: those without `ARG` in the name, such as
`LLAMA_SPEC_ADAPTIVE_RATE` or `GGML_METAL_RESIDENCY_KEEP_ALIVE_S`. They cannot go in an INI file.
Set them in the shell that starts `llama-server`; router mode passes its own environment on to the
child server. A variable that is unset takes the default given below; write `0` to turn an
"on by default" switch off.

## Defaults and `--flashnext`

Once a change has passed the study's bar, it becomes the engine's default, unless it needs a
download. So the fork's own switches, left unset, run the study's everyday values, and the old way
is one value away:

| Variable | Unset (the default) | The old way |
|---|---|---|
| `LLAMA_MOE_STREAM_LRU` | on: a full cache lets go the expert unused longest | `0`: the lowest recent use count |
| `LLAMA_MOE_STREAM_DESK_FIRST` | on: the GPU starts on the cached experts while the missing ones are read | `0`: each layer waits for every missing expert first |
| `LLAMA_RS_SKIP_SLOTS` | on: only the undo copies a rollback can reach | `0`: every copy |
| `LLAMA_SPEC_ADAPTIVE_RATE` | `1`: the draft head's measured depth | `0`: the older climb-table rule |
| `LLAMA_SPEC_CHAIN_P` | with `--spec-type draft-dflash`: `0.5`, the [chain stop](#the-chain-stop-and-the-draft-trace); with any other draft: off | `0`: off |
| `LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB` | `4096`: the cache claimed in 4 GiB steps | `0`: all at once |
| `GGML_METAL_RESIDENCY_KEEP_ALIVE_S` | `10000000`: the model stays wired while idle | `0`: upstream's 180 s |
| `GGML_METAL_STAFF_MV` | off: the check's matrix work on the measured route | `1`: the M5's matrix units |
| `LLAMA_MOE_STREAM_LOOKAHEAD_ALL`, `_DEPTH2`, `_KEEP`, `_CONF` | with the two-layer guesser named (`LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER`): `1`, `4`, `1`, `0.2`; without it: off | set each by hand |

The downloads stay yours to add: the draft head (`-md`), the two trained guessers (by their
variables) and the image encoder (`--mmproj`). So do the choices that depend on your Mac: the
expert cache, the second drive, the context, the chat window and the opening copy.

**`--flashnext`** (`LLAMA_ARG_FLASHNEXT=1`) sets llama.cpp's own options to the everyday values,
each only where you did not set that option yourself, on the command line, as its `LLAMA_ARG_*`
variable or in a config file. Without it, upstream's defaults are untouched.

| `--flashnext` sets | Upstream's default |
|---|---|
| `-ngl 99` | auto |
| `--moe-stream --moe-stream-direct --moe-stream-io-threads 8` | off, off, 9 |
| `-c 200000` | the model's training context |
| `-b 4096 -ub 4096` | 2048, 512 |
| `-cms 8192 -ctxcp 3` | 8192, 32 |
| `-np 1` | auto (4 slots) |
| `-fa on` | auto |
| `--cache-reuse 0` | 0 |
| `--cache-ram 400 --cache-ram-opening` | 8192 MiB, off |
| `--chat-window 90` | 0 (off) |
| `--jinja --reasoning-format deepseek` | on, `deepseek` |
| with a draft model (`-md`) only: `--spec-type draft-mtp-adaptive --spec-draft-n-max 5 --spec-draft-p-min 0.3 --spec-draft-ngl 99 --spec-max-prompt 0` | none, 3, 0.0, auto, 0 |
| with your `--spec-type draft-dflash` and `-md`: `--spec-draft-n-max 7 --spec-draft-p-min 0` in place of the head's 5 and 0.3 ([DFlash](#a-dflash-draft-instead-of-the-head)) | 3, 0.0 |

It never sets the expert cache (`--moe-stream-cache`), the files or the address. These are the
study's everyday values: its own server runs exactly these. Give any of them
yourself and `--flashnext` keeps yours.

At start the server prints one line that begins `flashnext:` and names the engine's switches in
force (yours marked "(yours)"), the guesser's tuning, and what `--flashnext` set and kept, for
example:

```
flashnext: engine defaults: LLAMA_MOE_STREAM_LRU=1, LLAMA_RS_SKIP_SLOTS=1, ...; the guesser's tuning: LLAMA_MOE_STREAM_LOOKAHEAD_ALL=1, ...; preset --flashnext: on, set -ngl 99 --moe-stream ..., yours kept: -c
```

## Expert streaming and the expert cache

The model has 48 layers of 512 experts, and each token uses 10 per layer. With streaming on,
the experts stay on the SSD and an expert cache in RAM keeps the ones used lately; a token whose
expert is missing waits for it to be read (a cache miss). Everything else (attention, the shared
weights, the router) stays resident.

| Option | Env var | Default | What it does |
|---|---|---|---|
| `--moe-stream` | `LLAMA_ARG_MOE_STREAM` | off; **`--flashnext`**: on | Stream the routed experts from disk. Any `--moe-stream-*` flag below turns it on too. |
| `--moe-stream-cache N` | `LLAMA_ARG_MOE_STREAM_CACHE` | auto (20 experts per layer); `--flashnext` leaves it to you | The expert cache: `28` = 28 GiB, or `200s` = 200 experts per layer. **The biggest speed setting.** Change it to suit how much memory you want spare. Study: **28** with the draft head on. See [Memory sizing](memory-sizing.md). |
| `--moe-stream-io-threads N` | `LLAMA_ARG_MOE_STREAM_IO_THREADS` | 9 (max 18); **`--flashnext`**: 8 | Parallel expert reads. Study: **8** (npanj measured 8, 12 and 16 the same). |
| `--moe-stream-direct` | `LLAMA_ARG_MOE_STREAM_DIRECT` | off; **`--flashnext`**: on | Read experts past macOS's file cache (`F_NOCACHE`), so cached copies of experts don't crowd out RAM the expert cache and the system need. **Always set it.** |
| `-ngl 99` | `LLAMA_ARG_N_GPU_LAYERS` | auto; **`--flashnext`**: 99 | Everything that is not streamed goes on the GPU. |

Environment-only switches:

| Variable | Default | What it does |
|---|---|---|
| `LLAMA_MOE_STREAM_LRU` | on | Which expert a full cache lets go. On: the one unused longest (least recently used). `0`: the old rule, the lowest recent use count. Measured at a 28 GiB cache: +9.6% writing on code, +9.7% on chat, 17-28% fewer experts fetched per written token; at 24 GiB it writes as fast as the old rule at 28. Never changes the words. |
| `LLAMA_MOE_STREAM_DESK_FIRST` | on | While writing (and for the draft head's checks, up to 16 tokens), a layer whose experts are not all in the cache no longer waits for the missing ones before the GPU starts: it works on the cached experts and the shared expert while they are read, then on the rest as they land. Measured at a 28 GiB cache: +9.1% writing on chat, +4.9% on code, with 62-66% of the wait for missing experts hidden. Exact by construction: never changes the words. `0` is the old way, every missing expert first. Off by itself with `LLAMA_MOE_STREAM_GPU_SLOT`, `LLAMA_MOE_STREAM_STATE_LOG`, or a cache the GPU does not read; the log says `desk first: on` or why it is off. |
| `LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB` | 4096 | Claim the expert cache in steps of this many MiB, with a pause between steps (`LLAMA_MOE_STREAM_ALLOC_PAUSE_MS`, 500 ms when chunking). Opening a 28 GiB cache at once drops macOS's free memory in seconds, and the study's only spill inside a test came at such an opening. `0` claims it all at once (the old way). |
| `GGML_METAL_RESIDENCY_KEEP_ALIVE_S` | 10000000 | Metal un-wires the model's buffers after this many idle seconds, and the next request wires ~41 GiB back at once, a small rush. The default keeps it wired for good. `0` is upstream's 180 s, if you want the memory back when the server is idle. |
| `LLAMA_MOE_STREAM_LOOKAHEAD` | the model's experts per token (10) | While writing, fetch this many of the next layer's likely experts ahead of time (about 1 in 4 is used before the cache lets it go). `0` turns it off. Lookahead makes the cache's contents depend on timing, but never changes the words. |
| `LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER` | none | The trained two-layer guesser's file ([crystal-ball](https://huggingface.co/skeggsguy/crystal-ball), `c1-guesser.gguf`, 121 MB). Each layer scores all 512 experts of the layer after next and fetches the confident ones early. Naming it gives the four switches below their everyday values; it needs the lookahead on (its default) and `_DEPTH2` above 0, and refuses a file of the wrong shape. Never changes the words. Study: **on** (+4.2% writing on code, +8.2% on chat). |
| `LLAMA_MOE_STREAM_LOOKAHEAD_CONF` | 0.2 with the guesser named | With the guesser: fetch the experts whose confidence is above this, a number in (0, 1]; `0` takes a fixed number of picks instead. Study: **0.2** (0.1 fetched more and gained less). |
| `LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2` | 4 with the guesser named; 0 (off) without | Also fetch two layers ahead. Without the guesser, the K most likely experts by the next layer's router (set by hand; the study did not keep it); with it, the guesser's confident picks. Study: **4**. |
| `LLAMA_MOE_STREAM_LOOKAHEAD_ALL` | on with the guesser named; off without | Look ahead for every token of a small batch (the draft head's check), not only the last. Study: **1**. |
| `LLAMA_MOE_STREAM_LOOKAHEAD_KEEP` | on with the guesser named; off without | Keep a guessed expert in the cache until its layer has run, so a right guess is not evicted unread. The cache still takes a kept slot rather than wait. Study: **1**: the keep rule is what turns the guesser's accuracy into speed. |
| `LLAMA_MOE_STREAM_GUESSER1` | none (the engine's own guess) | The trained one-layer guesser's file ([crystal-ball](https://huggingface.co/skeggsguy/crystal-ball), `c1-guesser-gap1.gguf`, 123 MB). The lookahead's next-layer guess then comes from this file instead of the next layer's own router fed this layer's input: 73-77% of the next layer's experts in its top 10, against 61-65%, on held-out conversations. Needs the lookahead on (its default); refuses a file of the wrong shape and the GPU slot servicer (`LLAMA_MOE_STREAM_GPU_SLOT`). Never changes the words. Study: **on**, beside the two-layer guesser and the keep rule (+7.5% writing on code, +7.3% on chat, on top of them). |
| `LLAMA_MOE_STREAM_STATS_MS` | 0 (off) | Print streaming statistics (cache hits, reads, the reading room) every N ms. Handy while tuning: `1000`. |

## Two drives

A byte-identical copy of the model on a second SSD lets two drives read at once. Each read goes to
the drive with fewer reads waiting; the split by expert id below breaks ties, so it still sets each
drive's share when both are equally busy.

| Option | Env var | Default | What it does |
|---|---|---|---|
| `--moe-stream-alt-path FILE` | `LLAMA_ARG_MOE_STREAM_ALT_PATH` | none | The **first** file of the copy on the other drive; its other files are found beside it. The server refuses to start if any copy's size differs. |
| `--moe-stream-alt-split N` | `LLAMA_ARG_MOE_STREAM_ALT_SPLIT` | 53 | Percent of expert ids read from the `-m` copy (1-99); the rest come from the alt copy. |

**Choosing the split:** give each drive its share of the combined read speed. Here the internal
drive reads 6.5 GB/s and the external 5.7 GB/s: 6.5 / 12.2 ≈ 53%. Measure your drives with a large
sequential read (for example Blackmagic Disk Speed Test) and round.

Measured (study, final setup): writing +15%, time to first token −31%, the cache's hit rate
unchanged. A second drive is worth more than a few GiB of cache.

Environment-only (both on unless set to `0`; they choose where bytes are read from, never which):

| Variable | Default | What it does |
|---|---|---|
| `LLAMA_MOE_STREAM_DYN_DRIVE` | on | Send each read to the drive with fewer reads in flight, not the drive its expert id names. Measured with the second drive: +12.5% writing on code, +10.4% on chat, together with `_QOS`. |
| `LLAMA_MOE_STREAM_QOS` | on | Run the read threads at macOS's interactive service class. |

`LLAMA_MOE_STREAM_SPLIT` (off unless set) reads one missing expert from both drives at once: `1`
sends its three parts to the two drives in turn, `2` reads each part in two halves, one from each
drive. Neither cut the time the GPU waits for experts in the study's probe (7.8 and 6.8 ms a token
against 6.9 ms off), so the study leaves it off.

## The reading room (prompt reading)

A long prompt needs nearly every expert on every layer. The reading room is a ring buffer carved
out of the expert cache: for a batch of 1,024+ tokens the drives fill it with every expert the cache
lacks, layer after layer, ahead of the GPU, and each layer's experts are handed back in parts as
soon as the GPU is done. Prompt reading then costs the slower of the drives and the GPU, not both,
and the cache's contents are left alone. Results are bit-identical to having every expert in RAM.
While writing, the buffer is lent back to hold experts the cache evicts (+5.5% writing).

| Option | Env var | Default | What it does |
|---|---|---|---|
| `--moe-stream-room MODE` | `LLAMA_ARG_MOE_STREAM_ROOM` | `auto` | `auto` = 1.25 layers of look-ahead (about 1.4 GiB for this model), `2f` = 2 layers, `1.5` = 1.5 GiB, `0` = off. Under `auto` it turns itself off, with a warning, where it can't be made (for example `-ub` under 1,024); asked for explicitly, it refuses to load instead. |
| `--moe-stream-room-parts N` | `LLAMA_ARG_MOE_STREAM_ROOM_PARTS` | 4 | Parts each layer is cut into on the belt, 1-16. |
| `-b 4096 -ub 4096` | `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH` | 2048 / 512; **`--flashnext`**: 4096 / 4096 | Prompt batch sizes. The reading room needs `-ub` of at least 1,024; 4096 is the study's. 8192 read 27% faster but costs ~4.5 GiB more. |

Environment-only: `LLAMA_MOE_ROOM_LEND=0` stops lending the buffer back while writing;
`LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS=N` moves the batch size from which the room is used (default
about 20 tokens per expert, 1,024 for this model). Prompts shorter than that read in "waves"
through the cache instead, which is faster for them.

Measured (study): reading in +32% and time to first token −28% over 12 real conversations; under
300-1,000 tokens it was 16% slower, which is why short batches keep waves.

## The n-gram shelf

Qwen3.8-Flash-Next has a large n-gram embedding table (the "PLE" table, 28.8 GB in this quant)
read a few rows per token. The shelf keeps the rows in use in a fixed buffer, read past the file
cache, instead of leaving whole pages of the table in macOS's file cache.

| Option | Env var | Default | What it does |
|---|---|---|---|
| `--ple-shelf SIZE` | `LLAMA_ARG_PLE_SHELF` | `auto` (128 MiB) | `auto`, a size in MiB (`256`, `256M`) or GiB (`1G`), or `0` for off. |

Measured: writing +3.0%, reading in +3.2%. The rows in use over 36 conversations came to ~74 MiB,
so 128 MiB is plenty.

## The MTP draft head

The model has a native multi-token-prediction (MTP) head. As a draft model it guesses the next few
tokens, and the main model checks them all in one pass: a right guess is nearly free. Here it is
loaded from npanj's shared head file.

| Option | Env var | Default | What it does |
|---|---|---|---|
| `-md FILE` (`--spec-draft-model`) | `LLAMA_ARG_SPEC_DRAFT_MODEL` | none | The draft head, `mtp-shared-Q4_K_M.gguf`. |
| `--spec-type TYPE` | `LLAMA_ARG_SPEC_TYPE` | none; **`--flashnext`** with `-md`: `draft-mtp-adaptive` | `draft-mtp-adaptive`: the head with a depth that adapts. (`draft-mtp` is a fixed depth.) |
| `--spec-draft-n-max N` | `LLAMA_ARG_SPEC_DRAFT_N_MAX` | 3; **`--flashnext`** with `-md`: 5 | The most tokens guessed per step. Study: **5**. 30 was far slower. |
| `--spec-draft-n-min-adaptive N` | `LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE` | 3 | The climb table's starting and lowest depth (used with `LLAMA_SPEC_ADAPTIVE_RATE=0`); must be 1 to n-max. |
| `--spec-draft-p-min P` | `LLAMA_ARG_SPEC_DRAFT_P_MIN` | 0.0; **`--flashnext`** with `-md`: 0.3 | Stop guessing when the head's confidence in a token falls below P. Study: **0.3**. |
| `--spec-draft-ngl 99` | `LLAMA_ARG_N_GPU_LAYERS_DRAFT` | auto; **`--flashnext`** with `-md`: 99 | The head on the GPU. |
| `--spec-max-prompt 0` | `LLAMA_ARG_SPEC_MAX_PROMPT` | 0 (no limit); **`--flashnext`** with `-md`: 0 | **Keep 0.** A number turns drafting off for any prompt longer than it, and in a chat the whole history counts, so drafting switches off for good once a conversation grows past it. |

Environment-only:

| Variable | Default | What it does |
|---|---|---|
| `LLAMA_SPEC_ADAPTIVE_RATE` | 1 | With `draft-mtp-adaptive`, the depth is measured from the guesses actually kept (depths 0 to n-max, a probe every 12 cycles, remembered across answers). `0`: a fixed climb table instead (the old way). |
| `LLAMA_MTP_VOCAB` | off (the whole vocabulary) | A text file of token ids, one per line (`#` comments allowed): the head guesses only from these, so each guess is cheaper; the words never change. |

**When the draft head pays:** on code, tool calls and structured output, where guesses land
(agent conversations +23%, up to 1.68x on one long tool-heavy reply). On free prose it breaks even:
every guess needs its own experts fetched, right or wrong, so the extra cache misses eat the gain.
If you mostly write prose, try it off (remove `-md`; `--flashnext` then sets no `--spec-*` flag) and keep ~1 GiB.

**It can change the words.** Checking 6 tokens in one batch rounds slightly differently from one
at a time, so at temperature 0 the text can differ from a run without the head. The exam score was
the same (18/20) either way.

Environment-only, with the draft head:

| Variable | Default | What it does |
|---|---|---|
| `LLAMA_RS_SKIP_SLOTS` | on | Before each check the model saves copies of its recurrent state so it can undo rejected guesses. On, it saves only the copies a rollback can reach, and refuses a rollback past them. Byte-identical output; +4.3% writing on code, chat level. `0` saves every copy (the old way). |
| `GGML_METAL_STAFF_MV` | off | `1` puts the check's Q8_0 matrix-vector work for 2-16 tokens on the GPU's matrix units (with Metal 4, M5 on). It did not clear the study's +3% bar (−0.8% and +1.1% on code), so the default is the older route the study's numbers were measured on. |

### A DFlash draft instead of the head

A DFlash draft is a small separate model that drafts a whole block of words in one go (the study
runs incoai's DFlash 2, Q8_0). It takes the head's flags. Give `-md` and `--spec-type draft-dflash`
yourself; `--flashnext` then sets DFlash's own depth and p-min (7 and 0) instead of the head's (5 and
0.3), and the chain stop is on at 0.5 unless you set `LLAMA_SPEC_CHAIN_P`. That is the study's
measured best: +10.8% chat and +8.7% coding over the head (C37, [Results](results.md#dflash-draft-c37)).
Any of the three you set yourself is kept. Without `--flashnext`, n-max and p-min are upstream's (3
and 0) and only the chain stop's default applies:

| Option | What it does |
|---|---|
| `-md FILE` | The DFlash gguf. A file with a Markov head (`markov_w1.weight`) is DSpark, otherwise DFlash. |
| `--spec-type draft-dflash` | The DFlash draft (`draft-dspark` for DSpark). |
| `--spec-draft-n-max N` | The most words drafted per round, at most the draft's trained block less one (more is clamped, with a warning). **`--flashnext`**: 7 (the whole block). Study: 7 with the chain stop, and 3. |
| `--spec-draft-p-min P` | Stop the draft at the first word whose confidence falls below P (`0`: no stop). DSpark needs its confidence head for any P above 0 and for the chain stop. **`--flashnext`**: 0. Study: 0 at n-max 7, 0.3 at n-max 3. |

### The chain stop and the draft trace

Environment-only. The chain stop works for every draft that has a p-min (the head, DFlash, DSpark, EAGLE3).

| Variable | Default | What it does |
|---|---|---|
| `LLAMA_SPEC_CHAIN_P` | `0.5` with `draft-dflash`; off with every other draft | A threshold in (0, 1]. The draft goes on while the product of its words' confidences so far stays at or above it; the word that takes the product under it ends the draft and is not drafted. P-min is checked first, so a word under both is a p-min stop. Unset: 0.5 for `draft-dflash` (the study's measured best with depth 7), off for the head, `draft-simple`, EAGLE3 and DSpark (with the head it stayed under the study's bar, +3.0% chat, +2.8% coding). `""` or `0`: off for any draft. Off, the words are the same as without it. Any other value outside (0, 1] (`1.5`, `0.0`, `abc`) stops the server at start. Under `LLAMA_SPEC_ADAPTIVE_RATE=2` it is ignored, with a warning (the price stops the draft there). The `flashnext:` line shows the value in force (`LLAMA_SPEC_CHAIN_P=0.5 (draft-dflash)`, `=off`, or yours marked "(yours)"), and a draft with the stop on logs `chain stop: on, ...` once at start. |
| `LLAMA_SPEC_RATE_TRACE` | off | A file path: one JSON line per round (`.srtr`, version 2), numbers only, never words: the depth in force, words drafted / checked / kept, each word's confidence and the ones past the stop, why the draft stopped (`cap`, `pmin`, `chain`, `trim`) and the round's times. A header line opens each session with the draft type, its block, n-max, p-min and `chain_p`. The file is appended to, so give each run its own. The head writes its own trace; any other draft is traced around its draft call (a draft model and EAGLE3 report no stops, so theirs read `cap`). Read by the study's `sim/spec_trace.py`. Off, it costs one check a round. |

## Images

With the image encoder loaded the model reads pictures: attach one in the web page at
<http://127.0.0.1:8080>, or send it in a request as an `image_url` part (OpenAI-style) or an
`image` block (Anthropic-style `/v1/messages`). Without it, a request with a picture is refused.

| Option | Env var | Default | What it does |
|---|---|---|---|
| `--mmproj FILE` | `LLAMA_ARG_MMPROJ` | none | The image encoder, `mmproj-BF16.gguf` from Unsloth's repo. About 0.9 GB of memory beside the cache. Study: **on** every day, at a 28 GiB cache with the draft head (an image test: described, no swap). Its effect on writing speed is not measured. |

## Context, attention and slots

| Option | Env var | Default | What it does |
|---|---|---|---|
| `-c N` (`--ctx-size`) | `LLAMA_ARG_CTX_SIZE` | the model's training context; **`--flashnext`**: 200000 | Context length in tokens. Memory grows about 27 KB per token. Study: 98,304 in tests, **200,000** every day. |
| `-np 1` (`--parallel`) | `LLAMA_ARG_N_PARALLEL` | auto (4 slots); **`--flashnext`**: 1 | **Keep 1.** Each extra slot costs working memory that comes out of the expert cache. |
| `-fa on` | `LLAMA_ARG_FLASH_ATTN` | auto; **`--flashnext`**: on | **Required**: reading a prompt with flash attention off aborts on this model's sparse attention layers. |

Environment-only:

| Variable | Default | What it does |
|---|---|---|
| `LLAMA_QSA_UNION` | `1` (union attention) | How the sparse attention reads a prompt. `1`: one shared, deduplicated block list per batch: least memory (1.9 GiB of working memory at 160K) and fastest, but a different summation order, so words can differ at near-ties (perplexity unchanged). `bias`: exact, 4.5 GiB. `0`: the original masked path, exact, 5.6 GiB. |

## The prompt cache

After each reply the server keeps the conversation's state, so the next request reads in only
the new text. This model is a hybrid: 12 layers keep a normal KV cache, and 36 keep a fixed-size
recurrent state that cannot be rewound. Saved copies of that state (checkpoints) let the server
step back to where a new request differs, such as an edited message.

| Option | Env var | Default | What it does |
|---|---|---|---|
| `-ctxcp N` (`--ctx-checkpoints`) | `LLAMA_ARG_CTX_CHECKPOINTS` | 32; **`--flashnext`**: 3 | Checkpoints kept per slot. Each costs ~113 MiB + ~2 MiB per 1K tokens of context, so 32 is far too many here. Study: **3** (one pinned, two rolling). |
| `-cms N` (`--checkpoint-min-step`) | `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT` | 8192; **`--flashnext`**: 8192 | The fewest tokens between checkpoints. 8192 instead of 512 cut time to first token by 17%, because a long prompt is read in fewer, bigger pieces. |
| `--cache-ram N` | `LLAMA_ARG_CACHE_RAM` | 8192 (MiB); **`--flashnext`**: 400 | An extra prompt cache in RAM for other conversations. On 64 GB that memory is better spent on the expert cache, so the study keeps **400** with `--cache-ram-opening`: room for chat's opening only. |
| `--cache-ram-opening` | `LLAMA_ARG_CACHE_RAM_OPENING` | off; **`--flashnext`**: on | Keep only one RAM copy: chat's opening (its system prompt and tools, ~232 MiB here), never a coding agent's conversation. With a coding agent sharing the server, a new chat then starts writing in 4.1 s instead of 14.2 s. It knows a coding agent's requests through the chat window's header. |
| `--cache-reuse 0` | `LLAMA_ARG_CACHE_REUSE` | 0; **`--flashnext`**: 0 | Reuse by shifting the KV cache; keep it off. |

Environment-only: `LLAMA_CKPT_PIN` (default on). One checkpoint stays pinned at the start of the
message where the latest two conversations part, so a new chat that shares a long system prompt
and tools resumes in 4.2 s instead of 14.2 s. It needs `-ctxcp` of 2 or more; `LLAMA_CKPT_PIN=0`
turns it off.

## The chat window

One slot shared by a person chatting and a background client, such as a coding agent. A chat
request cuts a running background request off (a 503 its client retries); background requests wait
until the window lapses. Off by default. Full guide: [The chat window](chat-window.md).

| Option | Env var | Default | What it does |
|---|---|---|---|
| `--chat-window SECONDS` | `LLAMA_ARG_CHAT_WINDOW` | 0 (off) | How long chat keeps the slot after its last reply. Study: **300** (5 minutes). |
| `--chat-window-header NAME` | `LLAMA_ARG_CHAT_WINDOW_HEADER` | `X-Lane` | The header that marks a background request. |
| `--chat-window-background VALUE` | `LLAMA_ARG_CHAT_WINDOW_BACKGROUND` | `code` | The header value for background requests. Every other request is chat. |
| `--sse-ping-interval N` | `LLAMA_ARG_SSE_PING_INTERVAL` | 30 | Seconds between keep-alive comments to a streamed request that is waiting. |

## Thinking

| Option | Env var | Default | What it does |
|---|---|---|---|
| `--jinja` | `LLAMA_ARG_JINJA` | on; **`--flashnext`**: on | Use the model's own chat template (needed for tool calls). |
| `--reasoning-format deepseek` | `LLAMA_ARG_THINK` | `deepseek`; **`--flashnext`**: `deepseek` | Return the model's thinking in `reasoning_content`, apart from the answer. |
| `--reasoning-budget N` | `LLAMA_ARG_THINK_BUDGET` | -1 (unlimited) | End the thinking after N tokens. Thinking costs the same per token as the answer. |
| `--reasoning-budget-message TEXT` | `LLAMA_ARG_THINK_BUDGET_MESSAGE` | none | Text added just before the thinking is ended at the budget. |

## Switches for A/B testing

These are on by default, measured and kept. Write `0` to turn one off when comparing:

| Variable | What it does |
|---|---|
| `LLAMA_MOE_STREAM_NOLOCK` | No lock at a layer when none of its experts is missing. |
| `LLAMA_MOE_STREAM_ONE_LOOKUP` | One cache lookup per layer instead of two. |
| `LLAMA_MOE_STREAM_KEEP_AWAKE` | A tiny GPU ping every 1,000 µs (or the value given) while waiting, because the M5 Pro's GPU powers down after ~1.75 ms idle and costs ~0.4 ms to wake. |
| `GGML_METAL_ENCODE_AHEAD` | Encode the next layer's GPU work while this one runs. |
| `LLAMA_MOE_STREAM_PARTITION` | In a wave, give each (token, expert) pair to exactly one wave. |

Together the first four gave +7% writing (+7% to +13% depending on the kind of text). None of them changes the words.

## A complete example in all three forms

The study's everyday server, with the trained guessers, a second drive and the chat window. The
engine's defaults and `--flashnext` carry everything else.

**Command line:**

```sh
LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER=$HOME/models/flashnext/c1-guesser.gguf \
LLAMA_MOE_STREAM_GUESSER1=$HOME/models/flashnext/c1-guesser-gap1.gguf \
./build/bin/llama-server --flashnext \
  -m ~/models/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md ~/models/flashnext/mtp-shared-Q4_K_M.gguf \
  --moe-stream-cache 28 \
  --moe-stream-alt-path /Volumes/ssd/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --moe-stream-alt-split 53
```

**Environment variables** (for example in a launchd job or a shell script; the paths must be
absolute, since `~` is not expanded inside a variable):

```sh
export LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER=/Users/you/models/flashnext/c1-guesser.gguf
export LLAMA_MOE_STREAM_GUESSER1=/Users/you/models/flashnext/c1-guesser-gap1.gguf
export LLAMA_ARG_FLASHNEXT=1
export LLAMA_ARG_MODEL=/Users/you/models/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
export LLAMA_ARG_SPEC_DRAFT_MODEL=/Users/you/models/flashnext/mtp-shared-Q4_K_M.gguf
export LLAMA_ARG_MOE_STREAM_CACHE=28
export LLAMA_ARG_MOE_STREAM_ALT_PATH=/Volumes/ssd/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
export LLAMA_ARG_MOE_STREAM_ALT_SPLIT=53
./build/bin/llama-server
```

**A model preset file**, `flashnext.ini`, started with
`LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER=$HOME/models/flashnext/c1-guesser.gguf LLAMA_MOE_STREAM_GUESSER1=$HOME/models/flashnext/c1-guesser-gap1.gguf ./build/bin/llama-server --models-preset flashnext.ini --models-max 1`
(the environment-only switches go on the command line, as before; requests name the model
`flashnext`):

```ini
version = 1

[flashnext]
flashnext = true
model = /Users/you/models/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
model-draft = /Users/you/models/flashnext/mtp-shared-Q4_K_M.gguf
moe-stream-cache = 28
moe-stream-alt-path = /Volumes/ssd/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
moe-stream-alt-split = 53
chat-window = 300
cache-ram = 400
cache-ram-opening = true
load-on-startup = true
```

In router mode the router itself also runs, so check the first start's memory use before relying
on it with a 28 GiB cache.

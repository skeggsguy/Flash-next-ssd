# Settings

Every option you would set to run Qwen3.8-Flash-Next with this fork: what it does, its default,
when to change it, and how to set it. The recommended values are the study's everyday setup
([Results](results.md)). For memory sizing (the expert cache and the context), see
[Memory sizing](memory-sizing.md).

- [Three ways to set an option](#three-ways-to-set-an-option)
- [Expert streaming and the expert cache](#expert-streaming-and-the-expert-cache)
- [Two drives](#two-drives)
- [The reading room (prompt reading)](#the-reading-room-prompt-reading)
- [The n-gram shelf](#the-n-gram-shelf)
- [The MTP draft head](#the-mtp-draft-head)
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

## Expert streaming and the expert cache

The model has 48 layers of 512 experts, and each token uses 10 per layer. With streaming on,
the experts stay on the SSD and an expert cache in RAM keeps the ones used lately; a token whose
expert is missing waits for it to be read (a cache miss). Everything else (attention, the shared
weights, the router) stays resident.

| Option | Env var | Default | What it does |
|---|---|---|---|
| `--moe-stream` | `LLAMA_ARG_MOE_STREAM` | off | Stream the routed experts from disk. Any `--moe-stream-*` flag below turns it on too. |
| `--moe-stream-cache N` | `LLAMA_ARG_MOE_STREAM_CACHE` | auto (20 experts per layer, far too small) | The expert cache: `28` = 28 GiB, or `200s` = 200 experts per layer. **The biggest speed setting.** Study: **28** with the draft head on. See [Memory sizing](memory-sizing.md). |
| `--moe-stream-io-threads N` | `LLAMA_ARG_MOE_STREAM_IO_THREADS` | 9 (max 18) | Parallel expert reads. Study: **8** (npanj measured 8, 12 and 16 the same). |
| `--moe-stream-direct` | `LLAMA_ARG_MOE_STREAM_DIRECT` | off | Read experts past macOS's file cache (`F_NOCACHE`), so cached copies of experts don't crowd out RAM the expert cache and the system need. **Always set it.** |
| `-ngl 99` | `LLAMA_ARG_N_GPU_LAYERS` | auto | Everything that is not streamed goes on the GPU. |

Environment-only switches:

| Variable | Default | What it does |
|---|---|---|
| `LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB` | 0 (claim the whole cache at once) | Claim the expert cache in steps of this many MiB, with a pause between steps (`LLAMA_MOE_STREAM_ALLOC_PAUSE_MS`, 500 ms when chunking). Opening a 28 GiB cache at once drops macOS's free memory in seconds, and the study's only spill inside a test came at such an opening. Study: **4096**. |
| `GGML_METAL_RESIDENCY_KEEP_ALIVE_S` | 180 | Metal un-wires the model's buffers after this many idle seconds, and the next request wires ~41 GiB back at once, a small rush. Study: **10000000** (keep it wired for good). Leave the default if you want the memory back when the server is idle. |
| `LLAMA_MOE_STREAM_LOOKAHEAD` | the model's experts per token (10) | While writing, fetch this many of the next layer's likely experts ahead of time (about 72% turn out right). `0` turns it off. Lookahead makes the cache's contents depend on timing, but never changes the words. |
| `LLAMA_MOE_STREAM_STATS_MS` | 0 (off) | Print streaming statistics (cache hits, reads, the reading room) every N ms. Handy while tuning: `1000`. |

## Two drives

A byte-identical copy of the model on a second SSD lets two drives read at once. The split is by
expert id, so it is stable and proportional, not by how often an expert is used.

| Option | Env var | Default | What it does |
|---|---|---|---|
| `--moe-stream-alt-path FILE` | `LLAMA_ARG_MOE_STREAM_ALT_PATH` | none | The **first** file of the copy on the other drive; its other files are found beside it. The server refuses to start if any copy's size differs. |
| `--moe-stream-alt-split N` | `LLAMA_ARG_MOE_STREAM_ALT_SPLIT` | 53 | Percent of expert ids read from the `-m` copy (1-99); the rest come from the alt copy. |

**Choosing the split:** give each drive its share of the combined read speed. Here the internal
drive reads 6.5 GB/s and the external 5.7 GB/s: 6.5 / 12.2 ≈ 53%. Measure your drives with a large
sequential read (for example Blackmagic Disk Speed Test) and round.

Measured (study, final setup): writing +15%, time to first token −31%, the cache's hit rate
unchanged. A second drive is worth more than a few GiB of cache.

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
| `-b 4096 -ub 4096` | `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH` | 2048 / 512 | Prompt batch sizes. The reading room needs `-ub` of at least 1,024; 4096 is the study's. 8192 read 27% faster but costs ~4.5 GiB more. |

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
| `--spec-type TYPE` | `LLAMA_ARG_SPEC_TYPE` | none | `draft-mtp-adaptive`: the head with a depth that adapts. (`draft-mtp` is a fixed depth.) |
| `--spec-draft-n-max N` | `LLAMA_ARG_SPEC_DRAFT_N_MAX` | 3 | The most tokens guessed per step. Study: **5**. 30 was far slower. |
| `--spec-draft-n-min-adaptive N` | `LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE` | 3 | The climb table's starting and lowest depth (used without `LLAMA_SPEC_ADAPTIVE_RATE`); must be 1 to n-max. |
| `--spec-draft-p-min P` | `LLAMA_ARG_SPEC_DRAFT_P_MIN` | 0.0 | Stop guessing when the head's confidence in a token falls below P. Study: **0.3**. |
| `--spec-draft-ngl 99` | `LLAMA_ARG_N_GPU_LAYERS_DRAFT` | auto | The head on the GPU. |
| `--spec-max-prompt 0` | `LLAMA_ARG_SPEC_MAX_PROMPT` | 0 (no limit) | **Keep 0.** A number turns drafting off for any prompt longer than it, and in a chat the whole history counts, so drafting switches off for good once a conversation grows past it. |

Environment-only:

| Variable | Default | What it does |
|---|---|---|
| `LLAMA_SPEC_ADAPTIVE_RATE` | off | `1`: the depth is measured from the guesses actually kept (depths 0 to n-max, a probe every 12 cycles, remembered across answers). Without it, `draft-mtp-adaptive` follows a fixed climb table. Study: **1**. |
| `LLAMA_MTP_VOCAB` | off (the whole vocabulary) | A text file of token ids, one per line (`#` comments allowed): the head guesses only from these. A 96K list built from the study's own outputs plus public code and prose gave +2.6% writing. The list is specific to the text it was built from, so build one from text like yours, or leave it off. |

**When the draft head pays:** on code, tool calls and structured output, where guesses land
(agent conversations +23%, up to 1.68x on one long tool-heavy reply). On free prose it breaks even:
every guess needs its own experts fetched, right or wrong, so the extra cache misses eat the gain.
If you mostly write prose, try it off (remove `-md` and the `--spec-*` flags) and keep ~1 GiB.

**It can change the words.** Checking 6 tokens in one batch rounds slightly differently from one
at a time, so at temperature 0 the text can differ from a run without the head. The exam score was
the same (18/20) either way.

## Context, attention and slots

| Option | Env var | Default | What it does |
|---|---|---|---|
| `-c N` (`--ctx-size`) | `LLAMA_ARG_CTX_SIZE` | the model's training context | Context length in tokens. Memory grows about 27 KB per token. Study: 98,304 in tests, **200,000** every day. |
| `-np 1` (`--parallel`) | `LLAMA_ARG_N_PARALLEL` | auto (4 slots) | **Keep 1.** Each extra slot costs working memory that comes out of the expert cache. |
| `-fa on` | `LLAMA_ARG_FLASH_ATTN` | auto | **Required**: reading a prompt with flash attention off aborts on this model's sparse attention layers. |

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
| `-ctxcp N` (`--ctx-checkpoints`) | `LLAMA_ARG_CTX_CHECKPOINTS` | 32 | Checkpoints kept per slot. Each costs ~113 MiB + ~2 MiB per 1K tokens of context, so 32 is far too many here. Study: **3** (one pinned, two rolling). |
| `-cms N` (`--checkpoint-min-step`) | `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT` | 8192 | The fewest tokens between checkpoints. 8192 instead of 512 cut time to first token by 17%, because a long prompt is read in fewer, bigger pieces. |
| `--cache-ram N` | `LLAMA_ARG_CACHE_RAM` | 8192 (MiB) | An extra prompt cache in RAM for other conversations. **Set 0** on 64 GB: that memory is better spent on the expert cache. |
| `--cache-reuse 0` | `LLAMA_ARG_CACHE_REUSE` | 0 | Reuse by shifting the KV cache; keep it off. |

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
| `--chat-window SECONDS` | `LLAMA_ARG_CHAT_WINDOW` | 0 (off) | How long chat keeps the slot after its last reply. Study: **1200** (20 minutes). |
| `--chat-window-header NAME` | `LLAMA_ARG_CHAT_WINDOW_HEADER` | `X-Lane` | The header that marks a background request. |
| `--chat-window-background VALUE` | `LLAMA_ARG_CHAT_WINDOW_BACKGROUND` | `code` | The header value for background requests. Every other request is chat. |
| `--sse-ping-interval N` | `LLAMA_ARG_SSE_PING_INTERVAL` | 30 | Seconds between keep-alive comments to a streamed request that is waiting. |

## Thinking

| Option | Env var | Default | What it does |
|---|---|---|---|
| `--jinja` | `LLAMA_ARG_JINJA` | on | Use the model's own chat template (needed for tool calls). |
| `--reasoning-format deepseek` | `LLAMA_ARG_THINK` | `deepseek` | Return the model's thinking in `reasoning_content`, apart from the answer. |
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

The study's everyday server, with a second drive and the chat window.

**Command line:**

```sh
LLAMA_SPEC_ADAPTIVE_RATE=1 LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB=4096 \
GGML_METAL_RESIDENCY_KEEP_ALIVE_S=10000000 \
./build/bin/llama-server \
  -m ~/models/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -ngl 99 \
  --moe-stream --moe-stream-cache 28 --moe-stream-io-threads 8 --moe-stream-direct \
  --moe-stream-alt-path /Volumes/ssd/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --moe-stream-alt-split 53 \
  -md ~/models/flashnext/mtp-shared-Q4_K_M.gguf --spec-draft-ngl 99 \
  --spec-type draft-mtp-adaptive --spec-draft-n-max 5 --spec-draft-p-min 0.3 --spec-max-prompt 0 \
  -c 200000 -b 4096 -ub 4096 -cms 8192 -ctxcp 3 -np 1 -fa on \
  --cache-reuse 0 --cache-ram 0 --jinja --reasoning-format deepseek \
  --chat-window 1200 --host 127.0.0.1 --port 8080
```

**Environment variables** (for example in a launchd job or a shell script; the paths must be
absolute, since `~` is not expanded inside a variable):

```sh
export LLAMA_SPEC_ADAPTIVE_RATE=1 LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB=4096
export GGML_METAL_RESIDENCY_KEEP_ALIVE_S=10000000
export LLAMA_ARG_MODEL=/Users/you/models/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
export LLAMA_ARG_N_GPU_LAYERS=99
export LLAMA_ARG_MOE_STREAM=1 LLAMA_ARG_MOE_STREAM_CACHE=28 LLAMA_ARG_MOE_STREAM_IO_THREADS=8
export LLAMA_ARG_MOE_STREAM_DIRECT=1
export LLAMA_ARG_MOE_STREAM_ALT_PATH=/Volumes/ssd/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
export LLAMA_ARG_MOE_STREAM_ALT_SPLIT=53
export LLAMA_ARG_SPEC_DRAFT_MODEL=/Users/you/models/flashnext/mtp-shared-Q4_K_M.gguf
export LLAMA_ARG_N_GPU_LAYERS_DRAFT=99 LLAMA_ARG_SPEC_TYPE=draft-mtp-adaptive
export LLAMA_ARG_SPEC_DRAFT_N_MAX=5 LLAMA_ARG_SPEC_DRAFT_P_MIN=0.3 LLAMA_ARG_SPEC_MAX_PROMPT=0
export LLAMA_ARG_CTX_SIZE=200000 LLAMA_ARG_BATCH=4096 LLAMA_ARG_UBATCH=4096
export LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT=8192 LLAMA_ARG_CTX_CHECKPOINTS=3 LLAMA_ARG_N_PARALLEL=1
export LLAMA_ARG_FLASH_ATTN=on LLAMA_ARG_CACHE_REUSE=0 LLAMA_ARG_CACHE_RAM=0
export LLAMA_ARG_JINJA=1 LLAMA_ARG_THINK=deepseek LLAMA_ARG_CHAT_WINDOW=1200
export LLAMA_ARG_HOST=127.0.0.1 LLAMA_ARG_PORT=8080
./build/bin/llama-server
```

**A model preset file**, `flashnext.ini`, started with
`LLAMA_SPEC_ADAPTIVE_RATE=1 LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB=4096 GGML_METAL_RESIDENCY_KEEP_ALIVE_S=10000000 ./build/bin/llama-server --models-preset flashnext.ini --models-max 1 --host 127.0.0.1 --port 8080`
(the environment-only switches go on the command line, as before; requests name the model
`flashnext`):

```ini
version = 1

[flashnext]
model = /Users/you/models/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
n-gpu-layers = 99
moe-stream = true
moe-stream-cache = 28
moe-stream-io-threads = 8
moe-stream-direct = true
moe-stream-alt-path = /Volumes/ssd/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
moe-stream-alt-split = 53
model-draft = /Users/you/models/flashnext/mtp-shared-Q4_K_M.gguf
spec-draft-ngl = 99
spec-type = draft-mtp-adaptive
spec-draft-n-max = 5
spec-draft-p-min = 0.3
spec-max-prompt = 0
ctx-size = 200000
batch-size = 4096
ubatch-size = 4096
checkpoint-min-step = 8192
ctx-checkpoints = 3
parallel = 1
flash-attn = on
cache-reuse = 0
cache-ram = 0
jinja = true
reasoning-format = deepseek
chat-window = 1200
load-on-startup = true
```

In router mode the router itself also runs, so check the first start's memory use before relying
on it with a 28 GiB cache.

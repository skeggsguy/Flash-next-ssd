# Results

The numbers on these pages come from a benchmark study run on one Mac mini M5 Pro (64 GB) in
September and October 2026. The full write-up, with every run's raw records, will be published
separately; this page summarises it. The last column of each table gives the study's run name, so
each number can be traced in the write-up.

## How it was measured

- **Model:** Qwen3.8-Flash-Next, Unsloth UD-Q4_K_XL (111.3 GB). The internal SSD reads about 6.5 GB/s and
  the external Thunderbolt 5 SSD 5.7 GB/s; read at the same time, 12.3 GB/s.
- **Work:** real agent conversations (coding and tool calls) from the author's own assistant,
  replayed turn by turn with the prompt cache on, as in real use; prose and writing tasks as a
  second set; and long prompts from cold at 4K, 32K and 100K tokens.
- **One change at a time, run A-B-A.** Each run compares one change against the setup before it,
  each arm on a fresh server, with the control run before and after so a warming Mac cannot flatter
  the second arm. The file cache is purged first, and the first request of each server is a
  warm-up that is not counted.
- **Writing** is decode speed, token-weighted: all tokens written over all the time spent writing
  them. **Time to first token** is the median over each conversation's first reply. **Reading a
  prompt** is prefill speed.
- **Swap voids a timing.** A monitor sampled swap, memory pressure and thermal state every second.
  Any swap stopped the run, and the timings it touched were thrown out, never counted as slow.
  Swap stayed at 0 in 57 of the 59 runs, over 13,485 counted replies.
- **Quality:** a fixed 20-task exam with known answers (10 from the assistant's own work, 10 coding
  tasks with unit tests). Changes meant to be exact had to reproduce three fixed answers byte for
  byte; changes that move words were checked with perplexity and the exam.

## Headline

| | Result | Compared with | Run |
|---|---|---|---|
| Writing, 120 agent conversations in order | **17.46 tokens/s** | 11.88 for the study's first setup: **+47%** | H-final vs T-paper |
| Time to first token, median (median prompt: 1,502 new tokens read, 12,775 from the prompt cache) | **5.46 s** | 11.35 s: **−52%** | H-final vs T-paper |
| Reading a prompt, from cold | **523 / 430 / 391 tokens/s** at 4K / 32K / 100K | | R-lengths |
| Writing after a 100K-token prompt | **~21 tokens/s** | | R-lengths |
| Exam | **18/20** | 18/20 with the draft head off and none of the later changes | final, sp-final-union, paper-2r |

The first setup (T-paper) was one drive, a 32 GiB expert cache and no draft head or reading room;
everything listed below changed between the two runs, so +47% is the whole climb, not one step.

## Feature by feature

Each line is one run: the change against the setup just before it, on that run's own conversations.
Runs happened on different days and settings, so compare within a line, not down the table.

| Feature | Gain | Run |
|---|---|---|
| Two drives (first setup) | +14.8% writing, +40.4% reading a prompt | E-paper |
| Two drives (final setup) | +15.4% writing (14.46 → 16.59), −31% time to first token | R-drives |
| Next-layer prefetch, 10 experts wide instead of 1 | +7.3% writing | R-lookahead |
| The MTP draft head on agent conversations | +22.8% writing (13.31 → 16.35); prose +1.1% | A-auto |
| The draft head's token list | +2.6% writing; October re-test +2.7% chat, +1.5% coding, then dropped (too small for an extra file) | G-guess, V46 |
| The reading room | +31.6% reading a prompt, −27.7% time to first token | RR-room |
| Lending the reading room to the cache while writing | +5.5% writing | RR-lend |
| Checkpoints every 8,192 tokens instead of 512 | +19.5% reading a prompt, −16.6% time to first token | C-cms |
| The draft head only records while reading a prompt | +14.8% reading a prompt, +3.1% writing | A-record |
| Union attention | +5.2% reading a prompt; working memory at 160K 5.63 → 1.94 GiB | U-union |
| The n-gram shelf | +3.0% writing, +3.2% reading a prompt | PS-shelf |
| Cheaper per-layer stops | +7.3% writing (15.52 → 16.66) | C-stops |
| Pinned first checkpoint | a new chat after a long one starts in 4.2 s instead of 14.2 s | pin gate |

## October: coding and chat

From October the study measured on two new sets from the author's own use, each replayed turn by
turn with the prompt cache on: 24 coding-agent conversations at high thinking effort (`code`), and
24 chat conversations at medium effort (`chat`). Writing is word-weighted over every reply. Each run
is A-B-A against the setup before it, the base pair had to agree within 3% (4% on the 12-conversation
lists the study uses for one change from 13 October), and a change had to gain at least 3% (or, for
an efficiency fix, stay level within that noise).

| Change | Coding | Chat | Run |
|---|---|---|---|
| The trained two-layer guesser with the keep rule | **+4.2%** (17.51 → 18.25 tokens/s) | **+8.2%** (17.62 → 19.06) | K-final |
| Fewer undo copies (`LLAMA_RS_SKIP_SLOTS=1`) | **+4.3%** (18.07 → 18.84) | level, −2.0% (20.86 → 20.44) | C-tax, C-taxfix-chat |
| The drive choice by load and read-thread priority (plumbing) | **+12.5%** (18.92 → 21.28) | **+10.4%** (19.14 → 21.13) | C-plumb |
| The trained one-layer guess, on top of all the above (12 conversations each) | **+7.5%** (20.19 → 21.71) | **+7.3%** (21.16 → 22.70) | T43-code-lean, T43-chat-lean |
| A full cache lets go the expert unused longest (LRU), not the lowest recent use count, on top of all the above (12 conversations each) | **+9.6%** (21.52 → 23.57) | **+9.7%** (22.75 → 24.96) | L44-code-lean, L44-chat-lean |
| The same at a 24 GiB cache, against the old rule at 28 GiB: ~4.1 GiB of memory back | level, +0.4% (21.52 → 21.59) | level, +2.8% (22.75 → 23.38) | L44-code-lean, L44-chat-lean |
| Chat's opening kept in RAM | | a new chat after coding work: first token in **4.1 s**, from 14.2 s | chat-return |
| The pin knows chat from coding work | | with the agent first, a new chat read in 463 tokens instead of 5-7K: **3.5 s** to the first token, from 14-20 s | pin-lanes live check |

The best measured arms: **23.6 tokens/s on code** and **25.0 on chat** with the LRU rule, on 12
conversations each (L44); the L44 bases are the mean of the old rule's two runs, before and after.
LRU cut the experts fetched from the SSD per written token by 28% (code) and 17% (chat), and gained
most on the longest conversations (+14-15%). Before it, the one-layer guess's arms wrote 21.7 and
22.7 (T43). Over all 24, before the one-layer guess: **21.2** and **21.1**, against 17.5 and 17.6 for
the same sets before the trained guessers, run in another week: about +21% and +20%. None of these
changes alters an output token (the guessers change only what is fetched early, LRU only what the
cache holds; the one-layer guess, the undo copies, the plumbing and LRU passed byte-identical word
checks), so the exam's 18/20 stands. With plumbing on, the GPU waits on the SSDs 14-15% of writing time,
from 20%. The one-layer guess wastes less drive time: 65-71% of its early fetches are used,
against 30-32% for the borrowed guess, and the fetches per token that are never used fall from about
42 to 7-8.

Tried and not kept in October: the check's matrix work on the M5's matrix units
(`GGML_METAL_STAFF_MV`, −0.8% and +1.1% on code, below the +3% bar), and raising the draft head's
ceiling from 5 to 8 guesses (+1.1%, below the bar). Both runs: C-tax. Later in October: reading one
missing expert from both drives at once (`LLAMA_MOE_STREAM_SPLIT`; the GPU's wait for experts did not
move, T11b-probe), and a lookup draft that copies the words that followed the same words earlier in
the conversation (best setting −0.06% on code in a replay of recorded rounds, under the +3% bar;
lookup replay).

From this release the switches behind these changes are the engine's defaults (the trained
guessers' tuning as soon as their files are named), and `--flashnext` sets llama.cpp's options to
the everyday values, the chat window (90 s) and chat's opening copy included
([Settings](settings.md#defaults-and---flashnext)). The study's own server runs exactly that, plus
the downloads, so the headline numbers are with `--flashnext` and all the extras: the 28 GiB
cache, the draft head, both guessers and the second SSD. They were also measured with a token list
for the draft head, which the study dropped afterwards (about 2% of writing: V46 above).

## Expert cache size (draft head off)

| Expert cache | Writing | Run |
|---|---|---|
| 28 GiB | 11.00 tokens/s | D-sweep |
| 32 GiB | 11.95 tokens/s (+8.6%) | D-sweep |
| 36 GiB | stopped early: memory pressure too high | D-sweep |

[Memory sizing](memory-sizing.md) explains how this became the everyday 28 GiB with the draft head on.

## Tried and not kept

| Idea | Result | Run |
|---|---|---|
| The UD-Q6_K_XL quant | writing −37%, reading a prompt −30% against UD-Q4_K_XL | E-trade |
| A 36 GiB expert cache | memory pressure, and a spill opening the next server | D-sweep, R-lookahead |
| Prompt batch 8,192 | +26.8% reading a prompt, but ~4.5 GiB more memory; the reading room instead | U-ub |
| Drafting 30 tokens ahead | 4.01 tokens/s against 14.20 | X-guess30 |
| Fetching experts for draft tokens early | +1.8% overall: off | A-fetch |
| Prefetching two layers ahead, by the router | the run spilled: void, parked; the trained guesser took it up in October | A-depth2 |

## Compared with others

llama-bench, 4,096 tokens read and 128 written (pp4096 / tg128). llama-bench reads through macOS's
file cache, so these numbers are softer than the server runs above.

| Who | Machine | pp4096 | tg128 |
|---|---|---|---|
| This fork, 32 GiB cache, one drive, no draft head | Mac mini M5 Pro 64 GB | 276.7 | 19.87 |
| npanj's fork, 36 GiB cache, Q4_0 experts | MacBook Pro M5 Pro 64 GB, 17.3 GB/s SSD | 337-367 | 18.0 (27.6 with the draft head) |

npanj's figures are his published numbers, not measured here.

## Prior work

- Marian Mihailescu's [MoE expert streaming](https://github.com/mihailescu2m/llama.cpp) for
  llama.cpp, which all of this builds on.
- npanj's [Qwen3.8-Flash-Next port](https://github.com/npanj/llama.cpp) and its
  [guide](../qwen38-flash-next-v3.md), including the first warning that a too-large cache collapses
  speed through swap.

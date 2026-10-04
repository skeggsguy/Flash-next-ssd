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
  Swap stayed at 0 in 33 of the 35 runs, over 7,266 counted replies.
- **Quality:** a fixed 20-task exam with known answers (10 from the assistant's own work, 10 coding
  tasks with unit tests). Changes meant to be exact had to reproduce three fixed answers byte for
  byte; changes that move words were checked with perplexity and the exam.

## Headline

| | Result | Compared with | Run |
|---|---|---|---|
| Writing, 120 agent conversations in order | **17.46 tokens/s** | 11.88 for the study's first setup: **+47%** | H-final vs T-paper |
| Time to first token, median | **5.46 s** | 11.35 s: **−52%** | H-final vs T-paper |
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
| The draft head's token list | +2.6% writing | G-guess |
| The reading room | +31.6% reading a prompt, −27.7% time to first token | RR-room |
| Lending the reading room to the cache while writing | +5.5% writing | RR-lend |
| Checkpoints every 8,192 tokens instead of 512 | +19.5% reading a prompt, −16.6% time to first token | C-cms |
| The draft head only records while reading a prompt | +14.8% reading a prompt, +3.1% writing | A-record |
| Union attention | +5.2% reading a prompt; working memory at 160K 5.63 → 1.94 GiB | U-union |
| The n-gram shelf | +3.0% writing, +3.2% reading a prompt | PS-shelf |
| Cheaper per-layer stops | +7.3% writing (15.52 → 16.66) | C-stops |
| Pinned first checkpoint | a new chat after a long one starts in 4.2 s instead of 14.2 s | pin gate |

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
| Prefetching two layers ahead | the run spilled: void, parked | A-depth2 |

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

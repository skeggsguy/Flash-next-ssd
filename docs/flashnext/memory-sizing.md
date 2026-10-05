# Memory sizing

The expert cache (`--moe-stream-cache`) is the biggest speed setting, and the context (`-c`) is
the second biggest claim on memory. Both come out of the same RAM that macOS and your other apps
need. This page gives the study's measured settings for 64 GB, an untested starting point for
48 GB, and how to check your own.

The rule the study ran under: **any swap is a failure, not a slow result.** When the model's memory
is swapped out, speed collapses (npanj saw writing fall from ~24 to ~3.7 tokens/s at a 38 GiB cache),
so size for zero swap with room to spare.

## Measure wired memory, not the memory column

Metal locks ("wires") every GPU buffer in RAM in full as soon as it is made resident or first used,
whether its pages were ever touched or not. The usual per-process figures (Activity Monitor's
Memory column, `footprint`'s total) count only pages the CPU touched, so they **undercount** this
server by many GiB: a half-empty context buffer costs its full size all the same.

So judge a setting by what is wired:

```sh
footprint --wired -p llama-server          # the server's wired memory
vm_stat | grep -E 'wired|compressor|Swap'  # the whole Mac: pages are 16 KiB
sysctl kern.memorystatus_level             # 100 = plenty free; under ~20 is tight
sysctl vm.swapusage                        # must stay at 0
```

The wired-memory limit (`sudo sysctl iogpu.wired_limit_mb=...`) caps the total the GPU may wire.
Past it, loading fails with an error the server survives, instead of the Mac locking up. It does
not reserve anything.

## Where the memory goes

Measured part by part at 160K context with the everyday settings (union attention, batch 4096, the
draft head on, 4 checkpoints):

| Part | GiB |
|---|---|
| Resident weights (attention, shared weights, router) | 5.13 |
| KV cache (12 layers) + the sparse attention's block index | 3.75 + 0.47 |
| Recurrent state of the other 36 layers (1 + 5 for the draft head) | 0.66 |
| Working memory while reading a prompt, batch 4096 | 1.94 |
| The draft head | 1.04 |
| Prompt-cache checkpoints (4 at 160K) | ~1.7 |
| The n-gram shelf | 0.125 |
| Small buffers and the server's heap | ~1 |
| **Everything but the expert cache** | **~15.8** |

To estimate it for another context length:

- about **9.9 GiB fixed** (weights, recurrent state, working memory, draft head, shelf, small buffers);
- plus about **27 KB per token** of context (KV cache and block index);
- plus each checkpoint: about **113 MiB + 2 MiB per 1K tokens** of context.

| Context (`-c`) | Checkpoints (`-ctxcp`) | Everything but the cache (estimate) |
|---|---|---|
| 65,536 | 3 | ~12.3 GiB |
| 131,072 | 3 | ~14.4 GiB |
| 200,000 | 3 | ~16.5 GiB |

The reading room (about 1.4 GiB) is carved out of the expert cache, not added to it. Without the
draft head, take off about 1 GiB plus most of the recurrent-state copies. `-np` above 1 adds a
whole working-memory set per slot. `--cache-ram` (default 8 GiB) comes on top unless set to 0.

These estimates come out a little above what was measured wired at the everyday setting (below),
so treat them as a safe upper bound.

## 64 GB Macs: measured

Wired limit 59,392 MiB (58 GiB) in every run.

| Expert cache | With | Result |
|---|---|---|
| **28 GiB** | draft head on, 200K context, 3 checkpoints | **The everyday setting.** About 41-42 GiB wired in all. The Mac then runs at memory level ~33 for hours; the study's final 120-conversation run at this setting had one small spill, caused from outside (below). |
| 30 GiB | draft head off, long prompts | Safe in the tests. |
| 32 GiB | draft head off, 96K context | The ceiling without the draft head: +8.6% writing over 28 GiB. Long prompts at 32 took the memory level down to 16-22, so use 30 for long prompts. |
| 36 GiB | draft head off | **Too big.** The early brake stopped one run (memory level 12 for 10 s), and another swapped as it opened. |

A bigger cache with the draft head on was tried at 200K context with 3 checkpoints (2026-10-04):
30 GiB wired 43.9 GiB and 29 GiB wired 43.0 GiB, and both left the Mac only 0.5-0.7 GiB of free memory
during a long prompt, under the study's 2 GiB bar. Swap stayed at 0, but neither was kept. 28 stays.

What tipped the everyday setting over, once: during a long run, Xcode and Apple's on-device
inference service woke and the compressed-memory pool jumped by ~400 MB in two seconds, and 16 pages
went to swap. At 28 GiB the Mac has little to spare, so **close heavy apps** (Xcode, browsers with
many tabs, other models) while the server runs, or use 26 GiB if you need them open.

## 48 GB Macs: a starting point (untested)

Nothing here was measured on a 48 GB Mac. Working from the table above:

- Leave macOS and your apps about as much as on 64 GB: plan for the server to wire at most about
  30 GiB in all.
- With a 64K context and 3 checkpoints, everything but the cache is about 12.3 GiB (estimate), which
  leaves about **16-18 GiB** for the expert cache.
- **Start at `--moe-stream-cache 16 -c 65536 -ctxcp 3`**, with the draft head on and `--cache-ram 0`.
  macOS's default wired limit is 75% of RAM, so 36 GiB (36,864 MiB) on a 48 GB Mac, which already
  covers ~30 GiB; if loading fails with a GPU allocation error, raise it in small steps (for example
  `sudo sysctl iogpu.wired_limit_mb=40960`) rather than to the maximum.
- Then raise the cache in 2 GiB steps, re-checking swap and the memory level under real work,
  including your longest prompt. Stop at the last step that stays at zero swap with the memory
  level comfortably above 20.

Expect it to be slower than on 64 GB: a smaller cache misses more often, and every miss is a read
from the SSD. Two drives help more here, since they make each miss cheaper.

## Opening the model safely

Most of the study's swap events happened in the first seconds after the server loads, when it
claims the whole cache at once and macOS has to compress other memory just as fast.

- Set `LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB=4096` so the cache is claimed in 4 GiB steps with short pauses.
- Check `vm_stat`'s "Pages stored in compressor" before a big start. The compressed pool grows
  with long prompts and does not drain by itself. One 28 GiB start spilled from a steady 124K pages
  stored; keep it near 90K or less, and run `sudo purge` or reboot if it is larger.
- Don't start a big cache in the first ~10 minutes after a reboot, while macOS is still busy.
- Keep the model wired between requests (`GGML_METAL_RESIDENCY_KEEP_ALIVE_S=10000000`). Otherwise
  Metal un-wires it after 3 idle minutes and wires ~41 GiB back in one go on the next request: a
  small repeat of the opening rush.

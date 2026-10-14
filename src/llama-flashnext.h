#pragma once

// The fork's proven defaults (the Flash-Next study's standard roll-out, 2026-10-15): once a change has passed the
// study's bar it becomes the engine's default, unless it needs a download. Unset, each switch below runs the study's
// everyday value; "0" (or the old value) restores the old way. A switch that only tunes a downloaded file (the trained
// two-floor guesser's) takes the everyday value only when that file is named, and the old way without it, where the
// study never proved it (K-keep: the untrained two-floor guess with the keep rule wrote 1.4-1.6% slower than none).
//
// Pure parsers: the switch's value as given (nullptr = unset) in, its setting out. The engine reads the environment
// and calls these; the start-up line (common/flashnext.cpp) asks them what an unset switch means, so the line and the
// engine cannot disagree. The other layers' switches: GGML_METAL_STAFF_MV and GGML_METAL_RESIDENCY_KEEP_ALIVE_S
// (ggml-metal), LLAMA_SPEC_ADAPTIVE_RATE (common/flashnext.h).

#include "llama.h" // LLAMA_API: common's start-up line asks these what an unset switch means

#include <cstdint>

// LLAMA_MOE_STREAM_LRU: the desk puts back the book idle longest (#44: +9.7% chat, +9.6% code at desk 28) instead of
// the one used least lately (hotness). Unset: on. Set: on unless it reads as 0 (so "0", "" and a word are hotness).
LLAMA_API bool llama_flashnext_lru(const char * value);

// LLAMA_RS_SKIP_SLOTS: each round writes only the undo copies a rollback can reach (llama-rs-skip.h; the tax fix:
// code +4.3%, chat level, same words). Unset or "1": on; anything else ("0" included): off.
LLAMA_API bool llama_flashnext_rs_skip_slots(const char * value);

// LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB: the desk is made in chunks of this many MiB with a pause between them (the gentle
// opening), not in one go. Unset: 4096. "0" (or anything that reads as 0 or less): one go, the old way.
LLAMA_API int64_t llama_flashnext_alloc_chunk_mib(const char * value);

// LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER names a file (set and not ""): the guesser tuning below takes its everyday values
LLAMA_API bool llama_flashnext_guesser_named(const char * guesser_value);

// The trained two-floor guesser's tuning, as K-final ran it. `guesser`: whether LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER
// names a file. Unset: the everyday value with the guesser, the old way (off) without it; set: the value given.
//   LLAMA_MOE_STREAM_LOOKAHEAD_ALL     a small batch guesses for every word: unset 1 / 0; set: on unless "" or "0"
//   LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2  books guessed two floors ahead:   unset 4 / 0; set: the number (0 or more)
//   LLAMA_MOE_STREAM_LOOKAHEAD_KEEP    the keep rule:                    unset 1 / 0; set: on unless "" or "0"
//   LLAMA_MOE_STREAM_LOOKAHEAD_CONF    the confidence cut:          unset 0.2 / 0; set: "" or "0" is 0 (the fixed
//                                      pick), else a number in (0, 1] or it throws, so a mistyped cut never runs
LLAMA_API bool     llama_flashnext_lookahead_all(const char * value, bool guesser);
LLAMA_API uint32_t llama_flashnext_lookahead_depth2(const char * value, bool guesser);
LLAMA_API bool     llama_flashnext_lookahead_keep(const char * value, bool guesser);
LLAMA_API float    llama_flashnext_lookahead_conf(const char * value, bool guesser);

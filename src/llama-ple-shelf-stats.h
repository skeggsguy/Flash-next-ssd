#pragma once

// The phrasebook shelf's counters and its lines (SHARE-PARTS-PLAN.md phase 5). Its own prefix,
// "phrasebook:", and its own counters: none of its reads reach the book manager's stats or the drives
// line, and no line says "moe stream", "room", "drives" or "lent belt", the first words the study's
// runner matches (runner/roomstats.py). One line per stats window (LLAMA_MOE_STREAM_STATS_MS, the book
// manager's clock) with asks, and a total when the model is freed:
//
//   phrasebook: window asks A, on the shelf H (P%) | writing a, h (p%) | reading in a, h (p%) | read R rows
//     (file F, alt G), M MiB, W ms waited | held X of C rows | since start asks TA, on the shelf TH (TP%),
//     writing (tw%), reading in (tr%)
//   phrasebook: total asks A, ... W ms waited | held X of C rows | B batches (K started ahead), P rows passed
//     (read, not kept)
//
// An ask is one distinct row of one batch (a batch reads a row once however many tokens want it, as the
// simulator counts). Writing is a batch of at most LLAMA_PLE_SHELF_WRITING_MAX tokens (one written token,
// or the apprentice's check of up to 5 guesses), reading in anything longer: ~/dev/ai/sim/ple_sim.py's
// --check-max-tokens, so the live hit rates compare with the simulator's. file / alt: rows read from the
// model's own copy of the table and from the other drive's (two runners). waited: time the graph waited
// for the rows after filling its other inputs.

#include <cstdint>
#include <string>

#define LLAMA_PLE_SHELF_WRITING_MAX 6

struct llama_ple_shelf_stats {
    int64_t calls        = 0;
    int64_t begun        = 0;      // calls started ahead, at the top of set_inputs (the rest were gathered in place)
    int64_t asks[2]      = {0, 0}; // [0] writing, [1] reading in
    int64_t hits[2]      = {0, 0};
    int64_t passed       = 0;      // read but not kept: the batch held every slot
    int64_t reads[2]     = {0, 0}; // [0] the model's own file, [1] the alt copy
    int64_t bytes        = 0;
    int64_t t_wait_us    = 0;
};

// 0 writing, 1 reading in, for a batch of n_tokens
int llama_ple_shelf_kind_of(int64_t n_tokens);

// the window line (now - prev); empty when the window had no asks
std::string llama_ple_shelf_window_line(const llama_ple_shelf_stats & now, const llama_ple_shelf_stats & prev,
                                        int64_t held, int64_t capacity);

// the whole run's line
std::string llama_ple_shelf_total_line(const llama_ple_shelf_stats & s, int64_t held, int64_t capacity);

// --ple-shelf: the size asked for (-1 auto, 0 off, else MiB) as slots of row_size bytes, at most n_rows
int64_t llama_ple_shelf_slots(int32_t mib, size_t row_size, int64_t n_rows);

// the MiB auto means: twice the knee of the PS-trace sitting (LRU reached the reuse ceiling at 64 MiB)
#define LLAMA_PLE_SHELF_AUTO_MIB 128

// the startup line, in plain words
std::string llama_ple_shelf_startup_line(int32_t mib, int64_t slots, size_t row_size, int64_t n_rows,
                                         int n_copies, bool nocache, bool locked, const std::string & lock_error);

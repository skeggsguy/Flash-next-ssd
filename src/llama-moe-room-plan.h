#pragma once

// The reading room's plan for one floor, and the ids each of its GEMM groups reads: pure, no manager,
// no ggml, unit-tested in tests/test-moe-room-plan.cpp.
//
// A read-in the room takes (sweep_min_tokens or more, ~20 slips per book: llama-moe-room-size.h) asks
// for enough of every floor's books that fetching them all without waiting for a slip pays, so a floor's
// missing books are simply the books not on its desk, known before its router runs. The runners fetch
// them in the order the plan gives, floor after floor, without waiting for any slip: first into EMPTY
// desk slots (a cold desk, never evicting a book), the rest onto the belt in `parts` parts.
//
// The floor's maths then run one GEMM group per source - the desk, then each part - over the whole
// ubatch, and each group reads an id plane: the pair's slot (desk) or record (part) where that source
// holds the pair's book, -1 (skip) where another does. Every pair is owned by exactly one plane, so
// each output row is written once, by the kernel a run without streaming uses.

#include <cstdint>
#include <utility>
#include <vector>

struct llama_moe_room_floor_plan {
    std::vector<std::pair<int32_t, int32_t>> fills; // (book, EMPTY desk slot): the desk's cold start
    std::vector<std::vector<int32_t>>        parts; // each part's books, in belt record order; may be empty
};

// Plans one floor from its desk: slot_expert[s] is the book in desk slot s (-1 = EMPTY; a LOADING
// book counts as on the desk, the desk op waits for it), is_alt[e] says book e is read by the second
// runner. Both runners are kept busy on every part: the missing books are interleaved so that every
// run of them holds each runner's books in proportion, to within one book.
llama_moe_room_floor_plan llama_moe_room_plan_floor(uint32_t n_expert, const std::vector<int32_t> & slot_expert,
        const std::vector<uint8_t> & is_alt, int32_t n_parts);

// Writes one group's id plane: out[i] = where[ids[i]] (a slot, a record, or -1). Returns the pairs it
// owns (out[i] >= 0), so the floor can check its planes own every pair exactly once.
int64_t llama_moe_room_emit(const int32_t * ids, int64_t n, const std::vector<int32_t> & where, int32_t * out);

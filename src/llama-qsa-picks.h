#pragma once

// qwen4exp QSA: which blocks the block top-k picker may pick while reading in, and a counter of the
// picks it wastes (SHARE-PARTS-PLAN.md phase 1). Pure: no ggml, no cells, so the rule and the counter
// are unit-tested on their own (tests/test-qsa-picks.cpp) and the model test checks the graph
// (tests/test-qsa-causal.cpp).
//
// A reading-in batch scores every block of the context for every token at once, adds a bias per
// block, and top_k keeps the best n_pick blocks; the attention mask, which hides the cells after the
// token, is only applied after that. The old rule gave +1e9 to every block at or after the token's
// tail, so the blocks written later in the same batch (the token's future) outranked every real
// block, were picked, and were then hidden by the mask: wasted picks. In a 4,096-token batch the
// first ~2,000 tokens spent every one of their 513 picks there, and a token that completes its block
// (position % 4 == 3) had no tail of its own left to see: it attended to nothing.
//
// The causal rule (LLAMA_QSA_CAUSAL_PICKS, on unless set to 0, read when the memory is made):
//   a block starting after the token            -inf  (never a candidate)
//   the block holding the token's tail           1e9  (always picked, as in the reference)
//   earlier blocks                                 0  (ranked by their score)
//   the spare block (the unpooled cells)         1e9 if it holds a cell at or before the token, else -inf
// Every row keeps a finite block: the token's own cell is either in a full block (its tail block, or
// an earlier block when it completes it) or unpooled, in the spare. Streams holding more than one
// sequence keep the old rule.

#include <cmath>
#include <cstdint>
#include <limits>

// the spare block holds no cell: its smallest index
static const int64_t LLAMA_QSA_SPARE_EMPTY = std::numeric_limits<int64_t>::max();

// blocks the block top-k keeps per token: whole blocks covering indexer_top_k cells plus the
// incomplete tail (the reference's indexer_top_k + r - 1), never more than the context has
inline int64_t llama_qsa_n_block_picks(int64_t n_kv, int64_t r, int64_t indexer_top_k) {
    const int64_t width_cells = n_kv < indexer_top_k + r - 1 ? n_kv : indexer_top_k + r - 1;
    const int64_t n_blocks    = (n_kv + r - 1)/r;
    const int64_t n_sel       = (width_cells + r - 1)/r;
    return n_sel < n_blocks ? n_sel : n_blocks;
}

// the bias of a pooled block starting at index blk_start, for the token at index q whose incomplete
// tail starts at tail_start ((q + 1)/r*r). finite values, so a score can never meet a -inf and give a nan
inline float llama_qsa_block_bias(int64_t blk_start, int64_t q, int64_t tail_start, bool causal) {
    if (causal && blk_start > q) {
        return -INFINITY;
    }
    return blk_start >= tail_start ? 1e9f : 0.0f;
}

// the bias of the spare block, whose smallest cell index is spare_min (LLAMA_QSA_SPARE_EMPTY if none).
// the old rule keeps it at 1e9 for every row: a sequence with fewer than r cells owns no full block,
// and a row of -inf only gives a nan
inline float llama_qsa_spare_bias(int64_t spare_min, int64_t q, bool causal) {
    if (causal) {
        return spare_min <= q ? 1e9f : -INFINITY;
    }
    return 1e9f;
}

// LLAMA_QSA_UNION (models/qwen4exp-qsa-union.h): the same causal rule as two bounds per block, so the
// bias of every (token, block) pair can be built on the GPU from one small table per block instead of
// an [n_blocks x n_tokens] host input:
//   bias(q) = q >= lo ? (q < hi ? 1e9 : 0) : -inf
// A full block starting at s: lo = s, hi = s + r - 1 (the tail rule: s >= (q + 1)/r*r <=> q <= s + r - 2).
// The spare block with smallest index m: lo = m, hi = never (1e9 from m on). A padding block id, or an
// empty spare: lo = never (-inf for every row). Causal rule, one sequence a stream, only.
static const int64_t LLAMA_QSA_BOUND_NEVER = int64_t(1) << 40; // past any index

struct llama_qsa_bounds {
    int64_t lo;
    int64_t hi;
};

inline llama_qsa_bounds llama_qsa_block_bounds(int64_t blk_start, int64_t r) {
    return { blk_start, blk_start + r - 1 };
}

inline llama_qsa_bounds llama_qsa_spare_bounds(int64_t spare_min) {
    return { spare_min == LLAMA_QSA_SPARE_EMPTY ? LLAMA_QSA_BOUND_NEVER : spare_min, LLAMA_QSA_BOUND_NEVER };
}

inline llama_qsa_bounds llama_qsa_no_block_bounds() {
    return { LLAMA_QSA_BOUND_NEVER, LLAMA_QSA_BOUND_NEVER };
}

inline float llama_qsa_bounds_bias(const llama_qsa_bounds & b, int64_t q) {
    return q >= b.lo ? (q < b.hi ? 1e9f : 0.0f) : -INFINITY;
}

// the bounds as the GPU tables hold them, so the graph needs only step(): [q >= lo] = step(q + lo_f) with
// lo_f = -(lo - 0.5), and [q < hi] = step(hi_f - q) with hi_f = hi - 0.5. Every index is an integer below
// 2^22, so q, lo - 0.5 and hi - 0.5 are exact in F32 (LLAMA_QSA_BOUND_NEVER rounds, harmlessly: it is
// only compared with indices far below it)
inline float llama_qsa_lo_f32(const llama_qsa_bounds & b) { return -((float) b.lo - 0.5f); }
inline float llama_qsa_hi_f32(const llama_qsa_bounds & b) { return (float) b.hi - 0.5f; }

// the most cells the F32 tables can index exactly (see above)
static const int64_t LLAMA_QSA_UNION_MAX_KV = int64_t(1) << 22;

// a cell's index as the per-cell table holds it; an empty cell is after every token
static const float LLAMA_QSA_CELL_EMPTY_F32 = 3e38f;

// what the graph computes from those tables, op for op (step: 1 where x > 0), for the unit test
inline float llama_qsa_bias_from_f32(float lo_f, float hi_f, float q) {
    const float vis  = (lo_f + q) > 0.0f ? 1.0f : 0.0f;
    const float tail = (hi_f - q) > 0.0f ? 1.0f : 0.0f;
    return std::log(vis) + tail*1e9f;
}

// what one ubatch's rows did with their picks, counted from the bias alone (block top-k only)
struct llama_qsa_pick_count {
    int64_t n_rows         = 0;
    int64_t n_rows_future  = 0; // every pick on a finite future-or-tail block: all the blocks at 1e9 fill the picks
    int64_t n_rows_blind   = 0; // ... and none of them is the token's own tail: the token sees nothing
    int64_t n_picks        = 0; // n_rows * n_pick
    int64_t n_picks_future = 0; // at least this many picks went to blocks after the token

    void add(const llama_qsa_pick_count & o);
};

// one row's blocks, sorted into the counter as the bias loop assigns them
struct llama_qsa_row_tally {
    int64_t n_future_hi = 0; // finite (1e9) blocks starting after the token
    int64_t n_tail_hi   = 0; // 1e9 blocks at or before the token: its own tail

    // a block (or the spare) got `bias`; `future` if it starts after the token (the spare: holds no cell at or before it)
    void add(float bias, bool future) {
        if (bias == 1e9f) {
            (future ? n_future_hi : n_tail_hi)++;
        }
    }

    void finish(int64_t n_pick, llama_qsa_pick_count & count) const;
};

// the running totals a memory keeps
struct llama_qsa_pick_stats {
    uint64_t             n_ubatch = 0; // calls with block top-k rows, one per ratio and ubatch
    llama_qsa_pick_count total;
    llama_qsa_pick_count last;         // the latest call's
};

// "qsa picks: ..." for one call, printed with LLAMA_QSA_PICK_STATS=1
void llama_qsa_pick_log(const llama_qsa_pick_count & c, int64_t n_blocks, int64_t n_pick, bool causal);

// the totals, printed when the memory goes
void llama_qsa_pick_log_total(const llama_qsa_pick_stats & s, bool causal);

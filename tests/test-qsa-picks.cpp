// The block top-k picker's bias rule and its counter (src/llama-qsa-picks.h), worked by hand, and the
// same rule as two bounds a block (LLAMA_QSA_UNION), brute-forced against it.
// The graph side is tests/test-qsa-causal.cpp.

#include "../src/llama-qsa-picks.h"

#include <cmath>
#include <cstdio>
#include <initializer_list>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

// the block starts of a 16-cell context with ratio 4: 0, 4, 8, 12
static void test_block_bias() {
    const int64_t r = 4;
    for (bool causal : { true, false }) {
        for (int64_t q = 0; q < 16; ++q) {
            const int64_t tail_start = (q + 1)/r*r;
            for (int64_t start = 0; start < 16; start += r) {
                const float v = llama_qsa_block_bias(start, q, tail_start, causal);
                if (start > q) {
                    // after the token: never a candidate with the rule on, the old tail value without it
                    CHECK(causal ? (std::isinf(v) && v < 0) : v == 1e9f);
                } else if (start == tail_start) {
                    CHECK(v == 1e9f); // the token's own tail: always picked
                } else {
                    CHECK(v == 0.0f); // an earlier block, or the one the token completes: ranked by score
                }
            }
        }
    }
    // the token that completes its block (q % 4 == 3) has no tail: its block competes, the next one is future
    CHECK(llama_qsa_block_bias(4, 7, 8, true) == 0.0f);
    CHECK(std::isinf(llama_qsa_block_bias(8, 7, 8, true)));
    CHECK(llama_qsa_block_bias(8, 7, 8, false) == 1e9f);
}

static void test_spare_bias() {
    // a spare holding cells from index 8 on: seen from 8, hidden before
    CHECK(std::isinf(llama_qsa_spare_bias(8, 7, true)));
    CHECK(llama_qsa_spare_bias(8, 8, true) == 1e9f);
    CHECK(llama_qsa_spare_bias(8, 9, true) == 1e9f);
    // an empty spare is never a candidate with the rule on
    CHECK(std::isinf(llama_qsa_spare_bias(LLAMA_QSA_SPARE_EMPTY, 1000, true)));
    // the old rule: 1e9 for every row
    CHECK(llama_qsa_spare_bias(8, 7, false) == 1e9f);
    CHECK(llama_qsa_spare_bias(LLAMA_QSA_SPARE_EMPTY, 7, false) == 1e9f);
}

static void test_n_block_picks() {
    // Flash-Next: indexer_top_k 2048, ratio 4 -> 2051 cells -> 513 blocks, never more than the context has
    CHECK(llama_qsa_n_block_picks(4096, 4, 2048) == 513);
    CHECK(llama_qsa_n_block_picks(163840, 4, 2048) == 513);
    CHECK(llama_qsa_n_block_picks(1024, 4, 2048) == 256);
    CHECK(llama_qsa_n_block_picks(2052, 4, 2048) == 513);
    CHECK(llama_qsa_n_block_picks(2048, 4, 2048) == 512);
    // the tiny fixture: indexer_top_k 8 -> 11 cells -> 3 blocks
    CHECK(llama_qsa_n_block_picks(256, 4, 8) == 3);
    CHECK(llama_qsa_n_block_picks(8, 4, 8) == 2);
}

// rows of a 10-cell batch at ratio 4 (blocks 0-3, 4-7 full, the spare holding 8 and 9), 3 picks,
// tallied as set_input_qsa tallies them: the numbers the model test's first batch prints
static llama_qsa_pick_count tally_batch(bool causal) {
    llama_qsa_pick_count c;
    const int64_t r = 4, n_cells = 10, spare_min = 8;
    for (int64_t q = 0; q < n_cells; ++q) {
        const int64_t tail_start = (q + 1)/r*r;
        llama_qsa_row_tally t;
        for (int64_t start = 0; start + r <= n_cells; start += r) {
            t.add(llama_qsa_block_bias(start, q, tail_start, causal), start > q);
        }
        t.add(llama_qsa_spare_bias(spare_min, q, causal), spare_min > q);
        t.finish(3, c);
    }
    return c;
}

static void test_tally() {
    // old rule: q 0-2 see block 0 (tail) + block 1 + the spare at 1e9, 2 of 3 picks past the token each;
    // q 3 completes block 0 and has block 1 and the spare at 1e9 (2 past); q 4-6 tail 1 + spare (1 past);
    // q 7 the spare (1 past); q 8-9 the spare is their tail (0 past)
    const auto o = tally_batch(false);
    CHECK(o.n_rows == 10);
    CHECK(o.n_picks == 30);
    CHECK(o.n_rows_future == 3);
    CHECK(o.n_rows_blind == 0);
    CHECK(o.n_picks_future == 2*3 + 2 + 1*3 + 1);

    const auto c = tally_batch(true);
    CHECK(c.n_rows == 10);
    CHECK(c.n_rows_future == 0);
    CHECK(c.n_picks_future == 0);

    // a token completing its block with every pick on future blocks sees nothing
    llama_qsa_pick_count b;
    llama_qsa_row_tally t;
    t.add(0.0f, false);
    for (int k = 0; k < 3; ++k) {
        t.add(1e9f, true);
    }
    t.finish(3, b);
    CHECK(b.n_rows_future == 1 && b.n_rows_blind == 1 && b.n_picks_future == 3);

    // add() sums
    llama_qsa_pick_count sum;
    sum.add(o);
    sum.add(b);
    CHECK(sum.n_rows == 11 && sum.n_picks_future == o.n_picks_future + 3 && sum.n_rows_blind == 1);
}

// LLAMA_QSA_UNION: the bounds give every row the causal rule's bias, brute force over every (q, block, r)
// and every spare, both as integers and through the F32 tables and the graph's step() ops
static void test_bounds() {
    int64_t n_cases = 0;
    const auto both = [&](const llama_qsa_bounds & b, int64_t q, float want) {
        const float got   = llama_qsa_bounds_bias(b, q);
        const float got_f = llama_qsa_bias_from_f32(llama_qsa_lo_f32(b), llama_qsa_hi_f32(b), (float) q);
        const bool  ok    = std::isinf(want) ? (std::isinf(got) && got < 0 && std::isinf(got_f) && got_f < 0)
                                             : (got == want && got_f == want);
        if (!ok) {
            fprintf(stderr, "  lo %lld hi %lld q %lld: want %g, got %g / %g\n",
                    (long long) b.lo, (long long) b.hi, (long long) q, want, got, got_f);
        }
        CHECK(ok);
        n_cases++;
    };
    for (int64_t r = 1; r <= 8; ++r) {
        for (int64_t q = 0; q < 12*r; ++q) {
            const int64_t tail_start = (q + 1)/r*r;
            for (int64_t start = 0; start < 12*r; start += r) {
                both(llama_qsa_block_bounds(start, r), q, llama_qsa_block_bias(start, q, tail_start, true));
            }
        }
    }
    for (int64_t q = 0; q < 64; ++q) {
        for (int64_t m = 0; m < 64; ++m) {
            both(llama_qsa_spare_bounds(m), q, llama_qsa_spare_bias(m, q, true));
        }
        both(llama_qsa_spare_bounds(LLAMA_QSA_SPARE_EMPTY), q, llama_qsa_spare_bias(LLAMA_QSA_SPARE_EMPTY, q, true));
        both(llama_qsa_no_block_bounds(), q, -INFINITY);
    }
    // near the top of the index range the F32 tables must still be exact (ratio 4, 2^22 cells)
    for (int64_t q : { (int64_t) 4194300, (int64_t) 4194301, (int64_t) 4194302, (int64_t) 4194303 }) {
        for (int64_t start : { (int64_t) 4194296, (int64_t) 4194300 }) {
            both(llama_qsa_block_bounds(start, 4), q, llama_qsa_block_bias(start, q, (q + 1)/4*4, true));
        }
        both(llama_qsa_spare_bounds(4194302), q, llama_qsa_spare_bias(4194302, q, true));
    }
    // an empty cell is after every token
    CHECK(LLAMA_QSA_CELL_EMPTY_F32 - 4194303.0f > 0.0f);
    CHECK(n_cases > 3000);
}

int main() {
    test_block_bias();
    test_spare_bias();
    test_n_block_picks();
    test_tally();
    test_bounds();

    fprintf(stderr, "test-qsa-picks: %s\n", n_fail == 0 ? "all tests OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}

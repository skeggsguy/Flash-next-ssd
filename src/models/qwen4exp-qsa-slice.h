#pragma once

// qwen4exp QSA, sliced: the block top-k picker and the attention of one layer, a slice of the batch's
// rows at a time (SHARE-PARTS-PLAN.md phase 3).
//
// Reading in a 4,096-token batch builds, per layer, a score for every (token, block) pair, and a full
// [context x batch] attention mask twice over (the -inf fill the picks are written into, and that fill
// plus the causal mask). At 160K of context that is most of the library's working space, and all of it
// is scratch that one layer's attention reads once. Slicing builds the same ops for rows [t0, t0 + n)
// only, runs attention on those rows, and writes the result into place before the next slice starts,
// so the scratch is sized by the slice, not the batch. Every row's work is exactly what it was: the
// ops are per row (a row's score, picks, mask and attention never read another row), and each slice
// starts on a multiple of 32 rows so the GPU's attention tiles fall where they did.
//
// LLAMA_QSA_SLICE (read when the memory is made, so per context): rows a slice. Unset means
// LLAMA_QSA_SLICE_DEFAULT; "0" is today's graph exactly (the whole batch at once); any other positive
// number is rounded down to a multiple of 32, and at least 128, so that every slice has 64 rows or
// more: attention adds in another order for fewer rows (the GPU's single-query kernel below 20 rows,
// the CPU's row-at-a-time path below its 64-row tile), and a slice would then not be exact. Slicing
// applies only where the block picker runs on one stream and the batch is longer than a slice;
// everything else (writing, the decode gather, several streams, the per-cell picker) is today's graph.

#include <cstdint>
#include <vector>

struct ggml_tensor;

// 0 until the Q-slice rung has measured reading in and writing with it (then 512)
#define LLAMA_QSA_SLICE_DEFAULT 0u

// the smallest LLAMA_QSA_SLICE, and the fewest rows a slice of the plan has: see above
#define LLAMA_QSA_SLICE_MIN      128u
#define LLAMA_QSA_SLICE_MIN_ROWS 64

// LLAMA_QSA_SLICE's value (nullptr: unset) as rows a slice, 0 for off
uint32_t llama_qsa_slice_parse(const char * value);

struct llama_qsa_slice_span {
    int64_t t0; // first row
    int64_t n;  // rows
};

// the slices of an n_tokens batch at `rows` a slice: none when rows is 0 or the batch fits in one.
// An even split: k = ceil(n_tokens/rows) slices, the batch cut in 32-row chunks shared out as evenly as
// they go, the larger slices last, so that no slice is longer than `rows` (the reserve graph was sized
// for exactly that) and none is short (LLAMA_QSA_SLICE_MIN_ROWS or more; the last one keeps the odd rows)
std::vector<llama_qsa_slice_span> llama_qsa_slice_plan(int64_t n_tokens, uint32_t rows);

// what build_qsa_top_k hands the sliced attention instead of its picks: the shared tensors, built once
// for the whole batch, that every slice reads its rows of
struct llama_qsa_slice_parts {
    ggml_tensor * pooled    = nullptr; // F32 [idx_dim, n_blocks, 1]     the block summaries
    ggml_tensor * q         = nullptr; // F32 [idx_dim, n_idx_h, n_tokens] the indexer's queries
    ggml_tensor * bias      = nullptr; // F32 [n_blocks, n_tokens, 1]    the block bias (llama-qsa-picks.h)
    ggml_tensor * blk_cells = nullptr; // I32 [r*n_blocks, 1]            each block's cells

    int64_t r         = 0;     // cells a block
    int64_t n_blocks  = 0;
    int64_t n_blk_sel = 0;     // blocks picked per row
    bool    sparse_fa = false; // pass the selection width to flash attention (qwen4exp_sparse_fa)
    uint32_t rows     = 0;     // LLAMA_QSA_SLICE
};

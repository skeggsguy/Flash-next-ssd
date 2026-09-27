#pragma once

// qwen4exp QSA: the picks go straight to attention (SHARE-PARTS-PLAN.md phase 6; the design is
// docs/wizard-step6-union.md in the study).
//
// Reading in builds two full-width inputs per batch on the CPU, each copied to the GPU: the F16 KQ mask
// [n_kv x n_tokens] and the picker's F32 block bias [n_blocks x n_tokens], ~6 B per (batch token x
// context token), ~3.8 GiB at 160K with a 4,096-token batch. Slicing (qwen4exp-qsa-slice.h) cannot shrink
// them: they are inputs for the whole batch.
//
//   "bias" (6a, exact): the block bias is built on the GPU, a slice at a time, from three small tables
//          (two bounds per block and each row's index, llama-qsa-picks.h) with step / log / scale ops.
//          The same values reach the same top_k, so every pick and every word is the masked path's.
//   "1"    (6b): the bias as above, and attention reads the picks directly through the fork's union
//          attention (ggml_flash_attn_union): each row's picked cells, with the ones after the token
//          replaced by -1 (which union_build drops), are the visible set, so no KQ mask is built. The
//          same picks and the same visible cells; the sum runs in another order, so words may differ at
//          the rounding level.
//   "0"    step 3's sliced masked path with the host bias, exactly.
//
// LLAMA_QSA_UNION is read when the memory is made (per context). The route is decided from shapes,
// cparams and switches only, never the batch's contents, so the reserve graph takes it too (and the
// working space is sized by it): the block picker with its cell table, the causal rule, one stream, one
// sequence a stream, no gather, and a batch of LLAMA_QSA_UNION_MIN_ROWS rows or more. Union attention
// also needs flash attention, an F16 cache and K and V heads of one size; without them "1" builds "bias".

#include <cstdint>
#include <string>

struct ggml_context;
struct ggml_tensor;

enum llama_qsa_union_mode : uint8_t {
    LLAMA_QSA_UNION_OFF  = 0, // "0": the host bias and the masked path
    LLAMA_QSA_UNION_BIAS = 1, // "bias": the bias built on the GPU, masked attention
    LLAMA_QSA_UNION_ATTN = 2, // "1": the bias built on the GPU, union attention, no mask
};

// off until the U-union rung; the default after it is Tom's call
#define LLAMA_QSA_UNION_DEFAULT LLAMA_QSA_UNION_OFF

// the fewest rows a batch takes the route with: every slice has at least this many (qwen4exp-qsa-slice.h),
// and union attention needs 8 or more; shorter batches (a written token, the apprentice's check batches)
// keep the small host inputs
#define LLAMA_QSA_UNION_MIN_ROWS 64

// LLAMA_QSA_UNION's value (nullptr: unset): "0" off, "bias", "1" union; anything else off, with *bad set
llama_qsa_union_mode llama_qsa_union_parse(const char * value, bool * bad = nullptr);

// what the route depends on, all known when the graph is built
struct llama_qsa_union_gate {
    llama_qsa_union_mode mode = LLAMA_QSA_UNION_OFF;

    bool    block_topk   = false; // the block picker (its bias per block), with the cell table
    bool    causal       = false; // LLAMA_QSA_CAUSAL_PICKS
    bool    one_seq      = false; // one sequence a stream: !kv_unified || n_seq_max == 1
    bool    gather       = false; // the decode gather path
    int64_t n_stream     = 0;
    int64_t n_tokens     = 0;
    int64_t n_kv         = 0;

    bool    flash_attn   = false; // union attention only
    bool    kv_f16       = false;
    bool    same_head    = false; // K and V heads of one size
};

// the route a layer takes: OFF (today's inputs and graph), BIAS or ATTN
llama_qsa_union_mode llama_qsa_union_route(const llama_qsa_union_gate & g);

// why "1" builds "bias" in this whole context (nullptr: it does not): the union-attention condition the
// context lacks, which no batch can supply
const char * llama_qsa_union_fallback(const llama_qsa_union_gate & g);

// the startup line, in plain words: the route the context takes, which is the mode unless `fallback`
// (llama_qsa_union_fallback) says why "1" builds "bias". Printed by the first QSA layer built, where the
// route is known (llama_memory_hybrid_idx::qsa_union_log)
std::string llama_qsa_union_describe(llama_qsa_union_mode mode, const char * fallback = nullptr);

// the small host tables that replace the block bias input (filled by set_input_qsa)
struct llama_qsa_union_tables {
    ggml_tensor * blk_lo   = nullptr; // F32 [n_blocks]  llama_qsa_lo_f32 of each block's bounds
    ggml_tensor * blk_hi   = nullptr; // F32 [n_blocks]  llama_qsa_hi_f32
    ggml_tensor * cell_pos = nullptr; // F32 [n_kv]      a live cell's index, LLAMA_QSA_CELL_EMPTY_F32 if empty (union attention only)
    ggml_tensor * q_idx    = nullptr; // F32 [n_tokens]  each row's index (its position, or its mrope rank)

    bool on() const { return blk_lo != nullptr; }
};

// rows [t0, t0 + n) of the block bias, F32 [n_blocks, n]: {-inf, 1e9, 0} exactly as the host fills them
ggml_tensor * llama_qsa_union_bias(ggml_context * ctx, const llama_qsa_union_tables & tab, int64_t t0, int64_t n);

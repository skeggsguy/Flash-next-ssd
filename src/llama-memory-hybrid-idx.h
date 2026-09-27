#pragma once

#include "llama-memory-hybrid.h"
#include "llama-ple-trace.h"
#include "llama-qsa-picks.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

//
// qwen4exp QSA: keeping the indexer's block summaries (patch 4i, LLAMA_QSA_KEEP=1)
//

// A full block's summary (its r raw keys pooled, normed and roped at the block's first position)
// changes only when one of its keys is written, so the summaries are kept in a store, one row per
// position bucket pb = pos/r: block ids are compacted and renumber, and cells move on a restore, but
// pb does not. The rows past the buckets are spares for the rows a graph pools but never keeps (the
// unpooled tail's "dead" block and the fillers of the unused block ids): each such row is written to
// a spare of its own, since ggml_set_rows leaves two rows written to one destination undefined.
// Each ubatch picks how the summaries are made before its graph is built:
//   LEGACY - today's graph; the store is not kept up to date (switch off, more than one sequence in a
//            stream, a repeated position, a position past the cell window)
//   FULL   - today's graph, plus every row written into the store: the rebuild after anything that
//            may have left a kept row stale
//   INCR   - pool only the blocks this ubatch filled (plus the dead block and a filler), write them
//            into the store and read every block back from it
// A plan counts only once its compute has finished: next() commits it, and any memory operation that
// finds a plan still in flight (a failed or aborted compute) marks its streams for a rebuild.
enum llama_qsa_keep_mode {
    LLAMA_QSA_KEEP_LEGACY,
    LLAMA_QSA_KEEP_FULL,
    LLAMA_QSA_KEEP_INCR,
};

// one per compress ratio and ubatch; every QSA layer with that ratio follows it
struct llama_qsa_keep_plan {
    llama_qsa_keep_mode mode = LLAMA_QSA_KEEP_LEGACY;

    uint32_t ratio   = 0;
    uint32_t n_slots = 0; // INCR: the filled blocks per stream the graph pools; it pools n_slots + 2 rows
    uint32_t s0      = 0; // the ubatch's streams are [s0, s0 + ns)
    uint32_t ns      = 0;
};

// the store's side of the QSA inputs; a null tensor is not in this graph
struct llama_qsa_keep_inputs {
    const llama_qsa_keep_plan * plan = nullptr;

    ggml_tensor * fresh_cells = nullptr; // I32 [ratio*F, ns]  INCR: member cells of each pooled row
    ggml_tensor * fresh_pos   = nullptr; // I32 [4*F*ns]       INCR: mrope position rows of each pooled row
    ggml_tensor * fresh_dst   = nullptr; // I64 [F, ns]        INCR: store row each pooled row goes to
    ggml_tensor * blk_src     = nullptr; // I32 [n_blocks, ns] INCR: store row each block is read from
    ggml_tensor * blk_dst     = nullptr; // I64 [n_blocks, ns] FULL: store row each block goes to
};

struct llama_qsa_keep_stats {
    uint64_t n_legacy = 0; // committed plans, one per ratio and ubatch
    uint64_t n_full   = 0;
    uint64_t n_incr   = 0;

    // LLAMA_QSA_KEEP_CHECK: sum over INCR ubatches of |kept - rebuilt| per layer; must stay 0
    std::vector<std::pair<int32_t, double>> check;
};

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* the largest ubatch: bounds the rows one INCR ubatch pools (patch 4i) */
                 uint32_t   n_ubatch,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to; optional for block top-k
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns], by the rule in
    // llama-qsa-picks.h (blocks after the token -inf, so block top-k never picks them)
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    // blk_cells and blk_pos may be null when INCR pools without them; keep carries the store's inputs
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio, int64_t n_kv,
                       bool blk_bias, const llama_qsa_keep_inputs & keep) const;

    // patch 4i. the switches are read when the memory is made, so each context has its own
    bool qsa_keep()       const { return keep.enabled; }
    bool qsa_keep_check() const { return keep.check; }

    // F32 [indexer_head_size, qsa_keep_n_rows(ratio), n_stream]: a stream's rows are its position
    // buckets, then the spares that the rows pooled afresh every ubatch are written to
    ggml_tensor * qsa_keep_rows(int32_t il) const;
    ggml_tensor * qsa_keep_sum (int32_t il) const; // F32 [1], LLAMA_QSA_KEEP_CHECK only

    uint32_t qsa_keep_n_buckets(uint32_t ratio) const; // every bucket a cell's position can fall in
    uint32_t qsa_keep_n_spare  (uint32_t ratio) const; // the most rows one INCR ubatch pools
    uint32_t qsa_keep_n_rows   (uint32_t ratio) const; // buckets + spares

    // plans the ubatch just applied (one per ratio) and holds them in flight until qsa_keep_commit
    //   pos_max_prev: per stream of the ubatch, its sequence's last position before the ubatch was applied
    std::vector<llama_qsa_keep_plan> qsa_keep_plan(
            const llama_ubatch & ubatch, uint32_t s0, uint32_t ns, uint32_t n_kv,
            const std::vector<llama_pos> & pos_max_prev);

    // the worst case for graph reservation: FULL everywhere (LEGACY with the switch off)
    std::vector<llama_qsa_keep_plan> qsa_keep_plan_reserve(uint32_t n_stream) const;

    // the ubatch's compute finished: its plans take effect
    void qsa_keep_commit();

    llama_qsa_keep_stats qsa_keep_stats() const;

    // the block top-k picker's rule (llama-qsa-picks.h), read when the memory is made:
    // LLAMA_QSA_CAUSAL_PICKS (on unless 0), LLAMA_QSA_PICK_STATS=1 prints a line per batch
    bool                 qsa_causal_picks() const { return picks.causal; }
    llama_qsa_pick_stats qsa_pick_stats()   const { return picks.stats; }

    // LLAMA_PLE_TRACE (llama-ple-trace.h), read when the memory is made: null when off
    llama_ple_trace * get_ple_trace() const { return ple_trace.get(); }

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    //
    // patch 4i: the kept block summaries
    //

    enum qsa_dup : uint8_t {
        QSA_DUP_FREE,    // no position repeats in the stream
        QSA_DUP_PRESENT, // one does
        QSA_DUP_UNKNOWN, // look before the next plan
    };

    struct qsa_keep_layer {
        ggml_tensor * rows = nullptr;
        ggml_tensor * sum  = nullptr;
    };

    struct {
        bool enabled = false;
        bool check   = false;
        bool debug   = false;

        uint32_t n_ubatch = 0; // no ubatch is larger, so no INCR ubatch pools more rows than the spares

        std::vector<uint32_t> ratios; // distinct ratios of the QSA layers

        std::unordered_map<int32_t, qsa_keep_layer> layers;

        // per ratio and stream: every full block's row in the store is what pooling it now would give
        std::map<uint32_t, std::vector<uint8_t>> synced;

        // per stream, independent of the ratio
        std::vector<uint8_t> dup;

        // planned for a ubatch whose compute has not finished yet
        std::vector<llama_qsa_keep_plan> inflight;

        uint64_t n_legacy = 0;
        uint64_t n_full   = 0;
        uint64_t n_incr   = 0;

        std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;
    } keep;

    struct {
        bool causal = true;
        bool log    = false;

        // set_input_qsa is const: the counter is bookkeeping, not state
        mutable llama_qsa_pick_stats stats;
    } picks;

    // the phrasebook trace lives here because the memory is made once per context
    std::unique_ptr<llama_ple_trace> ple_trace;

    void qsa_keep_init(const llama_model & model, bool offload, uint32_t n_ubatch);

    // a plan in flight never finished its compute: whatever it wrote into the store cannot be trusted
    void qsa_keep_settle();

    // seq_id's stream (every stream if seq_id < 0) needs a FULL rebuild before INCR can resume
    void qsa_keep_unsync(llama_seq_id seq_id);

    // cells left seq_id's stream (every stream if seq_id < 0): a repeated position may have gone with them
    void qsa_keep_dup_unknown(llama_seq_id seq_id);
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // first stream of the current slot info, where get_k/get_v's views start
    uint32_t get_stream0() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio, int64_t n_kv,
                       bool blk_bias, const llama_qsa_keep_inputs & keep) const;

    // patch 4i: how the current ubatch makes the block summaries of this ratio (LEGACY with the switch off)
    const llama_qsa_keep_plan & get_qsa_plan(uint32_t ratio) const;

    bool          get_qsa_keep_check()       const;
    ggml_tensor * get_qsa_keep_rows(int32_t il) const;
    ggml_tensor * get_qsa_keep_sum (int32_t il) const;

    llama_ple_trace * get_ple_trace() const; // null when LLAMA_PLE_TRACE is off

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;
    const std::vector<uint32_t> s0_ubatch; // first stream of each ubatch

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // only a batch context plans: the update context has no ubatch, the full one reserves
    const bool is_batch = false;

    // the current ubatch's plans, one per ratio; in flight in the memory until next() commits them
    std::vector<llama_qsa_keep_plan> qsa_plans;
    bool qsa_planned = false;
};

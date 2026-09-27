#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <stdexcept>

// patch 4i switches, read when the memory is made
// unset means the default: LLAMA_QSA_KEEP is on unless set to 0 (Tom, 2026-09-25, after F1-summaries);
// the CHECK and DEBUG switches stay off unless set
static bool llama_qsa_keep_env(const char * name, bool def = false) {
    const char * value = std::getenv(name);
    return value == nullptr ? def : std::atoi(value) > 0;
}

static const char * llama_qsa_keep_mode_name(llama_qsa_keep_mode mode) {
    switch (mode) {
        case LLAMA_QSA_KEEP_LEGACY: return "LEGACY";
        case LLAMA_QSA_KEEP_FULL:   return "FULL";
        case LLAMA_QSA_KEEP_INCR:   return "INCR";
    }
    return "?";
}

static bool qwen4exp_indexer_f16() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_QWEN4EXP_INDEXER_F16");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
                 uint32_t   n_ubatch,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // Quantization can change the discrete top-k block selection. Keep the raw index keys in
        // f16 by default.
        const ggml_type type_idx_k = qwen4exp_indexer_f16() ? GGML_TYPE_F16 : type_k;

        // The indexer never reads V - only the keys are scored. Present the cache to
        // llama_kv_cache as MLA-shaped, which is the flag it already uses to skip the V side
        // entirely (llama-kv-cache.cpp: `const bool has_v = !is_mla`), instead of allocating a
        // V cache that nothing touches. Saves n_idx_layers * kv_size * indexer_head_size *
        // sizeof(type_v): 12 * 128 bytes per cell for this checkpoint, i.e. 288 MiB of f16 at
        // ctx 98304 and 384 MiB at 131072. ref: upstream PR #28330.
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells, K (%s), no V\n",
                __func__, kv_size, ggml_type_name(type_idx_k));

        return new llama_kv_cache(
            model, hparams_idx, type_idx_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    qsa_keep_init(model, offload, n_ubatch);

    // the picker's rule, read here so that each context has its own (SHARE-PARTS-PLAN.md phase 1)
    picks.causal = llama_qsa_keep_env("LLAMA_QSA_CAUSAL_PICKS", true);
    picks.log    = mem_idx != nullptr && llama_qsa_keep_env("LLAMA_QSA_PICK_STATS");

    if (mem_idx != nullptr) {
        LLAMA_LOG_WARN("%s: qsa picks: %s\n", __func__, picks.causal
                ? "causal, a token never picks a block after it (LLAMA_QSA_CAUSAL_PICKS=0 restores the old rule)"
                : "old rule (LLAMA_QSA_CAUSAL_PICKS=0), blocks after a token share its tail's bias and can fill its picks");
    }

    // LLAMA_PLE_TRACE: the phrasebook trace, per context like the switches above (SHARE-PARTS-PLAN.md phase 2)
    ple_trace = llama_ple_trace::from_env();
}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() {
    llama_qsa_pick_log_total(picks.stats, picks.causal);

    if (!keep.enabled) {
        return;
    }

    const auto stats = qsa_keep_stats();

    LLAMA_LOG_INFO("%s: qsa keep: LEGACY = %" PRIu64 ", FULL = %" PRIu64 ", INCR = %" PRIu64 "\n",
            __func__, stats.n_legacy, stats.n_full, stats.n_incr);

    for (const auto & [il, sum] : stats.check) {
        LLAMA_LOG_INFO("%s: qsa keep check: layer %3d sum |kept - rebuilt| = %g\n", __func__, il, sum);
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    // no full block is left, so every stream is trivially in step with the store
    keep.inflight.clear();

    for (auto & [ratio, synced] : keep.synced) {
        std::fill(synced.begin(), synced.end(), 1);
    }
    std::fill(keep.dup.begin(), keep.dup.end(), QSA_DUP_FREE);
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // a removal only turns full blocks partial, and a partial block becomes full again only through a
    // write, which pools it afresh in that ubatch: so no kept row goes stale here (the MTP trim and the
    // checkpoint cuts cost nothing). a removal can take a repeated position away, though
    qsa_keep_settle();
    qsa_keep_dup_unknown(seq_id);

    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // the destination stream takes cells it did not pool (a cross-stream copy replaces all of them)
    qsa_keep_unsync(seq_id_dst);

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    qsa_keep_unsync(-1);

    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // moved positions move blocks: a kept row is keyed by position and roped at it
    qsa_keep_unsync(seq_id);

    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    qsa_keep_unsync(seq_id);

    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    for (const auto & [ctx, buf] : keep.ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams_idx.no_alloc) {
            mb[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            mb[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    // restored keys were never pooled into the store. a PARTIAL_ONLY restore brings back the
    // recurrent state alone, so the indexer's cells and the kept rows are untouched
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        qsa_keep_unsync(seq_id);
    } else {
        qsa_keep_settle();
    }

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }

    // as in seq_rm: the cells that went may have held the stream's repeated position
    qsa_keep_dup_unknown(seq_id);
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        int64_t n_kv,
        bool blk_bias,
        const llama_qsa_keep_inputs & keep) const {
    const llama_qsa_keep_mode mode = keep.plan == nullptr ? LLAMA_QSA_KEEP_LEGACY : keep.plan->mode;

    const bool keep_full = mode == LLAMA_QSA_KEEP_FULL;
    const bool keep_incr = mode == LLAMA_QSA_KEEP_INCR;

    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(n_kv > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);
    GGML_ASSERT(bias != nullptr);
    GGML_ASSERT(cell_blk != nullptr || blk_bias);

    // today's pooling reads blk_cells and blk_pos; INCR pools its own rows and may leave both out
    GGML_ASSERT(keep_incr || (blk_cells != nullptr && blk_pos != nullptr));
    GGML_ASSERT(keep_full == (keep.blk_dst != nullptr));
    GGML_ASSERT(keep_incr == (keep.fresh_cells != nullptr) && keep_incr == (keep.fresh_pos != nullptr) &&
                keep_incr == (keep.fresh_dst   != nullptr) && keep_incr == (keep.blk_src   != nullptr));
    GGML_ASSERT(keep.plan == nullptr || keep.plan->ratio == ratio);

    for (const ggml_tensor * t : { cell_blk, blk_cells, blk_pos, bias,
            keep.fresh_cells, keep.fresh_pos, keep.fresh_dst, keep.blk_src, keep.blk_dst }) {
        GGML_ASSERT(t == nullptr || ggml_backend_buffer_is_host(t->buffer));
    }

    const int64_t n_ns     = bias->ne[2];             // streams in this ubatch
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;
    const int64_t n_blocks = (n_kv + r - 1)/r;

    GGML_ASSERT(n_ns > 0 && n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    GGML_ASSERT(blk_cells == nullptr || (blk_cells->ne[0] == r*n_blocks && blk_cells->ne[1] == n_ns));
    GGML_ASSERT(blk_pos   == nullptr || blk_pos->ne[0] == 4*n_blocks*n_ns);
    GGML_ASSERT(cell_blk  == nullptr || (cell_blk->ne[0] == n_kv && cell_blk->ne[1] == n_ns));
    GGML_ASSERT(bias->ne[0] == (blk_bias ? n_blocks : n_kv));
    GGML_ASSERT(bias->ne[1] == n_tps);

    // INCR still needs today's member lists and positions, to copy the rows it pools from them
    std::vector<int32_t> blk_cells_buf;
    std::vector<int32_t> blk_pos_buf;

    if (blk_cells == nullptr) {
        blk_cells_buf.resize(r*n_blocks*n_ns);
    }
    if (blk_pos == nullptr) {
        blk_pos_buf.resize(4*n_blocks*n_ns);
    }

    int32_t * dst_cell_blk  = cell_blk == nullptr ? nullptr : (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = blk_cells == nullptr ? blk_cells_buf.data() : (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = blk_pos   == nullptr ? blk_pos_buf.data()   : (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;

    // FULL and INCR: the store's rows (the position buckets, then the spares), and the rows INCR pools
    const int64_t n_buckets = keep_full || keep_incr ? qsa_keep_n_buckets(ratio) : 0;
    const int64_t n_rows    = keep_full || keep_incr ? qsa_keep_n_rows(ratio)    : 0;

    const int64_t n_slots = keep_incr ? keep.plan->n_slots : 0;
    const int64_t n_fresh = keep_incr ? n_slots + 2 : 0;

    if (keep_incr) {
        GGML_ASSERT(keep.fresh_cells->ne[0] == r*n_fresh && keep.fresh_cells->ne[1] == n_ns);
        GGML_ASSERT(keep.fresh_pos->ne[0]   == 4*n_fresh*n_ns);
        GGML_ASSERT(keep.fresh_dst->ne[0]   == n_fresh && keep.fresh_dst->ne[1] == n_ns);
        GGML_ASSERT(keep.blk_src->ne[0]     == n_blocks && keep.blk_src->ne[1] == n_ns);
        GGML_ASSERT(n_fresh <= n_rows - n_buckets && "qsa keep: more pooled rows than spares");
    }
    if (keep_full) {
        GGML_ASSERT(keep.blk_dst->ne[0] == n_blocks && keep.blk_dst->ne[1] == n_ns);
    }
    if (keep_full || keep_incr) {
        GGML_ASSERT(keep.plan->ns == n_ns && n_blocks <= n_buckets);
    }

    std::vector<int32_t> fresh;

    // ggml_set_rows leaves two rows written to one destination undefined, so every row of a write
    // gets a destination of its own: checked here, where a slip would otherwise stay silent
    std::vector<uint8_t> row_used;

    const auto rows_distinct = [&](const int64_t * dst, int64_t n) {
        row_used.assign(n_rows, 0);

        for (int64_t k = 0; k < n; ++k) {
            GGML_ASSERT(dst[k] >= 0 && dst[k] < n_rows);

            if (row_used[dst[k]]) {
                return false;
            }
            row_used[dst[k]] = 1;
        }

        return true;
    };

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    // block top-k picks before the attention mask, so only it can waste a pick (llama-qsa-picks.h)
    const bool    count_picks = blk_bias && cell_blk == nullptr;
    const int64_t n_pick      = llama_qsa_n_block_picks(n_kv, r, hparams_idx.indexer_top_k);

    llama_qsa_pick_count pick_count;

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk == nullptr ? nullptr : dst_cell_blk + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        // the spare block's first cell: the causal rule shows it only to tokens at or after it
        int64_t spare_min = LLAMA_QSA_SPARE_EMPTY;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] < 0 && !cells.is_empty(j)) {
                spare_min = std::min<int64_t>(spare_min, ranked ? rank[j] : cells.pos_get(j));
            }

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            if (cur_cell_blk != nullptr) {
                cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
                continue;
            }

            // Block top-k has no cell_blk, so an unpooled cell cannot be routed to the spare block
            // by per-cell lookup - it has to appear IN that block's member list or it drops out of
            // the selection entirely, and the unpooled cells are the incomplete TAIL, i.e. the
            // newest tokens. Place each at its own position slot, exactly as a full block does.
            // One sequence has at most r-1 of them, so they never collide; several sequences in one
            // stream can, and the last writer wins - which still beats what the arithmetic version
            // did there, namely select unrelated cells.
            if (have_dead && blk_of[j] < 0 && !cells.is_empty(j)) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                cur_blk_cells[dead_bid*r + (idx%r)] = (int32_t) j;
            }
        }

        if (keep_full || keep_incr) {
            // the planner's cheap checks must agree with the full scan above: a stream it let through
            // holds one sequence, no repeated position, and nothing past the cell window
            GGML_ASSERT(one_seq && !dup && !oor && !ranked && "qsa keep: the plan disagrees with the cells");
            GGML_ASSERT(get_mem_idx()->seq_stream(seq_of_stream) == keep.plan->s0 + s);

            // a full block's store row is its position bucket. the unused rows keep today's values
            // (the dead block's own members, cell 0 at position 0 for the rest) and are pooled afresh,
            // each into a spare row of its own: the spares past the buckets first, then the buckets
            // of no full block, of which a FULL write of every block id can need more than the spares
            int64_t spare_next = n_buckets;
            int64_t spare_pb   = 0;

            const auto spare_row = [&]() -> int32_t {
                if (spare_next < n_rows) {
                    return (int32_t) spare_next++;
                }

                // one sequence in the stream: at most one group per bucket, full or not
                while (spare_pb < n_blocks && grp_head[spare_pb] >= 0 && grp_bid[grp_head[spare_pb]] >= 0) {
                    spare_pb++;
                }

                GGML_ASSERT(spare_pb < n_buckets && "qsa keep: out of spare rows");

                return (int32_t) spare_pb++;
            };

            if (keep_full) {
                int64_t * cur_blk_dst = (int64_t *) keep.blk_dst->data + s*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    cur_blk_dst[b] = b < n_bid ? bid_idx[b]/(int32_t) r : spare_row();
                }

                GGML_ASSERT(rows_distinct(cur_blk_dst, n_blocks) && "qsa keep: two blocks written to one row");
            }

            if (keep_incr) {
                // the full blocks this ubatch wrote a key into; the rest are in the store already
                fresh.clear();

                for (int64_t ii = 0; ii < n_tps; ++ii) {
                    const int64_t pb = ubatch->pos[s*n_tps + ii]/r;
                    const int32_t g  = grp_head[pb];

                    if (g >= 0 && grp_bid[g] >= 0) {
                        fresh.push_back(grp_bid[g]);
                    }
                }

                std::sort(fresh.begin(), fresh.end());
                fresh.erase(std::unique(fresh.begin(), fresh.end()), fresh.end());

                GGML_ASSERT((int64_t) fresh.size() <= n_slots && "qsa keep: more filled blocks than the plan made room for");

                int32_t * cur_fresh_cells = (int32_t *) keep.fresh_cells->data + s*(r*n_fresh);
                int32_t * cur_fresh_pos   = (int32_t *) keep.fresh_pos->data;
                int64_t * cur_fresh_dst   = (int64_t *) keep.fresh_dst->data + s*n_fresh;

                // rows [0, n_slots) are the filled blocks then fillers, row n_slots the dead block,
                // row n_slots + 1 a filler; each copies today's members and positions of its block
                for (int64_t k = 0; k < n_fresh; ++k) {
                    int64_t b   = -1;
                    int32_t dst = -1;

                    if (k < (int64_t) fresh.size()) {
                        b   = fresh[k];
                        dst = bid_idx[b]/(int32_t) r;
                    } else {
                        b   = k == n_slots && have_dead ? dead_bid : -1;
                        dst = spare_row();
                    }

                    for (int64_t i = 0; i < r; ++i) {
                        cur_fresh_cells[k*r + i] = b < 0 ? 0 : cur_blk_cells[b*r + i];
                    }

                    for (int64_t sec = 0; sec < 4; ++sec) {
                        cur_fresh_pos[sec*(n_fresh*n_ns) + s*n_fresh + k] =
                            b < 0 ? 0 : dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b];
                    }

                    cur_fresh_dst[k] = dst;
                }

                GGML_ASSERT(rows_distinct(cur_fresh_dst, n_fresh) && "qsa keep: two pooled rows written to one row");

                // every block reads its row back: a full block its bucket, the dead block the row
                // just pooled for it, the unused ids the filler pooled last (they all hold that value)
                const int32_t row_dead   = (int32_t) cur_fresh_dst[n_slots];
                const int32_t row_filler = (int32_t) cur_fresh_dst[n_slots + 1];

                int32_t * cur_blk_src = (int32_t *) keep.blk_src->data + s*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b < n_bid) {
                        cur_blk_src[b] = bid_idx[b]/(int32_t) r;
                    } else {
                        cur_blk_src[b] = have_dead && b == dead_bid ? row_dead : row_filler;
                    }
                }
            }
        }

        // a stream shared by several sequences keeps the old rule
        const bool causal = picks.causal && one_seq;

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it.
                // the caller adds the attention mask, which drops empty, foreign and future cells,
                // but block top-k picks before that mask: the causal rule keeps the blocks after
                // the token out of the picks, the old one gave them the tail's 1e9 (llama-qsa-picks.h)
                float * cur_blk_bias = dst_bias + i*n_blocks;

                llama_qsa_row_tally tally;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    cur_blk_bias[b] = llama_qsa_block_bias(bid_idx[b], q, tail_start, causal);
                    tally.add(cur_blk_bias[b], bid_idx[b] > q);
                }

                // the spare block holds the unpooled cells: the incomplete tail, for one sequence
                if (have_dead) {
                    cur_blk_bias[dead_bid] = llama_qsa_spare_bias(spare_min, q, causal);
                    tally.add(cur_blk_bias[dead_bid], spare_min > q);
                }

                if (count_picks) {
                    tally.finish(n_pick, pick_count);
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }

    if (count_picks) {
        picks.stats.n_ubatch++;
        picks.stats.total.add(pick_count);
        picks.stats.last = pick_count;

        if (picks.log) {
            llama_qsa_pick_log(pick_count, n_blocks, n_pick, picks.causal);
        }
    }
}

void llama_memory_hybrid_idx::qsa_keep_init(const llama_model & model, bool offload, uint32_t n_ubatch) {
    keep.enabled  = mem_idx != nullptr && llama_qsa_keep_env("LLAMA_QSA_KEEP", true);
    keep.check    = keep.enabled && llama_qsa_keep_env("LLAMA_QSA_KEEP_CHECK");
    keep.debug    = keep.enabled && llama_qsa_keep_env("LLAMA_QSA_KEEP_DEBUG");
    keep.n_ubatch = n_ubatch;

    if (!keep.enabled) {
        return;
    }

    GGML_ASSERT(n_ubatch > 0);

    const auto & hparams = model.hparams;

    const uint32_t n_stream = mem_idx->get_n_stream();
    const int64_t  n_embd   = hparams.indexer_head_size;

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    const std::vector<uint32_t> layer_ids = mem_idx->get_layer_ids();

    for (const uint32_t il : layer_ids) {
        const uint32_t r = hparams.dsv4_compress_ratios[il];
        if (r == 0) {
            continue; // a dense layer, no summaries
        }

        // on the layer's device, as the indexer cache it summarises
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
        }

        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*layer_ids.size()*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context for the qsa keep store");
            }

            it = ctx_map.emplace(buft, ctx).first;
        }

        // F32: a kept row must equal the rebuilt one bit for bit
        qsa_keep_layer layer;
        layer.rows = ggml_new_tensor_3d(it->second.get(), GGML_TYPE_F32, n_embd, qsa_keep_n_rows(r), n_stream);
        ggml_format_name(layer.rows, "qsa_keep_rows_l%d", il);

        if (keep.check) {
            layer.sum = ggml_new_tensor_1d(it->second.get(), GGML_TYPE_F32, 1);
            ggml_format_name(layer.sum, "qsa_keep_check_l%d", il);
        }

        keep.layers[il] = layer;

        if (std::find(keep.ratios.begin(), keep.ratios.end(), r) == keep.ratios.end()) {
            keep.ratios.push_back(r);
            keep.synced[r].assign(n_stream, 1);
        }
    }

    keep.dup.assign(n_stream, QSA_DUP_FREE);

    size_t size = 0;

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams_idx.no_alloc) {
            // same as the KV cache: a dummy buffer, so the scheduler does not allocate the tensors
            buf = ggml_backend_buft_alloc_buffer(buft, 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for the qsa keep store");
        }

        ggml_backend_buffer_clear(buf, 0);

        size += ggml_backend_buffer_get_size(buf);

        keep.ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    LLAMA_LOG_INFO("%s: qsa keep: %zu layers, %u streams, %.2f MiB%s%s\n", __func__,
            keep.layers.size(), n_stream, size/1024.0/1024.0,
            keep.check ? ", check" : "", keep.debug ? ", debug" : "");
}

ggml_tensor * llama_memory_hybrid_idx::qsa_keep_rows(int32_t il) const {
    const auto it = keep.layers.find(il);
    GGML_ASSERT(it != keep.layers.end());

    return it->second.rows;
}

ggml_tensor * llama_memory_hybrid_idx::qsa_keep_sum(int32_t il) const {
    const auto it = keep.layers.find(il);
    GGML_ASSERT(it != keep.layers.end() && it->second.sum != nullptr);

    return it->second.sum;
}

uint32_t llama_memory_hybrid_idx::qsa_keep_n_buckets(uint32_t ratio) const {
    GGML_ASSERT(ratio > 0 && mem_idx != nullptr);

    // every cache path keeps a cell's position below the cell count
    return (mem_idx->get_size() + ratio - 1)/ratio;
}

uint32_t llama_memory_hybrid_idx::qsa_keep_n_spare(uint32_t ratio) const {
    GGML_ASSERT(ratio > 0 && keep.n_ubatch > 0);

    // the rows INCR pools when every one of the largest ubatch's tokens fills a block (its n_slots,
    // see qsa_keep_plan), plus the dead block's and the filler's: one spare each, so that no two
    // rows of one write share a destination
    return (keep.n_ubatch + ratio - 2)/ratio + 1 + 2;
}

uint32_t llama_memory_hybrid_idx::qsa_keep_n_rows(uint32_t ratio) const {
    return qsa_keep_n_buckets(ratio) + qsa_keep_n_spare(ratio);
}

// true if two cells of the stream share a position; the caller knows the stream holds one sequence
static bool llama_qsa_keep_has_dup(const llama_kv_cells & cells, llama_pos pos_max) {
    std::vector<uint8_t> seen(pos_max + 1, 0);

    for (uint32_t j = 0; j < cells.used_max_p1(); ++j) {
        if (cells.is_empty(j)) {
            continue;
        }

        const llama_pos p = cells.pos_get(j);
        if (seen[p]) {
            return true;
        }
        seen[p] = 1;
    }

    return false;
}

std::vector<llama_qsa_keep_plan> llama_memory_hybrid_idx::qsa_keep_plan(
        const llama_ubatch & ubatch, uint32_t s0, uint32_t ns, uint32_t n_kv,
        const std::vector<llama_pos> & pos_max_prev) {
    GGML_ASSERT(keep.enabled);
    GGML_ASSERT(ns > 0 && ubatch.n_tokens % ns == 0 && pos_max_prev.size() == ns);

    // a plan still in flight belongs to a ubatch whose compute never finished
    qsa_keep_settle();

    const uint32_t n_tps = ubatch.n_tokens/ns;

    // per stream, whatever the ratio: its sequence's last position, or -1 when it cannot keep rows
    std::vector<llama_pos> pos_max(ns, -1);

    for (uint32_t s = 0; s < ns; ++s) {
        const llama_seq_id seq = ubatch.seq_id[s*n_tps][0];
        const uint32_t     st  = s0 + s;

        GGML_ASSERT(mem_idx->seq_stream(seq) == st);

        const auto & cells = mem_idx->get_cells(seq);

        // the store keys a block by position alone, so a second sequence in the stream would share it
        int n_seq_present = 0;
        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        // repeats are tracked for a sequence alone in its stream: the positions below belong to it.
        // a shared stream looks again once one sequence is left
        if (n_seq_present != 1) {
            keep.dup[st] = QSA_DUP_UNKNOWN;
            continue;
        }

        // positions past everything the stream held, rising: no repeat can have appeared
        if (keep.dup[st] == QSA_DUP_FREE) {
            llama_pos prev = pos_max_prev[s];

            for (uint32_t ii = 0; ii < n_tps; ++ii) {
                const llama_pos p = ubatch.pos[s*n_tps + ii];
                if (p <= prev) {
                    keep.dup[st] = QSA_DUP_UNKNOWN;
                    break;
                }
                prev = p;
            }
        }

        const llama_pos pmax = cells.seq_pos_max(seq);

        // beyond twice the cells the position cannot sit in any window; skip the scan's bitmap
        if (pmax < 0 || pmax >= 2*(llama_pos) cells.size()) {
            continue;
        }

        if (keep.dup[st] == QSA_DUP_UNKNOWN) {
            keep.dup[st] = llama_qsa_keep_has_dup(cells, pmax) ? QSA_DUP_PRESENT : QSA_DUP_FREE;
        }

        if (keep.dup[st] == QSA_DUP_FREE) {
            pos_max[s] = pmax;
        }
    }

    std::vector<llama_qsa_keep_plan> plans;

    std::vector<int64_t> touched;

    for (const uint32_t r : keep.ratios) {
        const int64_t n_blocks = (n_kv + r - 1)/r;

        llama_qsa_keep_plan plan;
        plan.ratio = r;
        plan.s0    = s0;
        plan.ns    = ns;

        // blocks n_tps positions in a row can reach; the spares were sized for the largest ubatch
        const int64_t n_slots = std::min<int64_t>(n_blocks, (n_tps + r - 2)/r + 1);

        GGML_ASSERT(n_slots + 2 <= (int64_t) qsa_keep_n_spare(r) && "qsa keep: a ubatch larger than the spares were sized for");

        bool eligible = true;
        bool synced   = true;

        for (uint32_t s = 0; s < ns; ++s) {
            eligible = eligible && pos_max[s] >= 0 && pos_max[s]/r < n_blocks;
            synced   = synced   && keep.synced.at(r)[s0 + s];
        }

        if (!eligible) {
            plan.mode = LLAMA_QSA_KEEP_LEGACY;
        } else if (!synced || n_slots + 2 >= n_blocks) {
            // a rebuild, or pooling the few fresh rows would cost as much as pooling them all
            plan.mode = LLAMA_QSA_KEEP_FULL;
        } else {
            plan.mode = LLAMA_QSA_KEEP_INCR;

            // positions that jump can touch more blocks than n_slots; set_input asserts the bound
            for (uint32_t s = 0; s < ns && plan.mode == LLAMA_QSA_KEEP_INCR; ++s) {
                touched.clear();
                for (uint32_t ii = 0; ii < n_tps; ++ii) {
                    touched.push_back(ubatch.pos[s*n_tps + ii]/r);
                }

                std::sort(touched.begin(), touched.end());

                if (std::unique(touched.begin(), touched.end()) - touched.begin() > n_slots) {
                    plan.mode = LLAMA_QSA_KEEP_FULL;
                }
            }

            if (plan.mode == LLAMA_QSA_KEEP_INCR) {
                plan.n_slots = (uint32_t) n_slots;
            }
        }

        if (keep.debug) {
            LLAMA_LOG_INFO("%s: qsa keep: n_tokens = %4u, streams [%u, %u), n_kv = %6u, ratio = %u, n_blocks = %6" PRId64
                    ", mode = %-6s, n_slots = %4u, synced = %d, committed LEGACY/FULL/INCR = %" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
                    __func__, ubatch.n_tokens, s0, s0 + ns, n_kv, r, n_blocks,
                    llama_qsa_keep_mode_name(plan.mode), plan.n_slots, synced ? 1 : 0,
                    keep.n_legacy, keep.n_full, keep.n_incr);
        }

        plans.push_back(plan);
    }

    keep.inflight = plans;

    return plans;
}

std::vector<llama_qsa_keep_plan> llama_memory_hybrid_idx::qsa_keep_plan_reserve(uint32_t n_stream) const {
    std::vector<llama_qsa_keep_plan> plans;

    for (const uint32_t r : keep.ratios) {
        llama_qsa_keep_plan plan;
        plan.mode  = LLAMA_QSA_KEEP_FULL;
        plan.ratio = r;
        plan.s0    = 0;
        plan.ns    = n_stream;

        plans.push_back(plan);
    }

    return plans;
}

void llama_memory_hybrid_idx::qsa_keep_commit() {
    for (const auto & plan : keep.inflight) {
        auto & synced = keep.synced.at(plan.ratio);

        switch (plan.mode) {
            case LLAMA_QSA_KEEP_LEGACY:
                {
                    // blocks it filled were never written into the store
                    std::fill(synced.begin() + plan.s0, synced.begin() + plan.s0 + plan.ns, 0);
                    keep.n_legacy++;
                } break;
            case LLAMA_QSA_KEEP_FULL:
                {
                    std::fill(synced.begin() + plan.s0, synced.begin() + plan.s0 + plan.ns, 1);
                    keep.n_full++;
                } break;
            case LLAMA_QSA_KEEP_INCR:
                {
                    keep.n_incr++;
                } break;
        }
    }

    keep.inflight.clear();
}

void llama_memory_hybrid_idx::qsa_keep_settle() {
    for (const auto & plan : keep.inflight) {
        auto & synced = keep.synced.at(plan.ratio);

        std::fill(synced.begin() + plan.s0, synced.begin() + plan.s0 + plan.ns, 0);
    }

    keep.inflight.clear();
}

void llama_memory_hybrid_idx::qsa_keep_unsync(llama_seq_id seq_id) {
    qsa_keep_settle();

    if (!keep.enabled) {
        return;
    }

    for (uint32_t s = 0; s < keep.dup.size(); ++s) {
        if (seq_id >= 0 && mem_idx->seq_stream(seq_id) != s) {
            continue;
        }

        for (auto & [ratio, synced] : keep.synced) {
            synced[s] = 0;
        }

        keep.dup[s] = QSA_DUP_UNKNOWN;
    }
}

void llama_memory_hybrid_idx::qsa_keep_dup_unknown(llama_seq_id seq_id) {
    if (!keep.enabled) {
        return;
    }

    const uint32_t st = seq_id < 0 ? 0 : mem_idx->seq_stream(seq_id);

    for (uint32_t s = 0; s < keep.dup.size(); ++s) {
        if (keep.dup[s] == QSA_DUP_PRESENT && (seq_id < 0 || st == s)) {
            keep.dup[s] = QSA_DUP_UNKNOWN;
        }
    }
}

llama_qsa_keep_stats llama_memory_hybrid_idx::qsa_keep_stats() const {
    llama_qsa_keep_stats res;

    res.n_legacy = keep.n_legacy;
    res.n_full   = keep.n_full;
    res.n_incr   = keep.n_incr;

    if (keep.check && !hparams_idx.no_alloc) {
        for (const uint32_t il : mem_idx->get_layer_ids()) {
            const auto it = keep.layers.find(il);
            if (it == keep.layers.end()) {
                continue;
            }

            float sum = 0.0f;
            ggml_backend_tensor_get(it->second.sum, &sum, 0, sizeof(sum));

            res.check.emplace_back(il, sum);
        }
    }

    return res;
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// first stream of each ubatch's slot info, where get_k/get_v's views start
static std::vector<uint32_t> llama_memory_hybrid_idx_s0(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s0);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    s0_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ 0 }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())),
    // the reserved graph is the largest one: every row pooled and written into the store
    qsa_plans(mem->get_mem_idx() == nullptr ? std::vector<llama_qsa_keep_plan>() :
        mem->qsa_keep_plan_reserve(mem->get_mem_idx()->get_n_stream())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    s0_ubatch(llama_memory_hybrid_idx_s0(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)),
    is_batch(true) {}

bool llama_memory_hybrid_idx_context::next() {
    // next() follows only a compute that finished, so this ubatch's plans take effect now
    if (qsa_planned) {
        mem->qsa_keep_commit();
        qsa_planned = false;
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    const bool plan = is_batch && ctx_idx && mem->qsa_keep();

    // each stream's last position before this ubatch lands: new positions above it cannot repeat one
    std::vector<llama_pos> pos_max_prev;

    if (plan) {
        const auto &   ubatch = get_ubatch();
        const uint32_t ns     = get_n_stream();

        for (uint32_t s = 0; s < ns; ++s) {
            const llama_seq_id seq = ubatch.seq_id[s*(ubatch.n_tokens/ns)][0];

            pos_max_prev.push_back(mem->get_mem_idx()->get_cells(seq).seq_pos_max(seq));
        }
    }

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // planned after the indexer's apply: the cells hold the ubatch and n_kv is known
    if (plan) {
        qsa_plans   = mem->qsa_keep_plan(get_ubatch(), s0_ubatch[i_cur], get_n_stream(), get_idx()->get_n_kv(), pos_max_prev);
        qsa_planned = true;
    }

    return res;
}

llama_ple_trace * llama_memory_hybrid_idx_context::get_ple_trace() const {
    return mem != nullptr ? mem->get_ple_trace() : nullptr;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

uint32_t llama_memory_hybrid_idx_context::get_stream0() const {
    GGML_ASSERT(i_cur < s0_ubatch.size());

    return s0_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        int64_t n_kv,
        bool blk_bias,
        const llama_qsa_keep_inputs & keep) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, n_kv, blk_bias, keep);
}

const llama_qsa_keep_plan & llama_memory_hybrid_idx_context::get_qsa_plan(uint32_t ratio) const {
    for (const auto & plan : qsa_plans) {
        if (plan.ratio == ratio) {
            return plan;
        }
    }

    // switch off, or no plan made for this context
    static const llama_qsa_keep_plan legacy;

    return legacy;
}

bool llama_memory_hybrid_idx_context::get_qsa_keep_check() const {
    return mem->qsa_keep_check();
}

ggml_tensor * llama_memory_hybrid_idx_context::get_qsa_keep_rows(int32_t il) const {
    return mem->qsa_keep_rows(il);
}

ggml_tensor * llama_memory_hybrid_idx_context::get_qsa_keep_sum(int32_t il) const {
    return mem->qsa_keep_sum(il);
}

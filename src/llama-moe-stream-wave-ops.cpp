#include "llama-moe-stream.h"

#include <mutex>

// Shared preamble of the two wave ops: plan the whole ubatch on wave 0, enforce that the waves run in
// order, and make wave w's expert slice resident while preloading the next. Returns with the manager
// mutex still held, because the emit that follows reads the slot table this just settled.
static std::unique_lock<std::mutex> stage_wave_for_op(llama_moe_stream_layer & sl, int32_t w,
        const int32_t * ids, int64_t n, uint32_t n_ids) {
    auto * mgr = sl.mgr;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_wave_calls++;

    if (w == 0) {
        mgr->plan_waves_locked(sl, ids, n);
        // the ubatch's own routing, recorded once per layer rather than once per wave. Always
        // kind 1: the multi-wave path only runs when a ubatch touches more experts than the cache
        // holds, which is reading in.
        mgr->trace_record_locked(sl.il, n_ids > 0 ? (uint32_t) (n/n_ids) : 0, /*kind =*/ 1, ids, n);
    }
    GGML_ASSERT(sl.plan_next_wave == w); // waves must run in order (enforced by the graph ordering token)

    mgr->stage_wave_locked(lk, sl, w, n_ids); // make this wave resident, preload the next, build the pool
    sl.plan_next_wave = w + 1;

    return lk;
}

// Custom-op callback for one pass of multi-pass prefill. When a ubatch touches more experts than the
// cache holds, build_moe_ffn runs the expert GEMMs in several waves; this runs once per wave (single-
// threaded on ith 0), in wave order. For wave w it makes that wave's expert slice resident (preloading
// the next wave), then writes the slot ids the GEMM indexes - see plan_waves_locked / stage_wave_locked
// / emit_wave_slots. The router's expert choice is untouched, so the output matches a non-streamed run.
void llama_moe_stream_wave_ids(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    // timed from before the lock: lock acquisition is on the critical path too, since the GPU sits
    // idle while this CPU-side op runs as a graph dependency
    const int64_t t_op0 = ggml_time_us();

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));
    GGML_ASSERT(dst->data != a->data); // the emit must not clobber the ids other waves read

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk = stage_wave_for_op(*sl, w, ids, n, (uint32_t) a->ne[0]);

    mgr->emit_wave_slots(*sl, ids, out, w, (uint32_t) a->ne[0], a->ne[1]);

    mgr->stats.t_wave_op_us += ggml_time_us() - t_op0;
}

// Partition path (LLAMA_MOE_STREAM_PARTITION=1) counterpart of llama_moe_stream_wave_ids: identical
// staging, but instead of slot ids for every pair it emits the index rows of wave w's own pairs, which
// build_moe_ffn gathers into a dense GEMM and scatters back. No mask op is needed - a pair is computed
// by the one wave that owns it, so there is nothing to discard.
void llama_moe_stream_wave_pairs(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    const int64_t t_op0 = ggml_time_us();

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(dst->ne[1] == LLAMA_MOE_PAIR_ROWS);

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk = stage_wave_for_op(*sl, w, ids, n, (uint32_t) a->ne[0]);

    mgr->emit_wave_pairs(*sl, ids, out, w, (uint32_t) a->ne[0], n, dst->ne[0]);

    mgr->stats.t_wave_op_us += ggml_time_us() - t_op0;
}

// multi-pass prefill: 1.0 for pairs whose expert belongs to wave w, 0.0 otherwise; multiplied into
// this wave's expert GEMM output so the masked-out (parked) pairs contribute nothing to the sum
void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          float   * out = (float *) dst->data;

    std::lock_guard<std::mutex> lock(mgr->mtx);

    GGML_ASSERT(sl->plan_next_wave > w); // this wave's ids op has already run

    for (int64_t i = 0; i < n; i++) {
        out[i] = sl->expert_wave[ids[i]] == (uint8_t) w ? 1.0f : 0.0f;
    }
}

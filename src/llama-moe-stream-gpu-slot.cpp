#include "llama-moe-stream.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>

// Mirror the CPU's residency into the device table the resolve kernel reads, and collect what the
// kernel reported since the last call.
//
// At mode 3 the GPU owns this table between calls, but the CPU path still runs for prefill (see
// LLAMA_MOE_GPU_SLOT_MAX_TOKENS), so every CPU-path call republishes the whole thing. That is what
// keeps the handover consistent in both directions: the servicer mirrors every GPU assignment back
// into the CPU structures, and this mirrors the CPU structures back out.
void llama_moe_stream_layer::publish_state_locked(llama_moe_stream & mgr) {
    if (state_host == nullptr) {
        state_host = (int32_t *) ggml_backend_tensor_get_host_ptr(state);
        if (state_host == nullptr) {
            return;
        }
    }

    const llama_moe_slot_state_layout lay = layout();

    int32_t * se   = state_host + lay.off_slot_expert();
    int32_t * lu   = state_host + lay.off_last_use();
    int32_t * es   = state_host + lay.off_expert_slot();
    int32_t * tail = state_host + lay.off_tail();

    if (state_init && tail[MOE_SLOT_TAIL_NBAD] != 0) {
        if (mgr.n_slot_warn < 8) {
            mgr.n_slot_warn++;
            LLAMA_LOG_WARN("%s: moe stream: GPU slot resolve DISAGREES with the CPU remap "
                    "(layer %d, %d pairs) - the CPU answer is still what is used\n",
                    __func__, il, tail[MOE_SLOT_TAIL_NBAD]);
        }
        mgr.n_slot_bad += tail[MOE_SLOT_TAIL_NBAD];
    }
    state_init = true;

    for (uint32_t s = 0; s < n_slots; s++) {
        se[s] = slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY ? -1 : slot_expert[s];
        lu[s] = (int32_t) slot_last_use[s];
    }
    std::fill(es, es + n_expert, -1);
    for (const auto & [e, sl] : expert_slot) {
        es[e] = sl;
    }

    tail[MOE_SLOT_TAIL_CLOCK] = (int32_t) use_counter;
    tail[MOE_SLOT_TAIL_NREQ]  = 0;
    tail[MOE_SLOT_TAIL_NBAD]  = 0;
}

// Load whatever the resolve kernel decided it needs. Runs on Metal's listener queue with the GPU
// stalled on a shared event, so this is the whole of the CPU's job at mode 3 - the kernel already
// chose the slots and wrote the ids the GEMM will use.
void llama_moe_stream_layer::service_requests(llama_moe_stream & mgr) {
    if (state_host == nullptr) {
        state_host = (int32_t *) ggml_backend_tensor_get_host_ptr(state);
        if (state_host == nullptr) {
            return;
        }
    }

    const llama_moe_slot_state_layout lay = layout();

    int32_t *       tail = state_host + lay.off_tail();
    const int32_t * req  = state_host + lay.off_requests();

    const int32_t nreq = tail[MOE_SLOT_TAIL_NREQ];

    // LLAMA_MOE_STREAM_SVC_NOLOCK=1 (probe): a call with nothing to load returns without the
    // manager lock, which the I/O workers also hold - the GPU waits on this reply at every layer.
    // Its only work under the lock is the call counter and the periodic stats line, so both
    // undercount decode calls while this is on.
    static const bool svc_nolock = [] {
        const char * s = std::getenv("LLAMA_MOE_STREAM_SVC_NOLOCK");
        return s && atoi(s) > 0;
    }();
    if (svc_nolock && nreq == 0 && tail[MOE_SLOT_TAIL_NBAD] == 0) {
        return;
    }

    std::unique_lock<std::mutex> lk(mgr.mtx);

    if (mgr.load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }
    if (tail[MOE_SLOT_TAIL_NBAD] != 0 || nreq < 0 || (uint32_t) nreq > n_slots) {
        GGML_ABORT("MoE expert streaming: invalid GPU slot request (layer %d, requests %d, error %d)",
                il, nreq, tail[MOE_SLOT_TAIL_NBAD]);
    }

    mgr.stats.n_calls++;
    mgr.maybe_dump_stats_locked();

    if (nreq <= 0) {
        return;
    }

    mgr.start_workers_locked();

    const int64_t t0 = ggml_time_us();

    demand_slots.clear();

    for (int32_t i = 0; i < nreq; i++) {
        const int32_t e = req[2*i + 0];
        const int32_t s = req[2*i + 1];

        if (e < 0 || (uint32_t) e >= n_expert || s < 0 || (uint32_t) s >= n_slots) {
            GGML_ABORT("MoE expert streaming: GPU emitted invalid expert/slot pair (%d, %d) for layer %d",
                    e, s, il);
        }

        if (!seen[e]) {
            mgr.stats.n_miss_cold++;
            n_miss_cold++;
        }

        // mirror the GPU's decision so the reader threads and a later CPU-path call agree with it
        mgr.reserve_slot_locked(*this, e, s);

        slot_pending[s] = (uint8_t) weights.size();
        for (size_t wi = 0; wi < weights.size(); wi++) {
            mgr.q_demand.push_back({ this, e, s, (int32_t) wi, slot_gen[s] });
            mgr.cv_work.notify_one();
        }

        mgr.stats.n_miss++;
        n_miss++;
        demand_slots.push_back(s);
    }

    mgr.cv_done.wait(lk, [&]{
        if (mgr.load_failed) {
            return true;
        }
        for (const int32_t s : demand_slots) {
            if (slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                return false;
            }
        }
        return true;
    });

    if (mgr.load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr.stats.t_stall_us += ggml_time_us() - t0;
}

void llama_moe_stream_service_gpu(void * user_data, void * state_host, int32_t layer) {
    auto * mgr = (llama_moe_stream *) user_data;

    llama_moe_stream_layer * sl = mgr->layer(layer);
    if (sl == nullptr || sl->state_host != state_host) {
        GGML_ABORT("MoE expert streaming: invalid Metal service request for layer %d", layer);
    }
    sl->service_requests(*mgr);
}

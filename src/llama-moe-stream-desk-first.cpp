// Desk books first (study #39): the floor stop that does not wait, and the gate that waits instead. What it is and
// why it is exact: llama-moe-stream-desk-first.h. The graph side is llama-graph-moe-desk-first.cpp.

#include "llama-moe-stream-desk-first.h"

#include "llama-moe-stream.h"
#include "llama-moe-stream-impl.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

bool llama_moe_stream_desk_first_env() {
    const char * s = std::getenv("LLAMA_MOE_STREAM_DESK_FIRST");
    return s != nullptr && *s != '\0' && std::strcmp(s, "0") != 0;
}

// ggml_backend_metal_set_moe_gate (ggml-metal.h), found through the backend's registry so this file needs no Metal
typedef void (*desk_first_gate_fn_t)(void * user_data, void * gate_userdata);
typedef void (*desk_first_set_gate_t)(ggml_backend * backend, desk_first_gate_fn_t fn, void * user_data);

bool llama_moe_stream_desk_first_register(llama_moe_stream & mgr, ggml_backend * const * backends, size_t n_backends,
        bool mtp) {
    if (!llama_moe_stream_desk_first_env() || mtp) {
        return false; // the apprentice's floors keep the plain stop: its graphs build no gates
    }
    const char * why = nullptr;
    if (mgr.gpu_slot != 0) {
        why = "the GPU slot servicer (LLAMA_MOE_STREAM_GPU_SLOT) is on";
    } else if (mgr.state_log) {
        why = "the state log (LLAMA_MOE_STREAM_STATE_LOG) is on";
    }
    for (const auto & c : mgr.ctxs) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(c.first);
        const bool gpu = dev != nullptr && (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (why == nullptr && !gpu) {
            why = "the desk is not the GPU's";
        }
    }
    desk_first_set_gate_t set = nullptr;
    ggml_backend * metal = nullptr;
    for (size_t i = 0; i < n_backends && set == nullptr; i++) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backends[i]);
        ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg != nullptr) {
            set   = (desk_first_set_gate_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_set_moe_gate");
            metal = backends[i];
        }
    }
    if (why == nullptr && set == nullptr) {
        why = "no Metal backend";
    }
    if (why != nullptr) {
        LLAMA_LOG_WARN("%s: moe stream: desk first: off, %s\n", __func__, why);
        return false;
    }
    // only now: with a fn set, Metal scans every graph of this backend for gates
    set(metal, llama_moe_stream_desk_first_gate, &mgr);
    LLAMA_LOG_WARN("%s: moe stream: desk first: on (batches of up to %lld tokens)\n", __func__,
            (long long) LLAMA_MOE_DESK_FIRST_MAX_TOKENS);
    return true;
}

// the stop: the remap's work up to its wait, then both planes; returns whether a book is still arriving
static bool llama_moe_stream_desk_first_stop(ggml_tensor * dst, llama_moe_stream_layer & sl) {
    const ggml_tensor * a = dst->src[0];
    GGML_ASSERT(a->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(dst));
    GGML_ASSERT(dst->ne[0] == a->ne[0] && dst->ne[1] == a->ne[1] && dst->ne[2] == 2 && ggml_nrows(a) == a->ne[1]);

    const int64_t t_op0 = ggml_time_us(); // from before the lock, as the remap's
    llama_moe_stream_stops * stops = llama_moe_stream_stops::get();
    const int64_t t_stop0 = stops ? llama_moe_stream_stops::now_ns() : 0;
    llama_moe_stream_stop_times times;

    auto * mgr = sl.mgr;
    auto & df  = sl.df;
    if (df.armed) {
        GGML_ABORT("moe stream: desk first: floor %d stopped again before its gate ran (is the floor on the GPU? "
                   "LLAMA_MOE_STREAM_DESK_FIRST=0 turns desk books first off)", sl.il);
    }

    const int64_t   n       = ggml_nelements(a);
    const int32_t * ids     = (const int32_t *) a->data;
          int32_t * out     = (int32_t *) dst->data;
          int32_t * out_arr = out + n;

    bool split = false;
    if (mgr->remap_quick(sl, ids, out, n, a->ne[1], t_op0, out_arr)) {
        sl.demand_slots.clear(); // every book on the desk: the gate has nothing to wait for
    } else {
        std::unique_lock<std::mutex> lk(mgr->mtx);
        df.arriving.assign(sl.n_slots, 0);
        split = llama_moe_stream_classify_locked(lk, sl, ids, n, a->ne[1], stops ? &times : nullptr,
                df.arriving.data());
        // the gate waits on the arrivals only; the desk's books are RESIDENT and stay so (only this thread evicts)
        auto & d = sl.demand_slots;
        d.erase(std::remove_if(d.begin(), d.end(), [&](int32_t s) { return !df.arriving[s]; }), d.end());
        llama_moe_stream_remap_out(sl, ids, out, n, out_arr, df.arriving.data());
        mgr->stats.t_remap_op_us += ggml_time_us() - t_op0;
    }

    auto & st = mgr->stats; // graph thread only, as the guesser's counts
    st.n_df_stops++;
    st.n_df_split += split;
    for (int64_t i = 0; i < n; i++) {
        st.n_df_rows_desk += out[i] >= 0;
        st.n_df_rows_arr  += out_arr[i] >= 0;
    }
    df.armed = true;

    if (stops) {
        const auto rec = stops->make(sl.il, t_stop0, llama_moe_stream_stops::now_ns(), (uint32_t) sl.uniq.size(),
                times.trips, times.send_ns, times.wait_ns);
        if (split) {
            df.rec      = rec; // put by the gate, with its wait
            df.rec_held = true;
        } else {
            stops->put_rec(rec);
        }
    }
    return split;
}

void llama_moe_stream_desk_first_remap(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    llama_moe_stream_desk_first_stop(dst, *(llama_moe_stream_layer *) userdata);
}

// with trips on their way the guesses wait for the gate, which sends them after its wait, where the remap's stop
// would have: the runners take the floor's own trips first, as without desk books first
void llama_moe_stream_desk_first_remap_la(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * la = (llama_moe_stream_lookahead *) userdata;
    const bool split = llama_moe_stream_desk_first_stop(dst, *la->sl);
    llama_moe_stream_lookahead_run(la, dst->src[1], split ? &la->sl->df : nullptr);
}

void llama_moe_stream_desk_first_gate(void * user_data, void * gate_userdata) {
    auto * sl  = (llama_moe_stream_layer *) gate_userdata;
    auto * mgr = sl->mgr;
    GGML_ASSERT(mgr == user_data);
    auto & df = sl->df;
    if (!df.armed) {
        GGML_ABORT("moe stream: desk first: floor %d's gate ran with no stop before it", sl->il);
    }
    df.armed = false;

    int64_t wait_ns = 0;
    if (!sl->demand_slots.empty()) {
        const int64_t t0    = ggml_time_us();
        const int64_t t0_ns = df.rec_held ? llama_moe_stream_stops::now_ns() : 0;
        std::unique_lock<std::mutex> lk(mgr->mtx);
        bool landed = true;
        for (const int32_t s : sl->demand_slots) {
            landed = landed && sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT;
        }
        if (!landed) {
            mgr->wait_trips_locked(lk, *sl, t0); // every arrival RESIDENT, or a load failed (keep-awake pings)
            if (mgr->load_failed) {
                GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
            }
            mgr->stats.n_df_gate_waits++;
            mgr->stats.t_df_gate_wait_us += ggml_time_us() - t0;
        }
        if (df.rec_held) {
            wait_ns = llama_moe_stream_stops::now_ns() - t0_ns;
        }
    }

    llama_moe_stream_lookahead_send_held(df); // takes the lock itself

    if (df.rec_held) {
        df.rec_held = false;
        df.rec.wait_ns += wait_ns;
        if (llama_moe_stream_stops * stops = llama_moe_stream_stops::get()) {
            stops->put_rec(df.rec);
        }
    }
}

void llama_moe_stream_desk_first_reset(llama_moe_stream & mgr) {
    for (auto & sl : mgr.layers) {
        if (sl) {
            sl->df.armed    = false;
            sl->df.held     = false;
            sl->df.rec_held = false;
            sl->demand_slots.clear();
        }
    }
}

void llama_moe_stream_desk_first_print(const llama_moe_stream_stats & s, const llama_moe_stream_stats * prev) {
    static const llama_moe_stream_stats zero;
    const llama_moe_stream_stats & p = prev != nullptr ? *prev : zero;
    const int64_t stops = s.n_df_stops - p.n_df_stops;
    if (stops <= 0) {
        return;
    }
    const int64_t waits = s.n_df_gate_waits - p.n_df_gate_waits;
    const int64_t t_us  = s.t_df_gate_wait_us - p.t_df_gate_wait_us;
    LLAMA_LOG_WARN("%s: moe stream: desk first: floors %lld, split %lld | picks desk/arriving %lld/%lld | "
                   "gate waits %lld, %.2f ms (%.3f ms each)\n",
            prev != nullptr ? "maybe_dump_stats_locked" : "print_stats", (long long) stops,
            (long long) (s.n_df_split - p.n_df_split), (long long) (s.n_df_rows_desk - p.n_df_rows_desk),
            (long long) (s.n_df_rows_arr - p.n_df_rows_arr), (long long) waits, t_us/1e3,
            waits > 0 ? t_us/1e3/waits : 0.0);
}

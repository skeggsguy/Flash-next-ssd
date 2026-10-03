#include "llama-moe-stream.h"
#include "llama-moe-stream-impl.h"
#include "llama-moe-room.h"
#include "llama-moe-stream-stops.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

// custom-op callback (single-threaded on ith 0): given the router's expert ids, ensure every touched
// expert is resident - reserving cache slots and demand-loading misses, stalling until they commit -
// then rewrite each id to its cache slot. this only relabels ids, so the same experts are computed
// in the same order; the result matches a non-streamed run (bit-exact when both paths use the same
// kernels, as on CUDA; a CPU build that repacks the non-streamed weights can differ in the last bits).
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    // timed from before the lock: lock acquisition is on the critical path too, since the GPU sits
    // idle while this CPU-side op runs as a graph dependency
    const int64_t t_op0 = ggml_time_us();

    // the floor-stop log (llama-moe-stream-stops.h): one pointer check when off
    llama_moe_stream_stops * stops = llama_moe_stream_stops::get();
    const int64_t t_stop0 = stops ? llama_moe_stream_stops::now_ns() : 0;
    int64_t  stop_send_ns = 0;
    int64_t  stop_wait_ns = 0;
    uint32_t stop_trips   = 0;

    auto * sl  = (llama_moe_stream_layer *) userdata;
    auto * mgr = sl->mgr;

    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    const int64_t n = ggml_nelements(a);

    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    mgr->maybe_dump_stats_locked();

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_calls++;
    mgr->start_workers_locked();

    // the borrowing log. One record per call, before any of the staging below, so a record means
    // "the router chose this", not "the cache managed to serve it". kind 0 is a single decode
    // token; a multi-token ubatch reaching the single-wave path is reading in, so it is kind 1.
    mgr->trace_record_locked(sl->il, (uint32_t) a->ne[1], a->ne[1] == 1 ? 0 : 1, ids, n);

    // distinct experts touched by this ubatch, in first-use order
    sl->touched.assign(sl->n_expert, 0);
    sl->uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl->n_expert);
        if (!sl->touched[e]) {
            sl->touched[e] = 1;
            sl->uniq.push_back(e);
        }
    }

    if (sl->uniq.size() > sl->n_slots) {
        GGML_ABORT("MoE expert streaming: layer %d needs %zu distinct experts but the cache has only %u slots; "
                   "increase --moe-stream-cache or reduce the ubatch size (-ub)",
                sl->il, sl->uniq.size(), sl->n_slots);
    }

    // route hotness for eviction; halved periodically so a formerly-hot expert ages out
    for (const int32_t e : sl->uniq) {
        sat_inc(sl->route_hotness[e]);
    }
    if (mgr->hot_decay_interval > 0 && mgr->stats.n_calls % mgr->hot_decay_interval == 0) {
        for (auto & sl2 : mgr->layers) {
            if (sl2) {
                for (auto & h : sl2->route_hotness) {
                    h >>= 1;
                }
            }
        }
    }

    // classify the touched experts; reserve and enqueue demand loads in deterministic order
    std::fill(sl->keep.begin(), sl->keep.end(), 0);
    sl->demand_slots.clear();

    // the lent belt (llama-moe-room-lend-ops.cpp): a miss may be copied back from it, and each book this
    // call puts back is kept on it; the desk's own choices below are the same with it off
    llama_moe_room * lender = mgr->room && mgr->room->lend_on ? mgr->room.get() : nullptr;
    uint32_t n_kept = 0;

    bool waited = false;
    for (const int32_t e : sl->uniq) {
        const auto it = sl->expert_slot.find(e);
        if (it != sl->expert_slot.end()) {
            const int32_t s = it->second;
            if (sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                mgr->promote_slot_locked(*sl, s);
                waited = true;
                mgr->stats.n_hit_loading++;
            } else {
                mgr->stats.n_hit_ready++;
            }
            mgr->stats.n_hit++;
            sl->n_hit++;
            sl->keep[s] = 1;
            sl->demand_slots.push_back(s);
        } else {
            const int64_t t_send0 = stops ? llama_moe_stream_stops::now_ns() : 0;
            const uint64_t lent = lender ? lender->lend_find_locked(*sl, e, false) : 0; // pinned before any keep
            int32_t v;
            while ((v = mgr->pick_victim_locked(*sl, sl->keep.data())) < 0) {
                // every allowed slot is loading; wait for a commit and retry
                const int64_t t_w0 = stops ? llama_moe_stream_stops::now_ns() : 0;
                mgr->cv_done.wait(lk);
                if (stops) {
                    const int64_t dt = llama_moe_stream_stops::now_ns() - t_w0;
                    stop_wait_ns += dt;
                    stop_send_ns -= dt; // inside the send span below, but it is a wait
                }
                if (mgr->load_failed) {
                    GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                }
            }
            if (!sl->seen[e]) {
                mgr->stats.n_miss_cold++;
                sl->n_miss_cold++;
            }
            const uint64_t save = lender ? lender->lend_keep_locked(*sl, v, n_kept) : 0; // before reserve forgets the old book
            mgr->reserve_slot_locked(*sl, e, v);
            // one work item PER SLAB: the 2-3 slabs of an expert are independent reads, and
            // issuing them together is what lifts device queue depth above 1.
            sl->slot_pending[v] = (uint8_t) sl->weights.size();

            for (size_t wi = 0; wi < sl->weights.size(); wi++) {
                llama_moe_stream_work w = { sl, e, v, (int32_t) wi, sl->slot_gen[v] };
                w.lent = lent;
                w.save = save;
                mgr->q_demand.push_back(w);
                mgr->cv_work.notify_one();  // one wakeup per slab
            }
            // (workers woken per slab inside the loop above)
            mgr->stats.n_miss++;
            sl->n_miss++;
            waited = true;
            sl->keep[v] = 1;
            sl->demand_slots.push_back(v);
            if (stops) {
                stop_send_ns += llama_moe_stream_stops::now_ns() - t_send0;
                stop_trips++;
            }
        }
    }

    if (lender) {
        const int64_t t_lend0 = stops ? llama_moe_stream_stops::now_ns() : 0;
        lender->lend_help_locked(lk); // copy the put-back books out while the runners read
        if (stops) {
            stop_send_ns += llama_moe_stream_stops::now_ns() - t_lend0;
        }
    }

    if (waited) {
        const int64_t t_wait0 = stops ? llama_moe_stream_stops::now_ns() : 0;
        const int64_t t0 = ggml_time_us();
        mgr->cv_done.wait(lk, [&]{
            if (mgr->load_failed) {
                return true;
            }
            for (const int32_t s : sl->demand_slots) {
                if (sl->slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        mgr->stats.t_stall_us += ggml_time_us() - t0;
        if (stops) {
            stop_wait_ns += llama_moe_stream_stops::now_ns() - t_wait0;
        }
    }

    for (int64_t i = 0; i < n; i++) {
        const int32_t s = sl->expert_slot.at(ids[i]);
        sl->slot_last_use[s] = ++sl->use_counter;
        out[i] = s;
        if (!sl->slot_la2.empty() && sl->slot_la2[s] == ids[i]) {
            sl->slot_la2[s] = -1; // a two-floors-ahead fetch, read by its floor
            mgr->stats.n_la2_used++;
        }
    }

    if (mgr->gpu_slot) {
        sl->publish_state_locked(*mgr);
        mgr->n_slot_chk++;
    }

    mgr->stats.t_remap_op_us += ggml_time_us() - t_op0;

    if (stops) {
        stops->put(sl->il, t_stop0, llama_moe_stream_stops::now_ns(), (uint32_t) sl->uniq.size(), stop_trips,
                stop_send_ns, stop_wait_ns);
    }
}

void llama_moe_stream::register_hash_router(int32_t il, ggml_tensor * tid2eid, uint32_t n_expert_used) {
    llama_moe_stream_layer * sl = layer(il);
    if (sl == nullptr || tid2eid == nullptr || n_expert_used == 0) {
        return;
    }
    for (const auto & hr : hash_routers) {
        if (hr.sl == sl) {
            return;
        }
    }
    hash_routers.push_back({ sl, tid2eid, {} });
    hash_n_used = n_expert_used;
}

void llama_moe_stream_prefetch_hash(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * mgr = (llama_moe_stream *) userdata;

    if (dst->data != a->data) {
        memcpy(dst->data, a->data, ggml_nbytes(a));
    }

    const int64_t   n_tokens = ggml_nelements(a);
    const int32_t * tokens   = (const int32_t *) a->data;

    // the tid2eid tables are model weights, so read them once - outside the lock, since a
    // cross-backend get can be slow
    for (auto & hr : mgr->hash_routers) {
        if (hr.rows.empty()) {
            hr.rows.resize(ggml_nelements(hr.map));
            ggml_backend_tensor_get(hr.map, hr.rows.data(), 0, ggml_nbytes(hr.map));
        }
    }

    std::unique_lock<std::mutex> lk(mgr->mtx);
    if (mgr->load_failed) {
        return;
    }
    mgr->start_workers_locked();

    std::vector<int32_t> want;
    for (auto & hr : mgr->hash_routers) {
        auto & sl = *hr.sl;

        want.clear();
        for (int64_t t = 0; t < n_tokens; t++) {
            const int64_t tok = tokens[t];
            const int64_t off = tok*mgr->hash_n_used;
            if (tok < 0 || off + mgr->hash_n_used > (int64_t) hr.rows.size()) {
                continue;
            }
            for (uint32_t k = 0; k < mgr->hash_n_used; k++) {
                const int32_t e = hr.rows[off + k];
                if (e >= 0 && (uint32_t) e < sl.n_expert && sl.expert_slot.find(e) == sl.expert_slot.end()) {
                    want.push_back(e);
                }
            }
        }
        std::sort(want.begin(), want.end());
        want.erase(std::unique(want.begin(), want.end()), want.end());

        // A prefill ubatch touches far more experts than the cache holds, and prefetching them all
        // would evict what it just loaded. Leave those to the wave planner, which orders them.
        if (want.size() > sl.n_slots/2) {
            continue;
        }

        for (const int32_t e : want) {
            if (sl.expert_slot.find(e) != sl.expert_slot.end()) {
                continue; // reserved by an earlier token of this same ubatch
            }
            const int32_t v = mgr->pick_victim_locked(sl, nullptr);
            if (v < 0) {
                break; // every slot busy; the layer's own remap will demand-load it
            }
            mgr->reserve_slot_locked(sl, e, v);
            sl.slot_pending[v] = (uint8_t) sl.weights.size();
            for (size_t wi = 0; wi < sl.weights.size(); wi++) {
                mgr->q_spec.push_back({ &sl, e, v, (int32_t) wi, sl.slot_gen[v] });
                mgr->cv_work.notify_one();
            }
            mgr->stats.n_preload_issued++;
        }
    }
}

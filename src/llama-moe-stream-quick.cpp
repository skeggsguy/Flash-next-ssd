// C2, cheaper floor stops: three changes to the remap op (llama-moe-stream-remap.cpp) that make a floor stop
// shorter without changing which book a token reads, where a book sits on the desk, or anything the GPU
// computes. Each is read at load and is on unless its switch is "0" (llama_moe_stream::nolock and friends).
//
// LLAMA_MOE_STREAM_NOLOCK, the quick path: when every book a floor asks for is already on the desk (RESIDENT),
// the remap does its bookkeeping without taking the manager's lock. Why that is safe, and the same:
//   - Which thread writes what. The desk's index (expert_slot, slot_expert) and a slot's LOADING state are
//     written only by reserve_slot_locked, and every caller of that is a custom op on the graph thread (the
//     remap, the lookahead, the hash prefetch, the wave staging, the room's cold fill), except the GPU slot
//     servicer (LLAMA_MOE_STREAM_GPU_SLOT), which runs on Metal's listener thread: the quick path is off
//     whenever that is set. So on the graph thread the index cannot change under the quick path's feet.
//   - The one thing other threads change is a slot turning RESIDENT: a runner's last slab, or a restore from
//     the lent belt. Both publish with a release store and the quick path reads with an acquire load
//     (moe_slot_state_peek, llama-moe-stream-impl.h), so a slot it sees RESIDENT has every byte of its book
//     in place, as the lock would have guaranteed. Nothing turns a RESIDENT slot back to anything else
//     except reserve_slot_locked, i.e. the graph thread itself.
//   - What keeps a slot this call returned from being evicted while the GPU reads it is the same thing that
//     keeps it in the locked path, which also drops the lock before the GPU runs: eviction happens only on
//     the graph thread, and after this floor's remap the graph thread touches this floor's desk again only
//     on the next graph (the floor before it guesses for it), by which time the scheduler has waited for
//     this graph's GPU work to finish. The lock never covered the GPU's read; the order of the ops does.
//   - What it writes without the lock is graph-thread data that no runner touches: the call counter, the
//     route hotness and its decay, the hit counters, the LRU stamps, slot_la2, the borrowing log (all of them
//     are read by other threads only from print_stats, between decodes). Each is updated exactly as the
//     locked path updates it, in the same order, so the desk (hotness, LRU, eviction choices) is identical.
//   - What the locked path does that the quick path skips, and why each is a no-op or moves elsewhere:
//     the stats dump (the quick path steps aside when one is due, so the locked path does it), starting the
//     runners (already started, or the quick path steps aside), the load-failed check (a failed load never
//     turns its slot RESIDENT, so the floor that needs it takes the locked path and aborts there), the slot
//     waits (nothing to wait for), and lend_help, which copies the lent belt's queued keeps out on the graph
//     thread: skipped, each keep is copied by its own runner at its gate (lend_gate_locked), as it already is
//     whenever the runner gets there first, and the next locked call helps with any still queued.
//   The lookahead's issue loop (llama-moe-stream-lookahead.cpp) skips a guess already on the desk or on its
//   way; when every guess is, it would do nothing under the lock, so the lock is skipped there too.
//
// LLAMA_MOE_STREAM_ONE_LOOKUP: the remap looks each book's slot up once a floor (slot_of, filled while it
// classifies the books) instead of once for every token's pick. Same slot, same stamps, same order.
//
// LLAMA_MOE_STREAM_KEEP_AWAKE: while a floor waits on a trip, ping the GPU (ggml-metal-awake.m) every
// keep_awake_us, so an idle GPU does not power down: past ~1.75 ms idle, its next launch costs ~450 us
// instead of ~90. The ping is a 4-byte fill of a buffer of its own on a queue of its own: no model memory.

#include "llama-moe-stream.h"
#include "llama-moe-stream-impl.h"

#include "ggml-backend.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

static bool c2_env_on(const char * name) {
    const char * s = std::getenv(name);
    return s == nullptr || strcmp(s, "0") != 0;
}

void llama_moe_stream::c2_init() {
    // the GPU slot servicer writes the desk's index on Metal's listener thread: neither may assume otherwise
    nolock     = MOE_STREAM_QUICK_BUILT && gpu_slot == 0 && c2_env_on("LLAMA_MOE_STREAM_NOLOCK");
    one_lookup = gpu_slot == 0 && c2_env_on("LLAMA_MOE_STREAM_ONE_LOOKUP");
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_KEEP_AWAKE")) {
        const long long v = std::atoll(s);
        keep_awake_us = strcmp(s, "0") == 0 ? 0 : (v > 1 ? v : 1000);
    }
}

// the remap's last step, shared by both paths: each pick's slot, its LRU stamp, and the two-floors-ahead count
void llama_moe_stream_remap_out(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int64_t n) {
    auto * mgr = sl.mgr;
    mgr->n_one_lookup += mgr->one_lookup;
    for (int64_t i = 0; i < n; i++) {
        const int32_t s = mgr->one_lookup ? sl.slot_of[ids[i]] : sl.expert_slot.at(ids[i]);
        sl.slot_last_use[s] = ++sl.use_counter;
        out[i] = s;
        if (!sl.slot_la2.empty() && sl.slot_la2[s] == ids[i]) {
            sl.slot_la2[s] = -1; // a two-floors-ahead fetch, read by its floor
            mgr->stats.n_la2_used++;
        }
    }
}

// The quick path (see the top of the file). Returns false, having changed nothing the locked path reads,
// when it cannot serve the call: a book not on the desk or still on its way, a stats dump due, or off.
bool llama_moe_stream::remap_quick(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int64_t n,
        int64_t n_tok, int64_t t_op0) {
    if (!nolock || !workers_started) {
        return false;
    }
    if (stats_dump_us > 0 && (stats_t_last_us == 0 || ggml_time_us() - stats_t_last_us >= stats_dump_us)) {
        return false; // maybe_dump_stats_locked has work to do: the locked path does it
    }

    // distinct books touched, in first-use order, exactly as the locked path builds them
    sl.touched.assign(sl.n_expert, 0);
    sl.uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
        if (!sl.touched[e]) {
            sl.touched[e] = 1;
            sl.uniq.push_back(e);
        }
    }
    for (const int32_t e : sl.uniq) {
        const auto it = sl.expert_slot.find(e);
        if (it == sl.expert_slot.end() || moe_slot_state_peek(sl.slot_state[it->second]) != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
            return false;
        }
        sl.slot_of[e] = it->second;
    }

    // every book is on the desk: the locked path's bookkeeping, in its order, without the lock
    stats.n_calls++;
    trace_record_locked(sl.il, (uint32_t) n_tok, n_tok == 1 ? 0 : 1, ids, n); // graph-thread data (see above)
    for (const int32_t e : sl.uniq) {
        sat_inc(sl.route_hotness[e]);
    }
    if (hot_decay_interval > 0 && stats.n_calls % hot_decay_interval == 0) {
        for (auto & sl2 : layers) {
            if (sl2) {
                for (auto & h : sl2->route_hotness) {
                    h >>= 1;
                }
            }
        }
    }
    for (size_t k = 0; k < sl.uniq.size(); k++) {
        stats.n_hit_ready++;
        stats.n_hit++;
        sl.n_hit++;
    }
    llama_moe_stream_remap_out(sl, ids, out, n);
    stats.t_remap_op_us += ggml_time_us() - t_op0;
    n_quick++;
    return true;
}

// The remap's wait for its trips, under mtx. With KEEP_AWAKE, from keep_awake_us after the op began (the GPU
// went idle a little before that) and every keep_awake_us after, the GPU gets a ping until the trips land.
void llama_moe_stream::wait_trips_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int64_t t_op0) {
    auto ready = [&] {
        if (load_failed) {
            return true;
        }
        for (const int32_t s : sl.demand_slots) {
            if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                return false;
            }
        }
        return true;
    };
    if (keep_awake_us > 0 && !keep_awake_looked) {
        keep_awake_looked = true;
        // only a desk the GPU reads: on a CPU desk there is no GPU to keep awake
        for (const auto & c : ctxs) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(c.first);
            if (dev != nullptr && keep_awake_fn == nullptr && (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                    ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU)) {
                keep_awake_fn = (void (*)()) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                        "ggml_backend_metal_keep_awake");
            }
        }
    }
    if (keep_awake_us <= 0 || keep_awake_fn == nullptr) {
        cv_done.wait(lk, ready);
        return;
    }
    int64_t deadline = t_op0 + keep_awake_us;
    while (!ready()) {
        const int64_t left = deadline - ggml_time_us();
        if (left > 0) {
            cv_done.wait_for(lk, std::chrono::microseconds(left), ready);
            continue;
        }
        lk.unlock();
        keep_awake_fn();
        lk.lock();
        n_awake++;
        deadline = ggml_time_us() + keep_awake_us;
    }
}

#include "llama-moe-stream.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <vector>

// make wave w's expert slice (uniq[wave_first[w] .. +wave_count[w])) resident, waiting for its loads,
// and best-effort preload the next wave so its loads overlap this wave's compute. leaves
// sl.demand_slots = this wave's slots and sl.plan_pool = the resident parking pool (>= n_ids slots)
// the emit draws masked pairs from
void llama_moe_stream::stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids) {
    // the slices come from plan_waves_locked rather than being w*plan_capacity: the partition path
    // needs exactly as many waves as the graph built, which may be more than uniq/plan_capacity
    const size_t first = (size_t) w < sl.wave_first.size() ? sl.wave_first[w] : sl.uniq.size();
    const size_t count = (size_t) w < sl.wave_count.size() ? sl.wave_count[w] : 0;

    std::fill(sl.keep.begin(), sl.keep.end(), 0);
    sl.demand_slots.clear();

    // a small final wave has fewer than n_ids own slots; borrow the rest from the previous wave's
    //   pool so every token row has n_ids distinct resident parking slots for its masked pairs
    std::vector<int32_t> borrowed;
    if (count < n_ids) {
        GGML_ASSERT(sl.plan_pool.size() >= n_ids - count);
        for (size_t i = 0; i < n_ids - count; i++) {
            borrowed.push_back(sl.plan_pool[i]);
            sl.keep[sl.plan_pool[i]] = 1; // parking slots must survive this wave's loads
        }
    }

    // protect the next wave's already-resident experts so this wave's victims do not evict them.
    //
    // Bounded by the slack actually available. Upstream relies on an implicit invariant - capacity is
    // (n_slots - n_expert_used)/2, so this wave's cap slots + the next wave's cap slots + n_expert_used
    // parking slots exactly fit - and protecting an unbounded next wave is only safe because of it.
    // Raise the capacity without this bound and the keep-set can cover every slot, at which point
    // pick_victim_locked returns -1 forever and the demand loop below blocks on a cv_done nobody will
    // signal: a silent hang at ~0.1% CPU. Deriving the limit from the slack instead makes any capacity
    // safe, and leaves the /2 case behaving exactly as before (slack there is >= cap).
    const size_t nw     = (size_t) w + 1;
    const size_t nfirst = nw < sl.wave_first.size() ? sl.wave_first[nw] : sl.uniq.size();
    const size_t ncount = nw < sl.wave_count.size() ? sl.wave_count[nw] : 0;

    const size_t reserved  = (size_t) sl.plan_capacity + n_ids + 1; // this wave, parking, one victim
    const size_t max_keep  = sl.n_slots > reserved ? sl.n_slots - reserved : 0;

    size_t n_kept = 0;
    for (size_t i = nfirst; i < nfirst + ncount && n_kept < max_keep; i++) {
        const auto it = sl.expert_slot.find(sl.uniq[i]);
        if (it != sl.expert_slot.end()) {
            sl.keep[it->second] = 1;
            n_kept++;
        }
    }

    // reserve and demand-load this wave's experts (per-expert, same path as the decode remap)
    bool waited = false;
    if (count > 0) {
        stats.n_waves_run++;
        for (size_t i = first; i < first + count; i++) {
            const int32_t e  = sl.uniq[i];
            const auto    it = sl.expert_slot.find(e);
            if (it != sl.expert_slot.end()) {
                // already in the cache (resident, or still loading from the previous wave's preload)
                const int32_t s = it->second;
                if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                    promote_slot_locked(sl, s);
                    waited = true;
                } else {
                    stats.n_preload_ready++; // resident from the previous wave's preload
                }
                stats.n_hit++;
                sl.n_hit++;
                sl.keep[s] = 1;
                sl.demand_slots.push_back(s);
            } else {
                // miss: evict a non-kept slot and queue the load
                int32_t v;
                while ((v = pick_victim_locked(sl, sl.keep.data())) < 0) {
                    cv_done.wait(lk);
                    if (load_failed) {
                        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                    }
                }
                if (!sl.seen[e]) {
                    stats.n_miss_cold++;
                    sl.n_miss_cold++;
                }
                reserve_slot_locked(sl, e, v);
                // one work item PER SLAB: the 2-3 slabs of an expert are independent reads, and
                // issuing them together is what lifts device queue depth above 1.
                sl.slot_pending[v] = (uint8_t) sl.weights.size();

                for (size_t wi = 0; wi < sl.weights.size(); wi++) {

                    q_demand.push_back({ &sl, e, v, (int32_t) wi, sl.slot_gen[v] });
                    cv_work.notify_one();  // one wakeup per slab

                }
                // (workers woken per slab inside the loop above)
                stats.n_miss++;
                sl.n_miss++;
                waited = true;
                sl.keep[v] = 1;
                sl.demand_slots.push_back(v);
            }
        }
    }

    // best-effort preload of the next wave so its loads overlap this wave's compute; never waits,
    //   whatever cannot be reserved now simply becomes the next wave's demand load
    if (std::getenv("LLAMA_MOE_STREAM_NO_PRELOAD") == nullptr) {
        for (size_t i = nfirst; i < nfirst + ncount; i++) {
            const int32_t e = sl.uniq[i];
            if (sl.expert_slot.find(e) != sl.expert_slot.end()) {
                continue;
            }
            const int32_t v = pick_victim_locked(sl, sl.keep.data());
            if (v < 0) {
                continue;
            }
            if (!sl.seen[e]) {
                stats.n_miss_cold++;
                sl.n_miss_cold++;
            }
            reserve_slot_locked(sl, e, v);
            sl.keep[v] = 1;
            // one work item PER SLAB: the 2-3 slabs of an expert are independent reads, and
            // issuing them together is what lifts device queue depth above 1.
            sl.slot_pending[v] = (uint8_t) sl.weights.size();

            for (size_t wi = 0; wi < sl.weights.size(); wi++) {

                q_demand.push_back({ &sl, e, v, (int32_t) wi, sl.slot_gen[v] });
                cv_work.notify_one();  // one wakeup per slab

            }
            // (workers woken per slab inside the loop above)
            stats.n_preload_issued++;
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        cv_done.wait(lk, [&]{
            if (load_failed) {
                return true;
            }
            for (const int32_t s : sl.demand_slots) {
                if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        stats.t_stall_wave_us += ggml_time_us() - t0;
    }

    // parking pool: this wave's own resident slots plus the borrowed ones (all keep-protected;
    //   the next same-layer reservation is ordered after this wave's GEMMs by the graph)
    sl.plan_pool = sl.demand_slots;
    sl.plan_pool.insert(sl.plan_pool.end(), borrowed.begin(), borrowed.end());
    GGML_ASSERT(sl.plan_pool.size() >= n_ids);
}

// write out[i] = the cache slot the GEMM should index for each (token, expert) pair of wave w, one
// token row at a time: pairs whose expert is in this wave get its real slot; the rest park on distinct
// resident pool slots (pool_used prevents a repeat within the row, required by the Metal kernel)
void llama_moe_stream::emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
        int32_t w, uint32_t n_ids, int64_t n_tok) {
    for (int64_t t = 0; t < n_tok; t++) {
        sl.pool_used.clear();

        // pass 1: pairs whose expert belongs to this wave -> that expert's real (resident) slot
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            const int32_t e = ids[i];
            GGML_ASSERT(sl.expert_wave[e] != 0xff);
            if (sl.expert_wave[e] == (uint8_t) w) {
                const int32_t s = sl.expert_slot.at(e);
                GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
                sl.slot_last_use[s] = ++sl.use_counter;
                out[i] = s;
                sl.pool_used.push_back(s);
            }
        }

        // pass 2: the remaining (masked) pairs -> the next pool slot not yet used in this row
        size_t pi = 0;
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            if (sl.expert_wave[ids[i]] == (uint8_t) w) {
                continue;
            }
            while (std::find(sl.pool_used.begin(), sl.pool_used.end(), sl.plan_pool[pi]) != sl.pool_used.end()) {
                pi++;
                GGML_ASSERT(pi < sl.plan_pool.size());
            }
            GGML_ASSERT(sl.slot_state[sl.plan_pool[pi]] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
            out[i] = sl.plan_pool[pi];
            sl.pool_used.push_back(sl.plan_pool[pi]);
            pi++;
        }
    }
}

// Partition path: write the four index rows describing wave w's dense pair list. Every (token, expert)
// pair belongs to exactly one wave, so across the waves each pair is emitted once and the GEMM covers
// it once - as opposed to emit_wave_slots above, where every wave covers every pair and masks the rest.
//
// The list is padded to the static chunk by REPEATING this wave's last pair: the GEMM then recomputes
// that pair and the scatter writes the same value to the same row, so padding needs neither a scratch
// row nor zero-initialised output.
//
// A wave can also own nothing at all - the graph fixes the wave count from the worst case (every expert
// touched), so a ubatch that touches fewer leaves the late waves empty. Such a wave has no pair it may
// legitimately write, and cannot borrow one either: another wave's expert is not necessarily still
// resident by the time this one runs. It therefore computes a throwaway row on a parked slot and
// scatters it to the scratch row past the end of the real pairs, which nothing reads.
void llama_moe_stream::emit_wave_pairs(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
        int32_t w, uint32_t n_ids, int64_t n_pairs, int64_t chunk) {
    const std::vector<int32_t> * pairs = (size_t) w < sl.plan_pair.size() ? &sl.plan_pair[w] : nullptr;
    if (pairs != nullptr && pairs->empty()) {
        pairs = nullptr;
    }
    GGML_ASSERT(pairs == nullptr || (int64_t) pairs->size() <= chunk);

    int32_t * r_tok  = out + LLAMA_MOE_PAIR_TOK *chunk;
    int32_t * r_pair = out + LLAMA_MOE_PAIR_PAIR*chunk;
    int32_t * r_slot = out + LLAMA_MOE_PAIR_SLOT*chunk;
    int32_t * r_exp  = out + LLAMA_MOE_PAIR_EXP *chunk;

    if (pairs == nullptr) {
        GGML_ASSERT(!sl.plan_pool.empty()); // stage_wave_locked leaves >= n_ids resident parking slots
        const int32_t s = sl.plan_pool[0];
        GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
        for (int64_t p = 0; p < chunk; p++) {
            r_tok [p] = 0;
            r_pair[p] = (int32_t) n_pairs; // scratch row
            r_slot[p] = s;
            r_exp [p] = sl.slot_expert[s];
        }
        return;
    }

    for (int64_t p = 0; p < chunk; p++) {
        const int32_t i = (*pairs)[p < (int64_t) pairs->size() ? (size_t) p : pairs->size() - 1];
        const int32_t e = ids[i];
        const int32_t s = sl.expert_slot.at(e);
        GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
        sl.slot_last_use[s] = ++sl.use_counter;

        r_tok [p] = i/(int32_t) n_ids;
        r_pair[p] = i;
        r_slot[p] = s;
        r_exp [p] = e;
    }
}

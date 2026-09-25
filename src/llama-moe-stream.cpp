#include "llama-moe-stream.h"
#include "llama-moe-stream-impl.h"

#include "llama-impl.h"

#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>       // sleep_for, for the gentle opening
#include <chrono>

// true iff all of the given exps tensors are this layer's cache tensors - guards against a second,
// non-streamed expert group on the same layer index (e.g. grovemoe chexps)
bool llama_moe_stream_layer::matches(const ggml_tensor * gate, const ggml_tensor * up,
                                     const ggml_tensor * down, const ggml_tensor * gate_up) const {
    auto is_cache = [this](const ggml_tensor * t) {
        for (const auto & w : weights) {
            if (w.cache == t) {
                return true;
            }
        }
        return false;
    };

    size_t n = 0;
    for (const ggml_tensor * t : { gate, up, down, gate_up }) {
        if (t == nullptr) {
            continue;
        }
        if (!is_cache(t)) {
            return false;
        }
        n++;
    }

    return n > 0 && n == weights.size();
}

// sizes the per-layer table and clamps the I/O thread count; workers are spawned lazily on first use
llama_moe_stream::llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct) : n_slots(n_slots) {
    layers.resize(n_layer);

    this->n_io_threads = n_io_threads <= 0 ? MOE_STREAM_IO_THREADS_DEFAULT : n_io_threads;
    this->n_io_threads = std::min<int32_t>(this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);

    debug         = std::getenv("LLAMA_MOE_STREAM_DEBUG") != nullptr;
    use_direct_io = direct;

    if (const char * s = std::getenv("LLAMA_MOE_STREAM_STATS_MS")) {
        stats_dump_us = std::max<int64_t>(0, std::atoll(s))*1000;
    }

    dump_hotness = std::getenv("LLAMA_MOE_STREAM_HOTNESS") != nullptr;

    if (const char * s = std::getenv("LLAMA_MOE_STREAM_HOTNESS_JSON")) {
        hotness_json = s;
        dump_hotness = true;   // the json path implies the dump
    }

    // the borrowing log: every slip in routing order (format documented on llama_moe_stream::trace_path)
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_TRACE")) {
        trace_path = s;
    }

    // MUST be read here and not in open_files: create_cache_tensor decides whether to allocate the
    // GPU residency table, and it runs while the model's tensors are being loaded - long before
    // open_files. Parsing this late silently disabled the whole feature.
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_GPU_SLOT")) {
        gpu_slot = std::max(0, atoi(s));
    }

    if (const char * s = std::getenv("LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB")) {
        alloc_chunk_bytes = (size_t) std::max<int64_t>(0, std::atoll(s)) << 20;
    }
    alloc_pause_ms = alloc_chunk_bytes > 0 ? 500 : 0;
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_ALLOC_PAUSE_MS")) {
        alloc_pause_ms = std::max(0, atoi(s));
    }

    if (const char * s = std::getenv("LLAMA_MOE_STREAM_LRU")) {
        pure_lru = atoi(s) != 0; // an EMPTY value must mean off, not on
    }

    // enough to keep every reader busy a few slabs deep, not so much that a prefetch is still
    // queued when its layer arrives
    q_spec_max = (size_t) this->n_io_threads * 8;
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_SPEC_MAX")) {
        q_spec_max = (size_t) std::max(0, atoi(s));
    }
}

// stop and join the I/O workers before the cache buffers and files they use are destroyed
llama_moe_stream::~llama_moe_stream() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
        q_demand.clear();
        q_spec.clear();
    }
    cv_work.notify_all();
    for (auto & w : workers) {
        w.join();
    }
    // after the workers are gone, so nothing can append to a closed file
    trace_close();
}

int64_t llama_moe_stream::n_streamed_layers() const {
    int64_t n = 0;
    for (const auto & sl : layers) {
        n += sl != nullptr;
    }
    return n;
}

ggml_tensor * llama_moe_stream::create_cache_tensor(
        int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
        uint16_t file_idx, size_t offs) {
    GGML_ASSERT(il >= 0 && (size_t) il < layers.size());
    GGML_ASSERT(ggml_is_contiguous(meta));
    GGML_ASSERT(meta->ne[2] > 0 && meta->ne[3] == 1);

    const uint32_t n_expert  = meta->ne[2];
    // the borrowing log narrows selected ids to i16; every MoE this engine streams is far below
    // that, and an expert pool that was not would silently corrupt the trace
    GGML_ASSERT(n_expert <= 32767);
    const size_t   nb_expert = ggml_nbytes(meta) / n_expert;
    GGML_ASSERT(nb_expert * n_expert == ggml_nbytes(meta));
    GGML_ASSERT(n_slots > 0 && n_slots < n_expert);

    // the newest context of this buft takes the tensor unless that would push it past the chunk
    // size; an empty context always takes it, so a slab bigger than a chunk still gets a home
    const size_t cache_bytes = nb_expert * n_slots;
    ggml_context * ctx = nullptr;
    for (size_t i = ctxs.size(); i-- > 0;) {
        if (ctxs[i].first == buft) {
            if (alloc_chunk_bytes == 0 || ctx_bytes[i] == 0 || ctx_bytes[i] + cache_bytes <= alloc_chunk_bytes) {
                ctx = ctxs[i].second.get();
                ctx_bytes[i] += cache_bytes;
            }
            break;
        }
    }
    if (ctx == nullptr) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(layers.size()*4 + 1),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context for MoE expert streaming");
        }
        ctxs.emplace_back(buft, ctx);
        ctx_bytes.push_back(cache_bytes);
    }

    ggml_tensor * cache = ggml_new_tensor_3d(ctx, meta->type, meta->ne[0], meta->ne[1], n_slots);
    ggml_format_name(cache, "%s.stream_cache", meta->name);
    GGML_ASSERT(ggml_nbytes(cache) == nb_expert * n_slots);

    auto & sl = layers[il];
    if (!sl) {
        sl = std::make_unique<llama_moe_stream_layer>();
        sl->mgr      = this;
        sl->il       = il;
        sl->n_expert = n_expert;
        sl->n_slots  = n_slots;
        sl->slot_expert  .resize(n_slots, -1);
        sl->slot_state   .resize(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        sl->slot_pending .resize(n_slots, 0);
        sl->slot_gen     .resize(n_slots, 0);
        sl->slot_last_use.resize(n_slots, 0);
        sl->route_hotness.resize(n_expert, 0);
        sl->seen         .resize(n_expert, 0);
        sl->keep         .resize(n_slots, 0);

        if (gpu_slot) {
            if (!ctx_state) {
                ggml_init_params sp = {
                    /*.mem_size   =*/ ggml_tensor_overhead()*(layers.size() + 1),
                    /*.mem_buffer =*/ NULL,
                    /*.no_alloc   =*/ true,
                };
                ctx_state.reset(ggml_init(sp));
                if (!ctx_state) {
                    throw std::runtime_error("failed to create ggml context for MoE slot state");
                }
                buft_state = buft;
            }
            GGML_ASSERT(buft_state == buft);
            const llama_moe_slot_state_layout lay = { (int32_t) n_expert, (int32_t) n_slots };
            sl->state = ggml_new_tensor_1d(ctx_state.get(), GGML_TYPE_I32, lay.size());
            ggml_format_name(sl->state, "blk.%d.stream_slot_state", il);
        }
    }
    GGML_ASSERT(sl->n_expert == n_expert);

    sl->weights.push_back({ cache, file_idx, offs, nb_expert });

    max_nb_expert = std::max(max_nb_expert, nb_expert);

    return cache;
}

void llama_moe_stream::alloc_bufs(bool no_alloc) {
    if (ctx_state && ggml_get_first_tensor(ctx_state.get()) != nullptr) {
        ggml_backend_buffer_t buf;
        if (no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft_state, /*size =*/ 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_state.get()); t != nullptr;
                    t = ggml_get_next_tensor(ctx_state.get(), t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_state.get(), buft_state);
        }
        if (buf == nullptr) {
            throw std::runtime_error("unable to allocate the MoE slot state buffer");
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        buf_state.reset(buf);

        // the buffer starts as whatever was in memory; the kernel would read that as residency
        for (auto & sl : layers) {
            if (!sl || !sl->state) {
                continue;
            }
            auto * host = (int32_t *) ggml_backend_tensor_get_host_ptr(sl->state);
            if (host == nullptr) {
                if (!no_alloc) {
                    throw std::runtime_error("GPU MoE slot resolution requires host-visible state buffers");
                }
                continue;
            }
            const llama_moe_slot_state_layout lay = sl->layout();
            std::fill(host, host + lay.off_tail(), -1);
            std::fill(host + lay.off_tail(), host + lay.size(), 0);
            sl->state_host = host;
            sl->state_init = true;
        }
    }

    size_t n_real = 0;
    for (auto & [buft, ctx_ptr] : ctxs) {
        ggml_context * ctx = ctx_ptr.get();
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        // the gentle opening: let the OS make room for the last chunk before pinning the next
        if (!no_alloc && n_real++ > 0 && alloc_pause_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(alloc_pause_ms));
        }

        ggml_backend_buffer_t buf;
        if (no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error(format("unable to allocate %s buffer for MoE expert streaming", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.emplace_back(buf);

        LLAMA_LOG_INFO("%s: %12s expert cache size = %8.2f MiB (%u slots per layer)\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0, n_slots);
    }
}

// least valuable evictable slot: empty first, then coldest resident (min route hotness, oldest use
// as tiebreak); LOADING and keep slots are never candidates. returns -1 when no candidate exists
int32_t llama_moe_stream::pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const {
    int32_t v = -1;

    for (uint32_t s = 0; s < sl.n_slots; s++) {
        if ((keep && keep[s]) || sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            continue;
        }
        if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
            return s;
        }
        if (v < 0) {
            v = s;
            continue;
        }
        // Pure LRU is what the GPU can own: victim selection collapses to an argmin over
        // slot_last_use, with no per-expert hotness table to keep in sync. Replay of real routing
        // put the cost at +11% misses; measure it here before building the kernel.
        if (pure_lru) {
            if (sl.slot_last_use[s] < sl.slot_last_use[v]) {
                v = s;
            }
            continue;
        }
        const uint32_t hs = sl.route_hotness[sl.slot_expert[s]];
        const uint32_t hv = sl.route_hotness[sl.slot_expert[v]];
        if (hs < hv || (hs == hv && sl.slot_last_use[s] < sl.slot_last_use[v])) {
            v = s;
        }
    }

    return v;
}

// bind expert -> slot and mark it LOADING: evict the slot's prior occupant, bump slot_gen (so any
// in-flight load for the old occupant is recognized as stale), and update the expert_slot index
void llama_moe_stream::reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot) {
    if (sl.slot_expert[slot] >= 0) {
        if (debug) {
            LLAMA_LOG_DEBUG("%s: layer %d: evict expert %d from slot %d\n", __func__, sl.il, sl.slot_expert[slot], slot);
        }
        sl.expert_slot.erase(sl.slot_expert[slot]);
    }

    sl.slot_expert[slot] = expert;
    sl.slot_state[slot]  = LLAMA_MOE_STREAM_SLOT_LOADING;
    sl.slot_gen[slot]++;
    sl.slot_last_use[slot] = ++sl.use_counter;
    sl.expert_slot[expert] = slot;
    sl.seen[expert] = 1;
}

// A speculative hit already has exactly one work item per slab. Move the queued items to the
// demand queue and leave the in-flight items alone. Re-enqueuing every slab and resetting
// slot_pending can publish the slot after duplicate completions while another slab is still unread.
void llama_moe_stream::promote_slot_locked(llama_moe_stream_layer & sl, int32_t slot) {
    bool moved = false;
    for (auto it = q_spec.begin(); it != q_spec.end();) {
        if (it->sl == &sl && it->slot == slot && it->gen == sl.slot_gen[slot]) {
            q_demand.push_back(*it);
            it = q_spec.erase(it);
            moved = true;
        } else {
            ++it;
        }
    }
    if (moved) {
        cv_work.notify_all();
    }
}

size_t llama_moe_stream::size_bufs() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }
    return size;
}

// stable per-wave userdata; grows lazily and records the per-wave expert capacity (set at build)
llama_moe_stream_wave * llama_moe_stream_layer::wave_userdata(int32_t wave, uint32_t capacity) {
    GGML_ASSERT(capacity >= 1 && capacity <= n_slots);
    plan_capacity = capacity;
    while ((size_t) wave >= wave_ud.size()) {
        auto ud = std::make_unique<llama_moe_stream_wave>();
        ud->sl   = this;
        ud->wave = (int32_t) wave_ud.size();
        wave_ud.push_back(std::move(ud));
    }
    return wave_ud[wave].get();
}

bool moe_stream_partition() {
    static const bool v = [] {
        const char * s = std::getenv("LLAMA_MOE_STREAM_PARTITION");
        return s == nullptr || atoi(s) != 0; // on by default; an EMPTY value must mean off, not on
    }();
    return v;
}

// wave 0 of a ubatch: record the distinct touched experts (sl.uniq, first-use order) and split them
// into consecutive groups of plan_capacity, one group per wave (sl.expert_wave[e] = e's wave)
void llama_moe_stream::plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    stats.n_calls++;
    start_workers_locked();
    maybe_dump_stats_locked();

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


    GGML_ASSERT(sl.plan_capacity > 0);
    sl.expert_wave.assign(sl.n_expert, 0xff);
    for (size_t i = 0; i < sl.uniq.size(); i++) {
        GGML_ASSERT(i/sl.plan_capacity < 0xff);
        sl.expert_wave[sl.uniq[i]] = (uint8_t) (i/sl.plan_capacity);
    }
    sl.plan_n_waves   = (uint32_t) ((sl.uniq.size() + sl.plan_capacity - 1)/sl.plan_capacity);
    sl.plan_next_wave = 0;

    // Wave slices. The masked path packs uniq into full groups of plan_capacity, which is what it has
    // always done. The partition path CANNOT: the graph sized its pair chunk from the wave count it
    // built, so the planner has to produce exactly that many waves. A ubatch touching few experts
    // otherwise plans fewer, fatter waves than the graph expects, and each then holds more pairs than
    // the chunk allows - a repetitive prompt planned 2 waves against a graph built for 5, putting
    // n_pairs/2 = 1536 pairs into a chunk of 1127. No balancing can fix a wave-count disagreement.
    const size_t n_uniq = sl.uniq.size();
    if (sl.plan_waves_want > 1 && n_uniq >= sl.plan_waves_want) {
        sl.plan_n_waves = sl.plan_waves_want;
    }
    sl.wave_first.assign(sl.plan_n_waves, 0);
    sl.wave_count.assign(sl.plan_n_waves, 0);
    {
        // spread the experts evenly over exactly plan_n_waves slices, never exceeding plan_capacity
        const size_t base = n_uniq/sl.plan_n_waves;
        const size_t rem  = n_uniq%sl.plan_n_waves;
        size_t at = 0;
        for (uint32_t w = 0; w < sl.plan_n_waves; w++) {
            const size_t cnt = std::min<size_t>(base + (w < rem ? 1 : 0), sl.plan_capacity);
            sl.wave_first[w] = (uint32_t) at;
            sl.wave_count[w] = (uint32_t) cnt;
            at += cnt;
        }
        GGML_ASSERT(at == n_uniq); // every touched expert belongs to exactly one wave
        for (uint32_t w = 0; w < sl.plan_n_waves; w++) {
            for (uint32_t i = 0; i < sl.wave_count[w]; i++) {
                sl.expert_wave[sl.uniq[sl.wave_first[w] + i]] = (uint8_t) w;
            }
        }
    }

    // keyed off the chunk the graph set, NOT off the env var: a ubatch too small to partition falls
    // back to the masked path, and planning pairs for it would check against a stale chunk
    if (sl.plan_pair_chunk > 0) {
        plan_pairs_locked(sl, ids, n);
    }
}

// Pair partitioning: give every (token, expert) pair to exactly one wave, so the expert GEMMs cover
// each pair once instead of once per wave. Called after the expert->wave split above, which it
// REORDERS: stage_wave_locked stages uniq[w*cap .. +cap], so keeping the waves as contiguous runs of
// uniq means the staging and preload paths need no changes at all - only the order within uniq moves.
//
// Why reorder: the graph fixes chunk_pairs before the router has run, so an unbalanced split (some
// waves owning far more pairs than others, since routing is skewed) would force chunk_pairs up to the
// worst case and give back the saving. Balancing the pair count across waves bounds it near the mean.
void llama_moe_stream::plan_pairs_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    const uint32_t n_waves = sl.plan_n_waves;
    if (n_waves <= 1) {
        sl.plan_pair.clear();
        return;
    }

    // pairs per expert - the weight each expert contributes to its wave
    sl.pair_count.assign(sl.n_expert, 0);
    for (int64_t i = 0; i < n; i++) {
        sl.pair_count[ids[i]]++;
    }

    // group sizes must match what stage_wave_locked will slice out of uniq: cap for every wave but
    // the last, which takes the remainder
    std::vector<uint32_t> room(sl.wave_count);

    // heaviest expert first into the currently lightest wave that still has room (LPT scheduling):
    // bounds the max wave load near the mean for skewed routing, which is what caps chunk_pairs
    std::vector<int32_t> order(sl.uniq);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return sl.pair_count[a] > sl.pair_count[b];
    });

    // SNAKE order, not greedy-lightest. Each wave must end up with a fixed NUMBER of experts (its
    // staging slice), and that cardinality constraint fights load balance: sending each expert to the
    // lightest wave fills the light waves' expert slots first, after which every remaining expert is
    // forced into whatever wave still has room regardless of its load. With 256 experts over 5 waves
    // of 52 there is almost no spare capacity, so that is forced rather than unlucky - it piled 1667
    // pairs into one wave against a mean of 615, all of them individually small.
    //
    // Sweeping back and forth (rank 0->wave 0, 1->1, .. k-1->k-1, k->k-1, k+1->k-2, ..) pairs each
    // heavy expert with a light one and fills every wave to exactly its room by construction.
    std::vector<std::vector<int32_t>> group(n_waves);
    std::vector<int64_t>              load (n_waves, 0);
    {
        std::vector<uint32_t> left(room);
        std::vector<uint32_t> seq;
        seq.reserve(order.size());

        uint32_t w = 0;
        int      dir = 1;
        while (seq.size() < order.size()) {
            if (left[w] > 0) {
                seq.push_back(w);
                left[w]--;
            }
            if (dir > 0) {
                if (w + 1 < n_waves) { w++; } else { dir = -1; }
            } else {
                if (w > 0) { w--; } else { dir = 1; }
            }
        }
        for (size_t i = 0; i < order.size(); i++) {
            group[seq[i]].push_back(order[i]);
            load [seq[i]] += sl.pair_count[order[i]];
        }
    }

    // Repair pass: swapping a heavy expert out of the worst wave for a lighter one from the best wave
    // preserves both cardinalities, so it can only help. Only runs when a wave is actually over the
    // chunk, which snake ordering already makes rare.
    for (int iter = 0; iter < 64; iter++) {
        uint32_t hi = 0, lo = 0;
        for (uint32_t w = 1; w < n_waves; w++) {
            if (load[w] > load[hi]) { hi = w; }
            if (load[w] < load[lo]) { lo = w; }
        }
        if (load[hi] <= (int64_t) sl.plan_pair_chunk || hi == lo) {
            break;
        }

        // best swap = the one that shrinks the gap most without inverting it
        const int64_t gap = load[hi] - load[lo];
        int64_t best_d = 0;
        size_t  bi = 0, bj = 0;
        for (size_t i = 0; i < group[hi].size(); i++) {
            for (size_t j = 0; j < group[lo].size(); j++) {
                const int64_t d = sl.pair_count[group[hi][i]] - sl.pair_count[group[lo][j]];
                if (d > best_d && 2*d <= gap + best_d) { best_d = d; bi = i; bj = j; }
            }
        }
        if (best_d <= 0) {
            break; // nothing left to trade
        }
        std::swap(group[hi][bi], group[lo][bj]);
        load[hi] -= best_d;
        load[lo] += best_d;
    }

    // rewrite uniq in wave order and record each wave's slice, so the stager needs no change
    sl.uniq.clear();
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.wave_first[w] = (uint32_t) sl.uniq.size();
        sl.wave_count[w] = (uint32_t) group[w].size();
        for (const int32_t e : group[w]) {
            sl.expert_wave[e] = (uint8_t) w;
            sl.uniq.push_back(e);
        }
    }

    // how far the worst wave sits above the mean - what the graph's chunk slack has to cover
    const int64_t mean = (int64_t) n/n_waves;
    for (uint32_t w = 0; w < n_waves; w++) {
        stats.pair_over_max = std::max(stats.pair_over_max, mean > 0 ? (load[w] - mean)*100/mean : 0);
    }

    // CHUNK UTILISATION: the worst wave load as a percentage of the bound that ABORTS when exceeded.
    // This is the number that predicts a crash; imbalance-over-mean does not, because the chunk is
    // floored at n_tokens and so is not a fixed multiple of the mean. 100% means the server died.
    int64_t worst = 0;
    for (uint32_t w = 0; w < n_waves; w++) worst = std::max(worst, load[w]);
    if (sl.plan_pair_chunk > 0) {
        stats.chunk_util_max = std::max(stats.chunk_util_max, worst*100/(int64_t) sl.plan_pair_chunk);
    }

    // flat pair indices (t*n_ids + k) owned by each wave
    sl.plan_pair.assign(n_waves, {});
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.plan_pair[w].reserve((size_t) load[w]);
    }
    for (int64_t i = 0; i < n; i++) {
        sl.plan_pair[sl.expert_wave[ids[i]]].push_back((int32_t) i);
    }

    // ---------------------------------------------------------------------------------------
    // SPLIT PASS: move surplus pairs off any over-full wave, staging that expert in the receiving
    // wave as well.
    //
    // Balancing alone cannot fix this, because all of an expert's pairs go wherever it is staged. A
    // single expert can hold n_tokens pairs (one per token), so even with the chunk floored at
    // n_tokens, that expert PLUS any other in the same wave overflows. Measured: an 800-word
    // repetitive prompt gave "wave 0 holds 1009 pairs but chunk is 1000" - overflowing by the size
    // of the second expert. Repetitive input reaches this trivially, so it is not a corner case.
    //
    // A distribution always exists: total capacity is n_waves*chunk, which exceeds n_pairs by the
    // slack. Only indivisibility stood in the way, and an expert may be staged in more than one wave
    // - it costs a slot there, and a second staging of a resident expert is a cache hit, not I/O.
    // ---------------------------------------------------------------------------------------------
    const size_t chunk = sl.plan_pair_chunk;
    for (uint32_t w = 0; w < n_waves && chunk > 0; w++) {
        while (sl.plan_pair[w].size() > chunk) {
            const size_t surplus = sl.plan_pair[w].size() - chunk;

            // the expert contributing most to this wave is the one worth moving
            std::unordered_map<int32_t, size_t> cnt;
            for (const int32_t idx : sl.plan_pair[w]) cnt[ids[idx]]++;
            int32_t hot = -1; size_t hot_n = 0;
            for (const auto & kv : cnt) if (kv.second > hot_n) { hot = kv.first; hot_n = kv.second; }
            if (hot < 0) break;

            // A receiving wave always needs pair room. It needs a free expert slot only if it does
            // not already stage `hot` - if it does, the pairs land on a slot that wave has already
            // reserved, so the move is free. Preferring those receivers is what the earlier version
            // had backwards: it SKIPPED them, spending a slot where none was needed.
            int32_t dst = -1; size_t room = 0; bool dst_has = false;
            for (uint32_t v = 0; v < n_waves; v++) {
                if (v == w || sl.plan_pair[v].size() >= chunk) continue;

                const bool has = std::find(group[v].begin(), group[v].end(), hot) != group[v].end();
                if (!has && group[v].size() >= sl.plan_capacity) continue;

                // a free move beats a bigger one that costs a slot
                const size_t r = chunk - sl.plan_pair[v].size();
                if (has != dst_has ? has : r > room) { room = r; dst = (int32_t) v; dst_has = has; }
            }
            if (dst < 0) {
                break; // no receiver; the check below reports it rather than truncating silently
            }

            const size_t move = std::min({surplus, room, hot_n});
            std::vector<int32_t> keep; keep.reserve(sl.plan_pair[w].size() - move);
            size_t moved = 0;
            for (const int32_t idx : sl.plan_pair[w]) {
                if (moved < move && ids[idx] == hot) { sl.plan_pair[dst].push_back(idx); moved++; }
                else                                  { keep.push_back(idx); }
            }
            sl.plan_pair[w].swap(keep);
            if (!dst_has) {
                group[dst].push_back(hot);   // stage it in the receiver too
            }
            stats.n_pair_splits++;
            if (moved == 0) break;
        }
    }

    // uniq must reflect the split staging, so rebuild the slices from the (possibly grown) groups.
    // expert_wave is left as the last writer sets it: the partition path keys off plan_pair, not
    // expert_wave, and the masked path never runs when partitioning is active.
    sl.uniq.clear();
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.wave_first[w] = (uint32_t) sl.uniq.size();
        sl.wave_count[w] = (uint32_t) group[w].size();
        for (const int32_t e : group[w]) sl.uniq.push_back(e);
    }

    // still loud if a wave cannot be represented - but this should now be unreachable
    for (uint32_t w = 0; w < n_waves; w++) {
        if (sl.plan_pair[w].size() > sl.plan_pair_chunk) {
            // Report which constraint bound, because the two have different fixes: no pair room
            // means the chunk itself is too small (raise LLAMA_MOE_STREAM_PAIR_SLACK), no slot room
            // means the wave count is too low for the expert count (the cap-1 sizing in
            // llama-graph.cpp did not apply, e.g. the pairs-per-wave floor blocked it).
            size_t free_pairs = 0, free_slots = 0;
            for (uint32_t v = 0; v < n_waves; v++) {
                free_pairs += sl.plan_pair[v].size()   < chunk            ? 1 : 0;
                free_slots += group[v].size() < (size_t) sl.plan_capacity ? 1 : 0;
            }
            GGML_ABORT("MoE expert streaming: wave %u holds %zu pairs but chunk is %u after splitting "
                       "(%u waves, %zu with pair room, %zu with a free expert slot, capacity %u, "
                       "%zu experts staged); report it with the prompt that caused it",
                    w, sl.plan_pair[w].size(), sl.plan_pair_chunk,
                    n_waves, free_pairs, free_slots, sl.plan_capacity, sl.uniq.size());
        }
    }
}

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

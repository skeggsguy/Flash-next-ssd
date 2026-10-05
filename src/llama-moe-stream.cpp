#include "llama-moe-stream.h"
#include "llama-moe-stream-impl.h"
#include "llama-moe-room.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
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

    c2_init(); // the C2 switches (llama-moe-stream-quick.cpp)
}

// stop and join the I/O workers before the cache buffers and files they use are destroyed
llama_moe_stream::~llama_moe_stream() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
        q_demand.clear();
        q_room.clear();
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
        sl->slot_of      .resize(n_expert, -1);

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

    // the reading room's belt, one more buffer after the desk's (llama-moe-room.cpp)
    if (room_layout.on) {
        room = std::make_unique<llama_moe_room>(*this, room_layout);
        room->alloc(no_alloc);
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

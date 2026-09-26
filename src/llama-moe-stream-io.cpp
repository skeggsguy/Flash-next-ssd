#include "llama-moe-stream.h"
#include "llama-moe-stream-impl.h"
#include "llama-moe-room.h"

#include "ggml-backend.h"

#include <cerrno>
#include <mutex>

#ifdef _WIN32
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// page-aligned allocation, required both for O_DIRECT reads and for Metal private-buffer uploads
void * moe_aligned_alloc(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, MOE_STREAM_DIRECT_ALIGN);
#else
    void * p = nullptr;
    if (posix_memalign(&p, MOE_STREAM_DIRECT_ALIGN, n) != 0) {
        p = nullptr;
    }
    return p;
#endif
}

void moe_aligned_free(void * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// read len bytes at file offset offs into staging (thread-safe positional read); staging must have
// room for len (+ 2*MOE_STREAM_DIRECT_ALIGN when direct). returns a pointer to the len bytes
// within staging, or nullptr on failure
const uint8_t * llama_moe_stream_pread(llama_file & file, uint8_t * staging, size_t len, size_t offs, bool direct) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    // no positional read primitive; serialize the seek+read pairs
    static std::mutex io_mtx;
    std::lock_guard<std::mutex> lock(io_mtx);
    try {
        file.seek(offs, SEEK_SET);
        file.read_raw(staging, len);
        return staging;
    } catch (...) {
        return nullptr;
    }
#else
    const int fd = file.file_id();

    if (direct) {
        // O_DIRECT requires the offset, length, and buffer all block-aligned
        const size_t a     = MOE_STREAM_DIRECT_ALIGN;
        const size_t aoffs = offs & ~(a - 1);
        const size_t head  = offs - aoffs;
        const size_t total = ((head + len + a - 1)/a)*a;
        ssize_t r;
        do {
            r = pread(fd, staging, total, aoffs);
        } while (r < 0 && errno == EINTR);
        if (r < 0 || (size_t) r < head + len) {
            return nullptr;
        }
        return staging + head;
    }

    uint8_t * p    = staging;
    size_t    left = len;
    while (left > 0) {
        const ssize_t r = pread(fd, p, left, offs);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return nullptr;
        }
        if (r == 0) {
            return nullptr; // unexpected EOF
        }
        p    += r;
        offs += (size_t) r;
        left -= (size_t) r;
    }
    return staging;
#endif
}

// spawn the I/O thread pool on first use (from the remap callback, under mtx)
void llama_moe_stream::start_workers_locked() {
    if (workers_started) {
        return;
    }
    workers_started = true;
    workers.reserve(n_io_threads);
    for (int32_t i = 0; i < n_io_threads; i++) {
        workers.emplace_back([this]() { worker_loop(); });
    }
}

// I/O worker: pops a reserved load, reads its expert slab(s) from the GGUF file into the cache
// slot, and marks the slot RESIDENT (or flags load_failed); stale/duplicate items are skipped
void llama_moe_stream::worker_loop() {
    // page-aligned staging (Metal private buffers require page-aligned source + page-multiple
    // length; O_DIRECT needs the extra head/tail slack for its aligned reads)
    uint8_t * staging = (uint8_t *) moe_aligned_alloc(max_nb_expert + 2*MOE_STREAM_DIRECT_ALIGN);
    GGML_ASSERT(staging != nullptr);

    std::unique_lock<std::mutex> lk(mtx);
    while (true) {
        cv_work.wait(lk, [&]{ return shutting_down || !q_demand.empty() || !q_room.empty() || !q_spec.empty(); });
        if (shutting_down) {
            break;
        }

        // demand first, always: a layer is blocked on those, nothing is blocked on a prefetch. The reading
        // room's sweep comes between: the GPU will need those books, but writing's trips go first.
        llama_moe_stream_work w;
        if (!q_demand.empty()) {
            w = q_demand.front();
            q_demand.pop_front();
        } else if (!q_room.empty()) {
            w = q_room.front();
            q_room.pop_front();
        } else {
            w = q_spec.front();
            q_spec.pop_front();
        }

        auto & sl = *w.sl;
        // a belt read: stale once its part is gone or cancelled (checked again when it lands)
        const bool ring = w.slot < 0;
        if (ring && !room->worker_begin_locked(w)) {
            continue;
        }
        // no per-slot exclusion: several workers legitimately hold different slabs of the SAME slot
        // at once, which is the entire point. Staleness is still checked per slot.
        if (!ring && (w.gen != sl.slot_gen[w.slot] ||
            sl.slot_state[w.slot] != LLAMA_MOE_STREAM_SLOT_LOADING ||
            sl.slot_expert[w.slot] != w.expert ||
            w.widx < 0 || (size_t) w.widx >= sl.weights.size())) {
            if (w.lent) {
                room->lend.unpin(w.lent); // a restore that will not happen reads nothing
            }
            continue; // stale item
        }
        // a restore copies the book back from the lent belt (llama-moe-room-lend-ops.cpp): no drive, no drive stats
        if (w.lent) {
            room->lend_restore_locked(lk, w);
            continue;
        }

        // two runners: low expert ids from the model's own shards, high ids from the alt copy.
        // The offset is the same in both, because the alt shards are byte-identical copies.
        const bool alt = use_alt(w.expert, sl.n_expert);

        // Read into the cache slot itself when the backend hands out a host pointer - on unified
        // memory the slot IS host memory, so staging then uploading is a pure extra copy of the
        // whole slab. The direct-io path keeps staging: it needs the head/tail slack for its
        // block-aligned reads, which would scribble outside the slot.
        const auto & wt = sl.weights[w.widx];
        uint8_t * dst = nullptr;
        if (!use_direct_io && !no_zerocopy) {
            if (ring) {
                dst = room->host ? room->host + w.ring_offs : nullptr;
            } else {
                auto * host = (uint8_t *) ggml_backend_tensor_get_host_ptr(wt.cache);
                dst = host ? host + (size_t) w.slot*wt.nb_expert : nullptr;
            }
        }
        // the slot's old book is being kept on the lent belt: a read straight into the slot waits until
        // this slab is copied out (a staged read waits just before its upload, below)
        if (w.save && dst != nullptr) {
            room->lend_gate_locked(lk, w);
        }
        busy_begin_locked(alt);

        lk.unlock();

        // Timed to separate the two halves of a miss. A miss currently reads its 2-3 weight slabs
        // SEQUENTIALLY on one thread, so the device sees queue depth 1 even though the reads are
        // independent - and it idles during each upload. Whether that is worth fixing depends on the
        // read:upload split, which is what these two counters measure.
        // exactly one slab, so N workers can be in flight on the same expert. Measured before this
        // change: read 1.00 ms/slab, upload 0.065 ms/slab, i.e. 94% of a miss is the read, and the
        // three reads of an expert were strictly serialised at the device's QD1 rate (~2.9 GB/s
        // against 7.3 GB/s at QD8). Issuing them together is what raises the depth.
        const int64_t t0 = ggml_time_us();
        const uint8_t * data = llama_moe_stream_pread(*(alt ? files_alt : files)[wt.file_idx], dst ? dst : staging,
                wt.nb_expert, wt.offs + (size_t) w.expert*wt.nb_expert, use_direct_io);
        const int64_t t1 = ggml_time_us();
        const bool ok = data != nullptr;
        if (ok && dst == nullptr) {
            if (ring) {
                ggml_backend_tensor_set(room->whole, data, w.ring_offs, wt.nb_expert);
            } else {
                if (w.save) {
                    lk.lock();
                    room->lend_gate_locked(lk, w); // the slot kept its old bytes until here
                    lk.unlock();
                }
                ggml_backend_tensor_set(wt.cache, data, (size_t) w.slot*wt.nb_expert, wt.nb_expert);
            }
        }
        const int64_t t2 = ggml_time_us();

        lk.lock();
        busy_end_locked(alt);

        stats.t_io_read_us   += t1 - t0;
        stats.t_io_upload_us += ok ? t2 - t1 : 0;
        stats.n_slabs_read   += 1;
        if (ok) {
            // which runner fetched it. With one runner every byte is n_bytes_file.
            (alt ? stats.n_bytes_alt : stats.n_bytes_file) += (int64_t) wt.nb_expert;
        }

        {
            int b = 0;
            while (b < MOE_STREAM_READ_BUCKETS - 1 && (t1 - t0) >= MOE_STREAM_READ_BUCKET_US[b]) {
                b++;
            }
            stats.n_read_bucket[b]++;
        }

        if (ring) {
            // never READY on a failed read; the room's ops abort on load_failed
            load_failed = load_failed || !ok;
            room->worker_end_locked(w, ok);
        } else if (!ok) {
            load_failed = true;
            sl.slot_pending[w.slot] = 0;
        } else if (w.gen == sl.slot_gen[w.slot] && sl.slot_pending[w.slot] > 0) {
            // the LAST slab to land publishes the slot; until then it stays LOADING, so no consumer
            // can observe a half-filled expert
            if (--sl.slot_pending[w.slot] == 0) {
                sl.slot_state[w.slot] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
            }
        }
        cv_done.notify_all();
    }
    lk.unlock();

    moe_aligned_free(staging);
}

// Each runner's busy time: from its first read in flight to its last, however many overlap. The reads'
// own durations overlap across threads and cannot say how busy a drive is; this can.
void llama_moe_stream::busy_begin_locked(bool alt) {
    if (busy_n[alt]++ == 0) {
        busy_t0[alt] = ggml_time_us();
    }
}

void llama_moe_stream::busy_end_locked(bool alt) {
    if (--busy_n[alt] == 0) {
        (alt ? stats.t_busy_alt_us : stats.t_busy_file_us) += ggml_time_us() - busy_t0[alt];
    }
}

void llama_moe_stream::busy_flush_locked(int64_t now) {
    for (int alt = 0; alt < 2; alt++) {
        if (busy_n[alt] > 0) {
            (alt ? stats.t_busy_alt_us : stats.t_busy_file_us) += now - busy_t0[alt];
            busy_t0[alt] = now;
        }
    }
}

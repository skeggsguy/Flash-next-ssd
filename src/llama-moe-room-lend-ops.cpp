// The lent belt's glue (llama-moe-room.h; its bookkeeping is llama-moe-room-lend.h). Between read-ins the
// reading room's belt sits idle, so it is lent to the desk while the library writes: the writing remap,
// and the lookahead's guesses for the next floor that it queues, keep a copy of each book they put back,
// and a later trip for one of those books (a miss, or a guess) copies it back into its new desk slot (a
// restore, ~0.1 ms) instead of a trip to the stacks (~0.8 ms). Waves, cold fills and the hash prefetch
// neither keep nor restore: a short read-in's waves would flush the belt with cold books. The guesses are
// in because the study writes with the lookahead on, where they are two thirds of what the desk puts back
// (RR-room: 77 guesses against 36 misses a written token).
//
// When the belt is lent and taken back:
//   borrow     at the first book the writing remap puts back once every room floor of the last read-in
//              has run (op_floor == order.size()). A remap op is a CPU op, so the scheduler has drained the
//              GPU before it (docs/wizard-reading-room.md section 2) and no GEMM still reads a part: every
//              part is handed back at once. The belt is lent only once no cancelled read is still landing.
//   take back  at the top of begin_ubatch_locked, in the first room floor's desk op (drained too). Every
//              keep and restore a remap op queued for itself is finished by then: the op waits for its
//              slots to turn RESIDENT, a slot turns RESIDENT only after each of its slabs landed, each
//              landing waits for that slab's keep (the gate), and each restore unpins its copy before its
//              slab counts down. A guess is not waited for, so its keep or restore may still be queued (its
//              copy goes with the belt: a queued restore turns into a plain read, a queued keep's slab was
//              never touched) or in flight (a memcpy outside the lock: the take-back waits for it, bounded
//              by one slab copy each). The gate then finds no copy and lets the write through.
//
// Why the words cannot change:
//   - the desk's choices are the remap's and the lookahead's own (pick_victim, reserve, hotness, LRU);
//     lending only changes where a load's bytes come from, so the id planes are the same with lending off
//     (the desk too, except that a restore lands sooner than a read, which the lookahead's timing already
//     makes run-dependent);
//   - a keep copies the old book: its slot's new slab is written only by a work item carrying `save`, and
//     the gate holds that write until the old slab is copied out (on the direct path the read lands in
//     staging and the gate sits before tensor_set; on the zero-copy path the read lands in the slot
//     itself, so the gate sits before the read); a LOADING slot is never reserved again, so nothing else
//     writes it;
//   - only a complete copy is found, and a found copy is pinned until its restore is copied, so no keep is
//     ever placed over it and no take-back drops it;
//   - a restore follows the trip protocol: slot_pending counts its slabs down and the slot turns RESIDENT
//     on the last, which the remap or the desk op waits for as for any trip;
//   - the GPU never reads a lent copy: only the room's views read the belt, in a room ubatch, and the belt
//     is taken back before the first of them.
// Copies run outside the lock; every state change runs under mgr.mtx. One graph thread drives a manager
// (the remap's per-floor scratch already assumes it); lend_queued is that thread's list.

#include "llama-moe-room.h"

#include "llama-impl.h"
#include "llama-moe-stream.h"
#include "llama-moe-stream-impl.h"

#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

// Where a tensor's bytes sit in the CPU's address space, or null. A Metal shared buffer hands out its host
// pointer; a CPU buffer is host memory itself, and says so through is_host rather than a host pointer.
static uint8_t * host_of(const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr) {
        return nullptr;
    }
    return (uint8_t *) (ggml_backend_buffer_is_host(t->buffer) ? t->data : ggml_backend_tensor_get_host_ptr(t));
}

// slab `widx` of desk slot `slot` (lend_init checked every desk slab is in host memory)
static uint8_t * desk_slab(const llama_moe_stream_layer & sl, int32_t slot, int32_t widx) {
    const llama_moe_stream_weight & w = sl.weights[widx];
    return host_of(w.cache) + (size_t) slot*w.nb_expert;
}

// the lent belt's bytes (lend_init checked the belt is in host memory)
static uint8_t * belt_at(const llama_moe_room & room, size_t offs) {
    return host_of(room.whole) + offs;
}

void llama_moe_room::lend_init(bool no_alloc) {
    const char * env = std::getenv("LLAMA_MOE_ROOM_LEND");
    const char * off = nullptr;
    size_t sum_strides = 0, n_floors = 0;
    for (const auto & F : floors) {
        if (F.sl == nullptr) {
            continue;
        }
        sum_strides += F.stride;
        n_floors++;
        for (const auto & w : F.sl->weights) {
            if (!no_alloc && host_of(w.cache) == nullptr) {
                off = "the desk is not in host memory, where books are copied";
            }
        }
    }
    if (env != nullptr && atoi(env) == 0) {
        off = "LLAMA_MOE_ROOM_LEND=0";
    } else if (mgr.gpu_slot >= 3) {
        off = "LLAMA_MOE_STREAM_GPU_SLOT=3 moves the desk's bookkeeping to the GPU, where nothing keeps books";
    } else if (!no_alloc && host_of(whole) == nullptr) {
        off = "the belt is not in host memory, where books are copied";
    }
    lend     = llama_moe_lend(lay.room_bytes);
    lend_cap = llama_moe_lend_cap(mgr.n_expert_used, lay.room_bytes, sum_strides);
    lend_on  = off == nullptr && !no_alloc && n_floors > 0;
    if (no_alloc) {
        return;
    }
    // WARN, like the room's startup line, so llama-server shows it; never the words "reading room:", which
    // the study's runner takes as the room's own line
    if (lend_on) {
        LLAMA_LOG_WARN("load_tensors: lent belt: while writing, the belt keeps up to ~%u books the desk puts back; "
                       "LLAMA_MOE_ROOM_LEND=0 turns it off\n", llama_moe_lend_capacity(lay.room_bytes, sum_strides, n_floors));
    } else {
        LLAMA_LOG_WARN("load_tensors: lent belt: off; %s\n", off ? off : "no floor streams through the room");
    }
}

void llama_moe_room::lend_take_back_locked(std::unique_lock<std::mutex> & lk) {
    if (!lent_out) {
        return;
    }
    // Guesses still queued lose their copies with the belt: a queued restore becomes a plain read (its
    // pins let go); a queued keep is forgotten (its slab is PENDING: nothing has touched the bytes), and
    // its worker's gate finds no copy. Only a queued item's own worker writes its (slot, slab), so no
    // copy-out of it can be running.
    for (auto * q : { &mgr.q_spec, &mgr.q_demand }) {
        for (auto & w : *q) {
            if (w.lent) {
                lend.unpin(w.lent);
                w.lent = 0;
                lstats.n_guess_unlent++;
            }
        }
    }
    // a guess in flight: a copy-out (RUNNING) or a restore (a pin) is a memcpy outside the lock; wait
    mgr.cv_done.wait(lk, [&] { return !lend.inflight(); });

    // POISON: a copy the room's parts now overwrite must never be found again; 0xFF makes one that is
    // found anyway fail the file check on its restore, and changes the words
    for (const auto & c : lend.ring) {
        lstats.n_kept_forgotten += !c.complete();
        if (poison) {
            memset(belt_at(*this, c.offs), 0xFF, c.bytes);
        }
    }
    lstats.n_cleared += (int64_t) lend.clear();
    lstats.n_taken_back++;
    lend_queued.clear();
    lent_out = false;
}

// Lends the belt to the desk, or says it cannot yet (see the top of the file for why this moment is safe).
static bool lend_borrow_locked(llama_moe_room & room) {
    if (room.op_floor < room.order.size()) {
        return false; // a read-in's floors still to run: their parts are the GPU's
    }
    room.pump_floor = room.order.size(); // nothing more goes on the belt until the next read-in begins
    room.in_use_seq = room.belt.next_seq - 1;
    room.release_locked();
    room.belt.cancel_all();
    room.belt.reclaim();
    if (!room.belt.parts.empty()) {
        return false; // a cancelled read is still landing on it
    }
    room.lent_out = true;
    return true;
}

uint64_t llama_moe_room::lend_find_locked(const llama_moe_stream_layer & sl, int32_t book, bool guess) {
    (guess ? lstats.n_guess : lstats.n_miss)++;
    const llama_moe_lend_copy * c = lent_out ? lend.find(sl.il, book) : nullptr;
    if (c == nullptr) {
        return 0;
    }
    // one pin per slab its restore copies, each let go as that slab lands: until then no keep drops it
    for (size_t wi = 0; wi < sl.weights.size(); wi++) {
        lend.pin(c->seq);
    }
    (guess ? lstats.n_guess_lent : lstats.n_lent)++;
    return c->seq;
}

uint64_t llama_moe_room::lend_keep_locked(const llama_moe_stream_layer & sl, int32_t slot, uint32_t & n_kept) {
    if (sl.slot_state[slot] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
        return 0; // an EMPTY slot puts nothing back (pick_victim never offers a LOADING one)
    }
    if (!lent_out && !lend_borrow_locked(*this)) {
        return 0;
    }
    if (n_kept >= lend_cap) {
        lstats.n_not_kept++;
        return 0;
    }
    const llama_moe_room_floor & F = floors[sl.il];
    llama_moe_lend_keep why;
    const llama_moe_lend_copy * c = lend.keep(sl.il, sl.slot_expert[slot], slot, F.stride, (int32_t) sl.weights.size(), why);
    if (c == nullptr) {
        lstats.n_not_kept      += why == LLAMA_MOE_LEND_BLOCKED; // HELD: its copy is already on the belt
        lstats.n_not_kept_busy += why == LLAMA_MOE_LEND_BLOCKED;
        return 0;
    }
    n_kept++;
    lstats.n_kept++;
    lend_queued.push_back(c->seq);
    return c->seq;
}

// copies slab `slab` of keep `seq` out of its desk slot onto the belt; the caller claimed it, so the copy
// is busy and stays where it is while the lock is dropped
static void lend_copy_out(llama_moe_room & room, std::unique_lock<std::mutex> & lk, uint64_t seq, int32_t slab) {
    const llama_moe_lend_copy *  c  = room.lend.at(seq);
    const llama_moe_room_floor & F  = room.floors[c->il];
    const size_t                 nb = F.sl->weights[slab].nb_expert;
    uint8_t *       dst = belt_at(room, c->offs + F.w_offs[slab]);
    const uint8_t * src = desk_slab(*F.sl, c->slot, slab);
    lk.unlock();
    memcpy(dst, src, nb);
    lk.lock();
    room.lend.finish(seq, slab);
    room.lstats.n_bytes_in += (int64_t) nb;
    room.mgr.cv_done.notify_all(); // a worker may wait at its gate
}

void llama_moe_room::lend_help_locked(std::unique_lock<std::mutex> & lk) {
    // the op thread would only wait for the trips now: copy out every slab no worker has claimed yet
    for (size_t i = 0; i < lend_queued.size(); i++) {
        const uint64_t seq = lend_queued[i];
        for (int32_t s = 0; lend.at(seq) != nullptr && s < lend.at(seq)->n_slabs; s++) {
            if (lend.claim(seq, s)) {
                lend_copy_out(*this, lk, seq, s);
            }
        }
    }
    lend_queued.clear();
}

void llama_moe_room::lend_gate_locked(std::unique_lock<std::mutex> & lk, const llama_moe_stream_work & w) {
    for (;;) {
        const llama_moe_lend_copy * c = lend.at(w.save);
        if (c == nullptr) {
            return; // the room took the belt back while this guess was queued or in flight: nothing to keep
        }
        GGML_ASSERT(c->il == w.sl->il && c->slot == w.slot); // a placement never drops a copy still copying
        if (c->slab[w.widx] == LLAMA_MOE_LEND_DONE) {
            return;
        }
        if (lend.claim(w.save, w.widx)) {
            lend_copy_out(*this, lk, w.save, w.widx);
            return;
        }
        mgr.cv_done.wait(lk); // RUNNING: someone else is copying it out
    }
}

// LLAMA_MOE_ROOM_POISON: a slab copied back must be the file's own bytes, or the lent belt handed the desk
// a wrong book
static void lend_check_file(llama_moe_stream & mgr, const llama_moe_stream_layer & sl,
        const llama_moe_stream_work & w, const uint8_t * got) {
    const llama_moe_stream_weight & wt = sl.weights[w.widx];
    uint8_t * staging = (uint8_t *) moe_aligned_alloc(wt.nb_expert + 2*MOE_STREAM_DIRECT_ALIGN);
    GGML_ASSERT(staging != nullptr);
    const bool alt = mgr.use_alt(w.expert, sl.n_expert);
    const uint8_t * want = llama_moe_stream_pread(*(alt ? mgr.files_alt : mgr.files)[wt.file_idx], staging,
            wt.nb_expert, wt.offs + (size_t) w.expert*wt.nb_expert, mgr.use_direct_io);
    const bool same = want != nullptr && memcmp(want, got, wt.nb_expert) == 0;
    moe_aligned_free(staging);
    if (!same) {
        GGML_ABORT("lent belt: floor %d book %d slab %d copied back from the belt is not the file's", sl.il, w.expert, w.widx);
    }
}

void llama_moe_room::lend_restore_locked(std::unique_lock<std::mutex> & lk, const llama_moe_stream_work & w) {
    llama_moe_stream_layer & sl = *w.sl;
    if (w.save) {
        lend_gate_locked(lk, w); // the slot's old book goes onto the belt first
    }
    const llama_moe_lend_copy * c = lend.at(w.lent);
    GGML_ASSERT(c != nullptr && c->complete() && c->pins > 0 && c->il == sl.il && c->book == w.expert);
    const size_t    nb  = sl.weights[w.widx].nb_expert;
    const uint8_t * src = belt_at(*this, c->offs + floors[sl.il].w_offs[w.widx]);
    uint8_t *       dst = desk_slab(sl, w.slot, w.widx);

    lk.unlock();
    const int64_t t0 = ggml_time_us();
    memcpy(dst, src, nb);
    const int64_t t1 = ggml_time_us();
    if (poison) {
        lend_check_file(mgr, sl, w, dst);
    }
    lk.lock();

    // no drive, so none of the drive's stats: n_slabs_read, the bytes and busy time of either runner
    lend.unpin(w.lent);
    lstats.n_bytes_out  += (int64_t) nb;
    lstats.t_restore_us += t1 - t0;
    if (w.gen == sl.slot_gen[w.slot] && sl.slot_pending[w.slot] > 0 && --sl.slot_pending[w.slot] == 0) {
        // the trip protocol: the last slab publishes (with release, for the remap's quick path)
        moe_slot_state_publish(sl.slot_state[w.slot], LLAMA_MOE_STREAM_SLOT_RESIDENT);
    }
    mgr.cv_done.notify_all();
}

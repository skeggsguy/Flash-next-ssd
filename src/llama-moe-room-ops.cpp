// The reading room's CPU ops (llama-moe-room.h): per floor, one desk op and one op per part, inserted by
// build_moe_room_experts (llama-graph-moe-room.cpp). Each writes the id plane its GEMM group reads.
//
// Why a part may be handed back here: these ops run on the CPU, and the scheduler drains the GPU before
// every CPU split (docs/wizard-reading-room.md section 2). A part op's input is a one-element view of the
// previous group's down link, so it runs after that group's GEMMs finished; a desk op's input is its
// floor's router, which the previous floor's output feeds. So when an op hands back every part up to the
// newest one it gave a GEMM (in_use_seq), no GEMM still running can read them.

#include "llama-moe-room.h"

#include "llama-impl.h"
#include "llama-moe-room-plan.h"
#include "llama-moe-stream.h"

#include <chrono>
#include <cinttypes>
#include <mutex>

// a part op that waits this long for its books has lost its runners (a hung drive, a bug): say so and
// stop, rather than hang the server silently
static const int64_t ROOM_WATCHDOG_US = 120*1000000ll;

void llama_moe_room::desk_locked(std::unique_lock<std::mutex> & lk, llama_moe_room_floor & F,
        const int32_t * ids, int64_t n, int32_t * out) {
    llama_moe_stream_layer & sl = *F.sl;

    // the first room floor's desk op begins the ubatch: its CPU split has drained the GPU, so nothing
    // from the last read-in is still running
    int32_t first = -1;
    for (const auto & f : floors) {
        if (f.sl != nullptr && f.in_graph) {
            first = f.sl->il;
            break;
        }
    }
    if (sl.il == first) {
        begin_ubatch_locked(lk);
    }
    if (op_floor >= order.size() || order[op_floor] != sl.il || op_group != -1) {
        GGML_ABORT("reading room: floor %d's desk op ran out of order", sl.il);
    }

    release_locked(); // every part earlier floors used

    // the desk books this ubatch reads: a LOADING one (a guess, or this read-in's cold fill) counts as on
    // the desk, so wait until it has landed, exactly as the remap does
    sl.touched.assign(sl.n_expert, 0);
    sl.demand_slots.clear();
    int64_t n_hit = 0;
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
        if (sl.touched[e]) {
            continue;
        }
        sl.touched[e] = 1;
        const auto it = sl.expert_slot.find(e);
        if (it == sl.expert_slot.end()) {
            continue;
        }
        if (sl.slot_state[it->second] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            mgr.promote_slot_locked(sl, it->second);
        }
        sl.demand_slots.push_back(it->second);
        n_hit++;
    }
    const int64_t t0 = ggml_time_us();
    mgr.cv_done.wait(lk, [&] {
        if (mgr.load_failed) {
            return true;
        }
        for (const int32_t s : sl.demand_slots) {
            if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                return false;
            }
        }
        return true;
    });
    if (mgr.load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }
    const int64_t t_wait = ggml_time_us() - t0;

    F.where.assign(sl.n_expert, -1);
    for (const int32_t s : sl.demand_slots) {
        F.where[sl.slot_expert[s]] = s;
    }
    F.owned = llama_moe_room_emit(ids, n, F.where, out);
    for (int64_t i = 0; i < n; i++) {
        if (out[i] >= 0) {
            sl.slot_last_use[out[i]] = ++sl.use_counter; // the desk's LRU stays sane after the sweep
        }
    }
    if (mgr.gpu_slot) {
        sl.publish_state_locked(mgr);
    }

    // counted as one wave, its books as hits, its cold fills as misses (the stats line's words)
    mgr.stats.n_calls++;
    mgr.stats.n_wave_calls++;
    mgr.stats.n_waves_run++;
    mgr.stats.n_hit         += n_hit;
    mgr.stats.n_miss        += (int64_t) F.fills.size();
    mgr.stats.t_stall_wave_us += t_wait;
    sl.n_hit  += n_hit;
    sl.n_miss += (int64_t) F.fills.size();
    stats.n_floors++;
    stats.n_groups++;
    stats.t_wait_us += t_wait;

    op_group = 0;
}

void llama_moe_room::part_locked(std::unique_lock<std::mutex> & lk, llama_moe_room_floor & F, int32_t g,
        const int32_t * ids, int64_t n, int32_t * out) {
    llama_moe_stream_layer & sl = *F.sl;
    if (op_floor >= order.size() || order[op_floor] != sl.il || op_group != g || (size_t) g >= F.parts.size()) {
        GGML_ABORT("reading room: floor %d's part %d op ran out of order", sl.il, g);
    }

    release_locked(); // the previous part: this op's input is its group's down link

    const std::vector<int32_t> & books = F.parts[g];
    F.where.assign(sl.n_expert, -1);
    int64_t t_wait = 0;
    if (!books.empty()) {
        const int64_t t0 = ggml_time_us();
        llama_moe_part * p = nullptr;
        for (bool first = true; ; first = false) {
            if (mgr.load_failed) {
                GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
            }
            p = F.part_seq[g] ? belt.find(F.part_seq[g]) : nullptr;
            if (p != nullptr && p->state == LLAMA_MOE_PART_READY) {
                stats.n_part_waits += !first;
                break;
            }
            GGML_ASSERT(p == nullptr || p->state == LLAMA_MOE_PART_FILLING);
            if (ggml_time_us() - t0 > ROOM_WATCHDOG_US) {
                GGML_ABORT("reading room: floor %d part %d has waited 120 s for its books; the runners have stopped",
                        sl.il, g);
            }
            cv.wait_for(lk, std::chrono::seconds(1));
            belt.reclaim();
            pump_locked();
        }
        t_wait = ggml_time_us() - t0;

        for (const auto & q : belt.parts) {
            stats.n_ahead += q.seq > p->seq && q.state == LLAMA_MOE_PART_READY;
        }
        p->state   = LLAMA_MOE_PART_IN_USE;
        in_use_seq = p->seq;

        const uint32_t r0 = p->first_record();
        for (size_t j = 0; j < books.size(); j++) {
            F.where[books[j]] = (int32_t) (r0 + j);
        }
    }
    const int64_t owned = llama_moe_room_emit(ids, n, F.where, out);
    F.owned += owned;

    // books no word on this floor read: the price of fetching before the router has run
    sl.touched.assign(sl.n_expert, 0);
    for (int64_t i = 0; i < n; i++) {
        sl.touched[ids[i]] = 1;
    }
    for (const int32_t e : books) {
        stats.n_books_idle += !sl.touched[e];
    }

    mgr.stats.n_wave_calls++;
    mgr.stats.n_waves_run++;
    mgr.stats.n_miss          += (int64_t) books.size();
    mgr.stats.t_stall_wave_us += t_wait;
    sl.n_miss += (int64_t) books.size();
    stats.n_groups++;
    stats.n_part_ops++;
    stats.t_wait_us    += t_wait;

    if ((size_t) g + 1 < F.parts.size()) {
        op_group = g + 1;
        return;
    }
    // the floor's planes together must own every pair exactly once (each is a partition of the books),
    // or an output row is left unwritten
    if (F.owned != n) {
        GGML_ABORT("reading room: floor %d's GEMM groups own %" PRId64 " of its %" PRId64 " (word, book) pairs",
                sl.il, F.owned, n);
    }
    op_floor++;
    op_group = -1;
}

// shared entry of the two ops: checks the tensors, takes the manager's lock, times the op
template <typename F>
static void room_op(ggml_tensor * dst, int ith, void * userdata, F && body) {
    if (ith != 0) {
        return;
    }
    const int64_t t_op0 = ggml_time_us();

    auto * op  = (llama_moe_room_op *) userdata;
    auto * mgr = op->sl->mgr;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));
    GGML_ASSERT(dst->data != a->data);

    std::unique_lock<std::mutex> lk(mgr->mtx);
    mgr->maybe_dump_stats_locked();
    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }
    mgr->start_workers_locked();

    llama_moe_room_floor * fl = mgr->room->floor(op->sl->il);
    GGML_ASSERT(fl != nullptr);
    body(lk, *mgr->room, *fl, op->group, (const int32_t *) a->data, ggml_nelements(a), (int32_t *) dst->data, a->ne[0]);

    mgr->stats.t_wave_op_us += ggml_time_us() - t_op0;
}

void llama_moe_room_desk_ids(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    room_op(dst, ith, userdata, [](std::unique_lock<std::mutex> & lk, llama_moe_room & room, llama_moe_room_floor & F,
            int32_t, const int32_t * ids, int64_t n, int32_t * out, int64_t n_ids) {
        // the borrowing log sees reading in through the room too: once per floor per ubatch, kind 1
        room.mgr.trace_record_locked(F.sl->il, (uint32_t) (n/n_ids), 1, ids, n);
        room.desk_locked(lk, F, ids, n, out);
    });
}

void llama_moe_room_part_ids(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    room_op(dst, ith, userdata, [](std::unique_lock<std::mutex> & lk, llama_moe_room & room, llama_moe_room_floor & F,
            int32_t g, const int32_t * ids, int64_t n, int32_t * out, int64_t) {
        room.part_locked(lk, F, g, ids, n, out);
    });
}

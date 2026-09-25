#pragma once

// The reading room, manager side: the book manager's (llama_moe_stream, llama-moe-stream.h) belt of
// books for long read-ins. The sizing arithmetic is llama-moe-room-size.h, the belt's bookkeeping
// llama-moe-room-belt.h, a floor's plan and id planes llama-moe-room-plan.h, the CPU ops that drive all
// of this from the graph llama-moe-room-ops.cpp, and the graph itself llama-graph-moe-room.cpp.
//
// How a read-in of sweep_min_tokens or more runs. The first room floor's desk op begins the ubatch: it
// plans every room floor from its desk (books not on the desk go on the belt, a cold desk's EMPTY slots
// are filled first) and the pump starts the runners on the belt, floor after floor in the order the GPU
// will need the books, as far ahead as the belt has room. Each floor then runs one desk op (wait for the
// desk books this ubatch reads, emit the desk's ids, hand back the parts earlier floors used) and one op
// per part (hand back the previous part, wait until this one is READY, emit its ids). Every one of
// these is a CPU op after GPU work, and the scheduler drains the GPU before any CPU split (the CPU
// backend has no async copy in, docs/wizard-reading-room.md section 2), so a part handed back in an op
// is never read again by a GEMM still running. The desk is never evicted: writing keeps its books.

#include "llama.h"
#include "llama-arch.h"
#include "llama-moe-room-belt.h"
#include "llama-moe-room-size.h"

#include "ggml-cpp.h"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

struct llama_hparams;
struct llama_model_loader;
struct llama_moe_stream;
struct llama_moe_stream_layer;
struct llama_moe_stream_work;

// Sizes the room for a model about to load, from the desk's slots with the room off: the layout's
// desk_slots are what the desk keeps. Logs the startup line (and any warning) in plain words; throws
// std::runtime_error with the plain-words reason when the room cannot be made - an arch whose expert
// maths a split floor would change, or a size the arithmetic refuses.
llama_moe_room_layout llama_moe_room_size_model(const llama_model_params & params, llm_arch arch,
        const llama_hparams & hparams, const llama_model_loader & ml, uint32_t n_slots);

// userdata of one room op: the floor and its group, -1 the desk, 0.. a part
struct llama_moe_room_op {
    llama_moe_stream_layer * sl = nullptr;
    int32_t group = -1;
};

// one floor's share of the room
struct llama_moe_room_floor {
    llama_moe_stream_layer * sl = nullptr;
    size_t   stride    = 0;              // a book's record: its weights in sl->weights order, each 256-aligned
    uint32_t n_records = 0;              // whole records of this floor that fit on the belt
    std::vector<size_t>        w_offs;   // each weight's offset in the record
    std::vector<ggml_tensor *> views;    // each weight's view of the whole belt: nb[2] = stride
    std::vector<std::unique_ptr<llama_moe_room_op>> ops; // stable op userdata, [1 + parts]

    bool in_graph = false; // this floor's graph takes the room (set at graph build, like plan_pair_chunk)

    // this ubatch
    std::vector<std::pair<int32_t, int32_t>> fills; // (book, desk slot) the cold desk takes
    std::vector<std::vector<int32_t>>        parts; // each part's books
    std::vector<uint64_t> part_seq;                 // each part's belt seq once placed, 0 = not (yet)
    int64_t owned = 0;                              // pairs its planes own so far
    std::vector<int32_t> where;                     // scratch: book -> slot or record
};

// cumulative; the stats dump prints the deltas of a window
struct llama_moe_room_stats {
    int64_t n_ubatches   = 0;
    int64_t n_floors     = 0; // floors run through the room
    int64_t n_groups     = 0; // GEMM groups (desk + parts)
    int64_t n_books      = 0; // books brought on the belt
    int64_t n_books_idle = 0; // of those, books no word on the floor read
    int64_t n_bytes      = 0; // bytes read onto the belt
    int64_t n_bytes_cancelled = 0;
    int64_t t_wait_us    = 0; // desk and part ops waiting for books
    int64_t n_part_waits = 0; // part ops that waited
    int64_t n_ahead      = 0; // sum over part ops of the parts already READY behind the one needed
    int64_t n_part_ops   = 0;
};

struct llama_moe_room {
    llama_moe_stream &    mgr;
    llama_moe_room_layout lay;

    ggml_context_ptr      ctx;             // its own no_alloc context: the manager's is sized for the desk only
    ggml_backend_buffer_t buf   = nullptr; // owned by mgr.bufs
    ggml_tensor *         whole = nullptr; // the whole belt as bytes, for uploads and poison
    uint8_t *             host  = nullptr; // the belt's host pointer, null when it has none

    llama_moe_belt belt;
    std::vector<llama_moe_room_floor> floors; // [n_layer], sl null = not streamed

    std::condition_variable cv; // a part turned READY, a read failed, or memory came free

    bool poison = false; // LLAMA_MOE_ROOM_POISON=1: handed-back parts are filled with 0xFF

    // this ubatch: its floors in graph order, the pump's place and the ops' expected order
    std::vector<int32_t> order;
    size_t   pump_floor = 0;
    int32_t  pump_part  = -1; // -1: the floor's fills are not queued yet
    size_t   op_floor   = 0;
    int32_t  op_group   = -1;
    uint64_t in_use_seq = 0;  // the newest part a GEMM has been given

    llama_moe_room_stats stats;
    llama_moe_room_stats stats_prev;

    llama_moe_room(llama_moe_stream & mgr, const llama_moe_room_layout & lay);

    // the belt buffer after the desk's (the gentle opening's pause first) and each floor's views of it
    void alloc(bool no_alloc);

    llama_moe_room_floor * floor(int32_t il) {
        return il >= 0 && (size_t) il < floors.size() && floors[il].sl ? &floors[il] : nullptr;
    }

    // graph build: whether a ubatch of n_tokens on floor il takes the room, recorded for the pump
    bool take(int32_t il, int64_t n_tokens, bool allowed);
    llama_moe_room_op * op_userdata(llama_moe_stream_layer * sl, int32_t group);

    // all under mgr.mtx
    void begin_ubatch_locked();
    void pump_locked();
    void release_locked();            // hand back every part up to in_use_seq, poison, reclaim, pump
    bool worker_begin_locked(const llama_moe_stream_work & w); // false: the read is stale, skip it
    void worker_end_locked(const llama_moe_stream_work & w, bool ok);

    // the stats lines (llama-moe-room-stats.cpp): a window's `moe stream: room` line, and the run's total
    void dump_stats_locked(int64_t dt_us);
    void print_stats_locked() const;

    // the ops (llama-moe-room-ops.cpp), under mgr.mtx: each writes its group's id plane
    void desk_locked(std::unique_lock<std::mutex> & lk, llama_moe_room_floor & F, const int32_t * ids, int64_t n, int32_t * out);
    void part_locked(std::unique_lock<std::mutex> & lk, llama_moe_room_floor & F, int32_t g, const int32_t * ids, int64_t n, int32_t * out);
};

// callbacks of the room's CPU ops, inserted by build_moe_room_experts (llama-graph-moe-room.cpp); src[0]
// is the contiguous selected ids, a part op's src[1] a one-element view of the previous group's down link
void llama_moe_room_desk_ids(ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_moe_room_part_ids(ggml_tensor * dst, int ith, int nth, void * userdata);

#pragma once

// The lent belt: while the library writes, the reading room's belt (llama-moe-room.h) sits idle, so it is
// lent back to the desk as a store of copies of the books the writing remap puts back (a victim cache).
// A later trip for one of those books becomes a copy from the belt instead of a read from the stacks. This
// file is the bookkeeping only, with no ggml and no locking (the manager holds its mutex around every
// call), so its invariants are unit-tested against an interval oracle (tests/test-moe-room-lend.cpp). The
// glue that drives it from the book manager is llama-moe-room-lend-ops.cpp; the design is the study's
// docs/lend-back-plan.md.
//
// The ring: one first-in first-out run of copies over the belt's bytes, in the order the books were put
// back. A copy is its floor's belt record (every weight of one book, each 256-aligned), placed at 256 B
// after the newest copy, or at the start when the end is too short (the bytes skipped at the end go with
// it); the oldest copies in its way are dropped. No GEMM view indexes the lent belt, so copies need not
// sit at record-stride multiples as the room's parts do. Each of a copy's slabs goes
//
//   PENDING -> RUNNING -> DONE      (claimed by whoever copies it out of the desk slot, then copied)
//
// A copy can be found (for a restore) only once every slab is DONE. It is busy while a slab is not DONE
// or a restore still reads it (a pin), and a busy copy is never dropped: a placement that would drop one
// fails instead, and that book is not kept. Seqs are never reused, so the index from (floor, book) to a
// copy may go stale: a lookup checks the seq is still in the ring. That makes clear() an O(1) epoch for
// the index, which the room pays each time it takes the belt back for a read-in.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

static const size_t  LLAMA_MOE_LEND_ALIGN     = 256;
static const int32_t LLAMA_MOE_LEND_SLABS_MAX = 4; // a book's weights: fused gate_up + down, or gate, up, down

enum llama_moe_lend_slab : uint8_t {
    LLAMA_MOE_LEND_PENDING = 0, // not copied yet
    LLAMA_MOE_LEND_RUNNING = 1, // someone is copying it out of the desk slot, outside the lock
    LLAMA_MOE_LEND_DONE    = 2,
};

enum llama_moe_lend_keep {
    LLAMA_MOE_LEND_KEPT    = 0, // a new copy, every slab PENDING
    LLAMA_MOE_LEND_HELD    = 1, // the book already has a copy on the belt: never copied twice
    LLAMA_MOE_LEND_BLOCKED = 2, // a busy copy is in the way (or the record is bigger than the belt)
};

struct llama_moe_lend_copy {
    uint64_t seq     = 0;
    int32_t  il      = -1;
    int32_t  book    = -1;
    int32_t  slot    = -1; // the desk slot its slabs are copied out of
    size_t   offs    = 0;  // [offs, offs + bytes) of the belt, offs a multiple of LLAMA_MOE_LEND_ALIGN
    size_t   bytes   = 0;
    int32_t  n_slabs = 0;
    uint8_t  slab[LLAMA_MOE_LEND_SLABS_MAX] = {}; // llama_moe_lend_slab
    int32_t  pins    = 0;  // restores reading it right now

    bool complete() const;
    bool busy() const { return pins > 0 || !complete(); }
};

struct llama_moe_lend {
    size_t size = 0;

    std::deque<llama_moe_lend_copy> ring; // oldest first; their seqs are consecutive
    uint64_t next_seq  = 1;
    size_t   head      = 0; // the end of the newest copy placed since the last clear
    int64_t  n_dropped = 0; // copies dropped to make room, over the whole run

    explicit llama_moe_lend(size_t size = 0) : size(size) {}

    // where a copy of `bytes` would go and how many of the oldest copies it would drop; false when a busy
    // copy is in the way or the copy is bigger than the belt
    bool place(size_t bytes, size_t & offs, size_t & n_drop) const;

    // a new copy of `book` of floor `il`, out of desk slot `slot`, with every slab PENDING; nullptr when not
    // kept, `why` saying whether the book is already held or the placement was blocked (the ring unchanged)
    llama_moe_lend_copy * keep(int32_t il, int32_t book, int32_t slot, size_t bytes, int32_t n_slabs,
            llama_moe_lend_keep & why);

    llama_moe_lend_copy * at(uint64_t seq); // the copy with this seq, nullptr once it is dropped or cleared

    llama_moe_lend_copy * find(int32_t il, int32_t book); // the book's copy, only when complete

    // a slab PENDING -> RUNNING (false: it was not PENDING), and RUNNING -> DONE (true: the copy is now complete)
    bool claim(uint64_t seq, int32_t slab);
    bool finish(uint64_t seq, int32_t slab);

    void pin(uint64_t seq);
    void unpin(uint64_t seq);

    size_t clear();     // drops every copy (returns how many); must not be busy
    bool   busy() const; // any copy busy: a keep not finished or a restore still reading
    size_t held() const; // complete copies

private:
    std::vector<std::vector<uint64_t>> index; // [il][book] -> seq, possibly stale; 0 = none
    uint64_t & index_of(int32_t il, int32_t book);
};

// How many books one remap call may keep on one floor: max(k, the belt's bytes over one record of every
// floor), i.e. a floor's fair share of the belt (~10 at Flash-Next's 1.44 GiB belt over 48 floors), but
// never fewer than the books one written word reads there, so a single token's put-backs always fit.
uint32_t llama_moe_lend_cap(uint32_t n_expert_used, size_t belt_bytes, size_t sum_strides);

// how many books the lent belt holds at the floors' mean record size (the load line's "up to ~N books")
uint32_t llama_moe_lend_capacity(size_t belt_bytes, size_t sum_strides, size_t n_floors);

#pragma once

// The reading room's belt: one buffer of book records, filled first-in first-out, pure bookkeeping
// with no ggml and no locking (the manager, llama-moe-room.h, holds its mutex around every call), so
// its invariants are unit-tested against an interval-set oracle (tests/test-moe-room-belt.cpp).
//
// A part is some of one floor's books, contiguous on the belt, starting at a multiple of that floor's
// record stride so its records are whole indices of the floor's view of the belt. It goes
//
//   FILLING -> READY -> IN_USE -> RELEASED        (the runners filled it, the GPU used it, handed back)
//   FILLING | READY | IN_USE -> CANCELLED         (a read-in that stopped before it was used)
//
// and its memory is reused only once it is RELEASED or CANCELLED, it is the oldest part still holding
// memory (first in, first out), and no read is landing in it (inflight == 0): the runners read straight
// into the belt, so a cancelled part's in-flight read still lands where the part was.

#include <cstddef>
#include <cstdint>
#include <deque>

enum llama_moe_part_state : uint8_t {
    LLAMA_MOE_PART_FILLING   = 0,
    LLAMA_MOE_PART_READY     = 1,
    LLAMA_MOE_PART_IN_USE    = 2,
    LLAMA_MOE_PART_RELEASED  = 3,
    LLAMA_MOE_PART_CANCELLED = 4,
};

struct llama_moe_part {
    uint64_t seq       = 0; // never reused, so a queued read can tell its part is gone
    int32_t  il        = -1;
    int32_t  index     = -1; // which part of its floor
    size_t   offs      = 0;  // [offs, offs + bytes) of the belt
    size_t   bytes     = 0;
    size_t   stride    = 0;  // its floor's record stride; offs is a multiple of it
    uint32_t n_records = 0;
    int32_t  pending   = 0;  // reads still to land before it is READY
    int32_t  inflight  = 0;  // reads landing right now
    llama_moe_part_state state = LLAMA_MOE_PART_FILLING;

    uint32_t first_record() const { return (uint32_t) (offs/stride); }
};

struct llama_moe_belt {
    size_t size = 0;

    std::deque<llama_moe_part> parts; // every part still holding memory, oldest first
    uint64_t next_seq = 1;

    explicit llama_moe_belt(size_t size = 0) : size(size) {}

    // where a part of n_records records of `stride` bytes would start, or false while there is no room.
    // Contiguous and after the newest part (wrapping to the start when the end is too short), never over
    // a part still holding memory.
    bool place(uint32_t n_records, size_t stride, size_t & offs) const;

    // places and appends a FILLING part that waits for `pending` reads; nullptr while there is no room
    llama_moe_part * push(int32_t il, int32_t index, uint32_t n_records, size_t stride, int32_t pending);

    llama_moe_part * find(uint64_t seq);

    // a runner is about to read into part seq: nullptr when that part is gone or cancelled (the read is
    // stale and must not happen), otherwise the part, now with one more read in flight
    llama_moe_part * begin_read(uint64_t seq);

    // that read landed (ok) or failed: true when it was the part's last and the part is now READY. A
    // failed read never makes a part READY; a cancelled part only counts its landing reads down.
    bool end_read(uint64_t seq, bool ok);

    // READY or IN_USE parts with seq <= through are handed back
    void release_through(uint64_t through);

    // every part not yet handed back is cancelled; its memory waits for its landing reads
    void cancel_all();

    // frees the oldest parts that are handed back or cancelled and have no read landing; returns how many
    size_t reclaim();

    size_t held_bytes() const; // bytes of every part still holding memory
};

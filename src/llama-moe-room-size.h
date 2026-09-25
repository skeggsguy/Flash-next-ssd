#pragma once

// The reading room's size (--moe-stream-room, --moe-stream-room-parts): pure arithmetic, no manager,
// no ggml, so the whole of it is unit-tested (tests/test-moe-room-size.cpp).
//
// The reading room is a belt of books for long read-ins, carved out of the desk's budget
// (--moe-stream-cache): desk = budget - room. It is sized in floors of look-ahead, one floor being the
// books a floor does NOT have on its desk, so a room of N floors lets the runners work N floors ahead of
// the desk work. A floor's books go on the belt as records (every weight of one book, each weight
// 256-aligned for the reads), cut into `parts` parts; a part is handed back as soon as the GPU is done
// with it. llama-moe-room.h is the manager side, llama-moe-room-belt.h the belt itself.

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>

static const double   LLAMA_MOE_ROOM_AUTO_FLOORS   = 1.25; // --moe-stream-room auto
static const int32_t  LLAMA_MOE_ROOM_PARTS_DEFAULT = 4;
static const int32_t  LLAMA_MOE_ROOM_PARTS_MAX     = 16;
static const uint32_t LLAMA_MOE_ROOM_RECORDS_MAX   = 1024; // Metal's book index map (map0) holds at most 1024 books
static const size_t   LLAMA_MOE_ROOM_SLAB_ALIGN    = 256;

// what was asked for: llama_model_params' moe_stream_room_mode / _value / _parts
struct llama_moe_room_request {
    int32_t mode  = LLAMA_MOE_ROOM_OFF; // enum llama_moe_room_mode
    double  value = 0.0;                // GiB or floors, by mode
    int32_t parts = LLAMA_MOE_ROOM_PARTS_DEFAULT;
};

// what the model's books look like, measured from the file before anything is allocated
struct llama_moe_room_books {
    uint64_t budget        = 0; // desk + room, bytes (--moe-stream-cache; in slots mode slots x book_bytes)
    uint32_t desk_slots    = 0; // slots per floor the whole budget buys, i.e. the desk with the room off
    uint64_t book_bytes    = 0; // one desk slot on every streamed floor, summed: what a slot per floor costs
    uint64_t stride_min    = 0; // one book's belt record on the floor with the smallest books
    uint64_t stride_max    = 0; // ... and on the floor with the largest (q4_K and q5_K floors differ)
    uint32_t n_expert      = 0; // books per floor
    uint32_t n_expert_used = 0; // books a word reads per floor (the slip's length)
    uint32_t sweep_min_tokens = 0; // read-ins of at least this many tokens use the room
};

struct llama_moe_room_layout {
    bool     on               = false;
    uint64_t room_bytes       = 0;
    uint32_t desk_slots       = 0; // slots per floor left to the desk
    uint64_t desk_bytes       = 0;
    int32_t  parts            = 0;
    double   floors           = 0.0; // floors of look-ahead the room holds (largest floor)
    uint64_t part_bytes_max   = 0;   // the largest part a floor can be cut into
    uint32_t n_records_max    = 0;   // records of the smallest-book floor that fit: the GEMM's ne02
    uint32_t sweep_min_tokens = 0;

    std::string error;   // non-empty: the room cannot be made; the model must not load
    std::string warning; // non-empty: it loads, but this is worth saying
};

// a book's belt record: its weights one after another, each starting 256-aligned
size_t llama_moe_room_record_stride(const size_t * nb_weight, size_t n_weight);

// Read-ins of at least this many tokens sweep the room rather than run waves on the desk. The default
// is the size at which, if every book were equally likely, fewer than 1% of a floor's books would go
// unread: ceil(ln 0.01 / ln(1 - k/n)), 234 at 10 of 512 and 35 at 8 of 64. Real routing is skewed, so
// more books than that go unread; what makes the sweep worth it is cost (a whole sweep at 12.3 GB/s
// against today's waves at 300-2,500 tokens), which is why the rung may move it with
// LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS (read by the manager, not here).
uint32_t llama_moe_room_sweep_min_tokens(uint32_t n_expert, uint32_t n_expert_used);

// the layout the request makes of these books, or the plain-words reason it cannot
llama_moe_room_layout llama_moe_room_resolve(const llama_moe_room_request & req, const llama_moe_room_books & books);

// the startup line, in the study's words
std::string llama_moe_room_describe(const llama_moe_room_layout & lay);

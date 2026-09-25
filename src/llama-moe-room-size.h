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

// The room takes a reading-in batch once the batch's slips name each book about this many times on
// average. The room fetches every book a floor's desk lacks whether or not a slip asks for it, while
// waves fetch only the books the slips name, so the room only pays once few books go unasked.
// Measured on Flash-Next (RR-room, 2026-09-26: 12 real papers, room vs waves at desk 30, two runners):
// read-ins of 300-1,000 tokens (6-20 slips per book) were 16% slower with the room, 1,000-2,000 (20-40)
// level, 2,000 and up 21-29% faster, and replaying those steps with the room from 1,024 tokens gave the
// best total (-22.4% pen to paper, level with 1,536; 512 gave -21.8%). It is counted in slips per
// book, not tokens, because how many books go unasked follows the slips each book gets, t x k / n
// (t tokens, k books a word reads, n books a floor), not t alone: under even routing the unasked share
// is about e^-(t k / n) on any shape. So the same 20 carries to a model with more or fewer books,
// assuming its real routing is skewed alike.
static const uint32_t LLAMA_MOE_ROOM_SLIPS_PER_BOOK = 20;

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

// Read-ins of at least this many tokens sweep the room rather than run waves on the desk:
// ceil(LLAMA_MOE_ROOM_SLIPS_PER_BOOK x n / k) for n books a floor and k a word reads, 1,024 at 10 of 512
// (Flash-Next) and 160 at 8 of 64 (the test fixtures); 1 when every word reads every book, because then
// waves fetch every book too. The threshold is per reading-in batch (ubatch): a read-in cut into
// batches takes the room for each batch that reaches it, so a long read-in's last, shorter batch
// (a 17K paper at -ub 4096 ends with a few hundred tokens) takes waves - meant, not a gap: RR-room
// measured a batch that size as faster with waves, and the room's batches before it never evict from
// the desk it then uses.
uint32_t llama_moe_room_sweep_min_tokens(uint32_t n_expert, uint32_t n_expert_used);

// The threshold the room runs with. env_value is LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS's value (the manager
// reads the environment, not this file): when it is set it wins, as a whole number of at least 1, so a
// rung can move the threshold without a rebuild; null gives the default above.
uint32_t llama_moe_room_sweep_min_tokens_env(const char * env_value, uint32_t n_expert, uint32_t n_expert_used);

// the layout the request makes of these books, or the plain-words reason it cannot
llama_moe_room_layout llama_moe_room_resolve(const llama_moe_room_request & req, const llama_moe_room_books & books);

// the startup line, in the study's words
std::string llama_moe_room_describe(const llama_moe_room_layout & lay);

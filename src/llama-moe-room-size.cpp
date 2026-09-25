#include "llama-moe-room-size.h"

#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

static const double GIB = 1024.0*1024.0*1024.0;
static const double MIB = 1024.0*1024.0;

static uint64_t align_up(uint64_t n, uint64_t a) {
    return (n + a - 1)/a*a;
}

size_t llama_moe_room_record_stride(const size_t * nb_weight, size_t n_weight) {
    size_t stride = 0;
    for (size_t i = 0; i < n_weight; i++) {
        stride += (size_t) align_up(nb_weight[i], LLAMA_MOE_ROOM_SLAB_ALIGN);
    }
    return stride;
}

uint32_t llama_moe_room_sweep_min_tokens(uint32_t n_expert, uint32_t n_expert_used) {
    if (n_expert_used == 0 || n_expert_used >= n_expert) {
        return 1; // every word reads every book: any read-in is a sweep
    }
    // t tokens hand each book t x k / n slips on average: the smallest t that reaches the target
    const uint64_t slips = (uint64_t) LLAMA_MOE_ROOM_SLIPS_PER_BOOK*n_expert;
    return (uint32_t) ((slips + n_expert_used - 1)/n_expert_used);
}

uint32_t llama_moe_room_sweep_min_tokens_env(const char * env_value, uint32_t n_expert, uint32_t n_expert_used) {
    if (env_value != nullptr) {
        return (uint32_t) std::max(1, atoi(env_value));
    }
    return llama_moe_room_sweep_min_tokens(n_expert, n_expert_used);
}

// the room's bytes for `floors` floors of look-ahead when the desk has `desk` slots per floor: that
// many copies of the books a floor leaves off its desk, on the floor whose books are largest
static uint64_t room_for_floors(double floors, uint32_t desk, const llama_moe_room_books & b) {
    const uint32_t missing = b.n_expert - std::min(desk, b.n_expert);
    return align_up((uint64_t) std::ceil(floors*missing*(double) b.stride_max), LLAMA_MOE_ROOM_SLAB_ALIGN);
}

llama_moe_room_layout llama_moe_room_resolve(const llama_moe_room_request & req, const llama_moe_room_books & b) {
    llama_moe_room_layout lay;
    lay.desk_slots       = b.desk_slots;
    lay.desk_bytes       = (uint64_t) b.desk_slots*b.book_bytes;
    lay.sweep_min_tokens = b.sweep_min_tokens;

    if (req.mode == LLAMA_MOE_ROOM_OFF) {
        return lay;
    }
    if (req.parts < 1 || req.parts > LLAMA_MOE_ROOM_PARTS_MAX) {
        lay.error = format("--moe-stream-room-parts must be between 1 and %d (got %d)", LLAMA_MOE_ROOM_PARTS_MAX, req.parts);
        return lay;
    }
    const bool by_floors = req.mode == LLAMA_MOE_ROOM_AUTO || req.mode == LLAMA_MOE_ROOM_FLOORS;
    const double floors  = req.mode == LLAMA_MOE_ROOM_AUTO ? LLAMA_MOE_ROOM_AUTO_FLOORS : req.value;
    if ((req.mode != LLAMA_MOE_ROOM_AUTO && !(req.value > 0.0)) || (!by_floors && req.mode != LLAMA_MOE_ROOM_GIB)) {
        lay.error = format("--moe-stream-room: cannot make a reading room of mode %d, size %g: give auto, a size in "
                           "GiB, or a number of floors like 1.25f", req.mode, req.value);
        return lay;
    }
    if (b.book_bytes == 0 || b.stride_min == 0) {
        lay.error = "--moe-stream-room: this model has no streamed books (no floor with expert weights), so there is "
                    "nothing for a reading room to hold";
        return lay;
    }
    if (b.desk_slots >= b.n_expert) {
        lay.error = format("--moe-stream-room: the desk already holds every book (%u slots per floor for %u books), so "
                           "reading in never fetches and a reading room would only shrink it; run with --moe-stream-room 0",
                           b.desk_slots, b.n_expert);
        return lay;
    }

    // The room shrinks the desk, and a smaller desk leaves more books per floor for the room to hold,
    // so a room sized in floors is the fixed point of the two. Starting from the whole budget the desk
    // only ever shrinks, one step at a time, so this ends within n_expert steps.
    uint64_t room = 0;
    uint32_t desk = b.desk_slots;
    for (uint32_t it = 0; it <= b.n_expert + 1; it++) {
        room = by_floors ? room_for_floors(floors, desk, b)
                         : (uint64_t) (req.value*GIB)/LLAMA_MOE_ROOM_SLAB_ALIGN*LLAMA_MOE_ROOM_SLAB_ALIGN;
        if (room >= b.budget) {
            lay.error = format("--moe-stream-room: the reading room (%.2f GiB) takes the whole desk budget "
                            "(%.2f GiB); make it smaller or give --moe-stream-cache more",
                            room/GIB, b.budget/GIB);
            return lay;
        }
        const uint32_t next = (uint32_t) std::min<uint64_t>((b.budget - room)/b.book_bytes, desk);
        if (next == desk || !by_floors) {
            desk = next;
            break;
        }
        desk = next;
    }

    const uint32_t desk_min = 3*b.n_expert_used;
    if (desk < desk_min) {
        lay.error = format("--moe-stream-room: the reading room (%.2f GiB) leaves the desk %u slots per floor, fewer "
                        "than the %u (3 x %u books a word reads) reading in on the desk needs; make the room "
                        "smaller or give --moe-stream-cache more", room/GIB, desk, desk_min, b.n_expert_used);
        return lay;
    }

    lay.desk_slots = desk;
    lay.desk_bytes = (uint64_t) desk*b.book_bytes;

    const uint32_t missing = b.n_expert - desk; // >= 1: the desk only shrinks from desk_slots < n_expert

    lay.room_bytes     = room;
    lay.parts          = req.parts;
    lay.floors         = (double) room/((double) missing*b.stride_max);
    lay.part_bytes_max = (uint64_t) ((missing + req.parts - 1)/req.parts)*b.stride_max;
    lay.n_records_max  = (uint32_t) std::min<uint64_t>(room/b.stride_min, UINT32_MAX);

    if (lay.part_bytes_max > room) {
        lay.error = format("--moe-stream-room: a floor's missing books cut into %d parts make parts of up to %.0f MiB, "
                        "bigger than the whole reading room (%.0f MiB); use more parts or a bigger room",
                        req.parts, lay.part_bytes_max/MIB, room/MIB);
        return lay;
    }
    if (lay.n_records_max > LLAMA_MOE_ROOM_RECORDS_MAX) {
        lay.error = format("--moe-stream-room: the reading room holds %u books of one floor, but the GPU indexes at "
                        "most %u books in one step; make the room smaller", lay.n_records_max, LLAMA_MOE_ROOM_RECORDS_MAX);
        return lay;
    }
    if (lay.floors < 1.0) {
        lay.warning = format("--moe-stream-room: the reading room holds %.2f floors of look-ahead, less than one, so "
                          "the runners cannot get a whole floor ahead of the desk work", lay.floors);
    }

    lay.on = true;
    return lay;
}

std::string llama_moe_room_describe(const llama_moe_room_layout & lay) {
    if (!lay.on) {
        return format("reading room: off; reading in uses waves on the desk (%u slots per floor)", lay.desk_slots);
    }
    return format("reading room: %.2f GiB = %.2f floors of look-ahead, each floor in %d part%s (up to %.0f MiB); "
               "desk %.1f GiB (%u slots per floor); reading in of %u+ tokens uses the room, shorter uses waves",
               lay.room_bytes/GIB, lay.floors, lay.parts, lay.parts == 1 ? "" : "s", lay.part_bytes_max/MIB,
               lay.desk_bytes/GIB, lay.desk_slots, lay.sweep_min_tokens);
}

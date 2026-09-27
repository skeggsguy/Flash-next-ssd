#include "llama-ple-shelf.h"

#include "ggml.h"

#include <algorithm>

llama_ple_shelf::llama_ple_shelf(int64_t capacity) {
    GGML_ASSERT(capacity >= 1 && capacity <= INT32_MAX);

    slot_row.assign(capacity, -1);
    ref.assign(capacity, 0);
    stamp.assign(capacity, 0);

    // at least twice the slots, so a probe run stays short at a full shelf
    uint32_t bits = 4;
    while (bits < 32 && (((uint64_t) 1) << bits) < 2 * (uint64_t) capacity) {
        ++bits;
    }
    table.assign(((size_t) 1) << bits, 0);
    shift = 32 - bits;
}

int64_t llama_ple_shelf::locate(int32_t row) const {
    const size_t mask = table.size() - 1;
    for (size_t i = home(row); table[i] != 0; i = (i + 1) & mask) {
        if (slot_row[table[i] - 1] == row) {
            return (int64_t) i;
        }
    }
    return -1;
}

int64_t llama_ple_shelf::find(int32_t row) const {
    const int64_t i = locate(row);
    return i < 0 ? -1 : (int64_t) table[i] - 1;
}

void llama_ple_shelf::insert(int32_t row, int64_t slot) {
    const size_t mask = table.size() - 1;
    size_t i = home(row);
    while (table[i] != 0) {
        i = (i + 1) & mask;
    }
    table[i] = (uint32_t) slot + 1;
}

// backward-shift deletion: an entry further along the run moves into the hole unless its home lies
// cyclically in (hole, entry], where a lookup would no longer reach it from its home
void llama_ple_shelf::erase(int32_t row) {
    const size_t mask = table.size() - 1;
    const int64_t at = locate(row);
    GGML_ASSERT(at >= 0);

    size_t i = (size_t) at;
    for (size_t j = (i + 1) & mask; table[j] != 0; j = (j + 1) & mask) {
        const size_t k = home(slot_row[table[j] - 1]);
        const bool stays = i <= j ? (i < k && k <= j) : (i < k || k <= j);
        if (!stays) {
            table[i] = table[j];
            i = j;
        }
    }
    table[i] = 0;
}

int64_t llama_ple_shelf::victim() {
    const int64_t cap = capacity();
    for (;;) {
        const int64_t s = hand;
        hand = hand + 1 < cap ? hand + 1 : 0;
        if (stamp[s] == call || ref[s]) {
            ref[s] = 0; // the call's own slots are passed like a set bit, and never taken
            continue;
        }
        return s;
    }
}

void llama_ple_shelf::plan(const int32_t * rows, int64_t n, llama_ple_shelf_pick * picks) {
    if (++call == 0) {
        // the stamp wrapped: no slot may look like it belongs to this call
        std::fill(stamp.begin(), stamp.end(), 0);
        call = 1;
    }
    n_call = 0;

    for (int64_t i = 0; i < n; ++i) {
        const int32_t row = rows[i];
        GGML_ASSERT(row >= 0);

        int64_t s = find(row);
        if (s >= 0) {
            GGML_ASSERT(stamp[s] != call && "a call's rows must be distinct");
            ref[s]   = 1;
            picks[i] = { s, LLAMA_PLE_SHELF_HIT };
        } else if (filled < capacity()) {
            s = filled++;
            slot_row[s] = row;
            insert(row, s);
            ref[s]   = 0;
            picks[i] = { s, LLAMA_PLE_SHELF_MISS };
        } else if (n_call < capacity()) {
            s = victim();
            erase(slot_row[s]);
            slot_row[s] = row;
            insert(row, s);
            ref[s]   = 0;
            picks[i] = { s, LLAMA_PLE_SHELF_MISS };
        } else {
            picks[i] = { -1, LLAMA_PLE_SHELF_PASS };
            continue;
        }
        stamp[s] = call;
        ++n_call;
    }
}

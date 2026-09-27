#pragma once

// The phrasebook shelf's bookkeeping (SHARE-PARTS-PLAN.md phase 5): which phrasebook rows sit in which
// slot of a fixed shelf, and which slot a row that is not there goes into. Pure: no file, no memory
// but its own tables, so tests/test-ple-shelf.cpp checks it against an oracle.
//
// The phrasebook (per_layer_tok_embd) is read a few rows a token; left to the file cache, every 90 B row
// brings a 16 KB page into macOS's cache, where it grows the pile (the PS-trace sitting: 74 MiB of rows
// held 10.4 GiB of pages). The shelf keeps the rows themselves, raw, in slots of one row each.
//
// The rule is the simulator's CLOCK with cold entry (~/dev/ai/sim/ple_policies.py clock_hits with
// insert_bit 0), which matched LRU at the knee and beat it below: a hit sets the slot's bit; a miss
// takes an empty slot while there is one, else sweeps the hand, clearing set bits, to the first clear
// one and replaces that row; a new row enters with its bit clear, so a row asked for once goes first.
// One addition the simulator does not need: a call's rows are read and converted in parallel after its
// plan, so a slot that a row of the same call hit or was given is never taken again in that call (the
// hand clears its bit and passes on). When every slot is held by the call itself, the call's remaining
// rows PASS: they are read into scratch and never kept. So a shelf smaller than one batch still gives
// every row its bytes; it just keeps only what fits.

#include <cstdint>
#include <vector>

enum llama_ple_shelf_kind : uint8_t {
    LLAMA_PLE_SHELF_HIT  = 0, // on the shelf: convert it from its slot
    LLAMA_PLE_SHELF_MISS = 1, // read it into its (new) slot, then convert it from there
    LLAMA_PLE_SHELF_PASS = 2, // read it into scratch and convert it; the shelf does not keep it
};

struct llama_ple_shelf_pick {
    int64_t slot = -1; // HIT, MISS: the slot; PASS: -1
    uint8_t kind = LLAMA_PLE_SHELF_PASS;
};

class llama_ple_shelf {
public:
    // capacity: slots (one row each), 1 .. 2^31 - 1
    explicit llama_ple_shelf(int64_t capacity);

    int64_t capacity() const { return (int64_t) slot_row.size(); }
    int64_t held()     const { return filled; }

    // one call: its distinct rows (each row id at most once, >= 0), in the order they are served;
    // picks[i] says where rows[i] is or goes. Updates the shelf as if every MISS had been read.
    void plan(const int32_t * rows, int64_t n, llama_ple_shelf_pick * picks);

    int64_t find(int32_t row) const;       // the slot holding row, or -1
    int32_t row_at(int64_t slot) const { return slot_row[slot]; } // -1 while the slot is empty
    int64_t hand_at() const { return hand; }

private:
    size_t home(int32_t row) const { return (size_t) (((uint32_t) row * 0x9E3779B1u) >> shift); }

    int64_t locate(int32_t row) const;     // the table index holding row, or -1
    void    erase(int32_t row);
    void    insert(int32_t row, int64_t slot);
    int64_t victim(); // the hand's next slot, skipping (and clearing) the call's own slots

    std::vector<int32_t>  slot_row; // row id per slot, -1 empty
    std::vector<uint8_t>  ref;      // CLOCK bit per slot
    std::vector<uint32_t> stamp;    // the call that last hit or filled the slot
    std::vector<uint32_t> table;    // open addressing, linear probing: slot + 1, 0 empty
    uint32_t shift  = 0;            // 32 - log2(table size)
    int64_t  filled = 0;
    int64_t  hand   = 0;
    uint32_t call   = 0;            // the current plan's stamp
    int64_t  n_call = 0;            // slots stamped by the current plan
};

#pragma once

#include <cstddef>
#include <cstdint>

// One missing book across both runners (LLAMA_MOE_STREAM_SPLIT, the study's backlog #11b).
//
// A book the GPU is waiting on (a demand trip) is read as one job per slab (2-3 slabs of ~1 MB), and each slab
// goes to whichever runner the drive rule picks (pick_alt). The book lands when its LAST slab lands, so a book
// whose slabs all queue on one drive waits for that drive alone while the other may idle. The switch:
//   0  today: one job per slab, the runner picked at the read (byte-for-byte the same reads and choices)
//   1  a demand book's slabs are spread across both runners on purpose, alternating slab by slab
//   2  as 1, and each demand slab is cut into two halves read from both runners at once; the cut sits on a
//      4096-byte boundary of the file (the direct reads' block), so the halves never read the same page
// Only demand trips change. The lookahead's guesses, the reading room's reads and a copy back from the lent
// belt keep today's whole-slab jobs (a guess promoted to demand keeps the job it was queued with). The bytes
// are the same file bytes whichever runner reads them (the alt copy is byte-identical), so only the timing can
// change, never the words. Off unless set, for now; with one runner there is nothing to split across.

struct llama_moe_stream;
struct llama_moe_stream_layer;

static const int32_t LLAMA_MOE_SPLIT_MAX = 2;

// LLAMA_MOE_STREAM_SPLIT read at load: unset, empty or "0" -> 0; clamped to [0, LLAMA_MOE_SPLIT_MAX]
int32_t llama_moe_stream_split_env();

// the manager's switch and counters (llama_moe_stream::split), all under the manager's mutex
struct llama_moe_stream_split {
    int32_t  level = 0; // asked for; the queue uses 0 while there is one runner
    uint32_t rr    = 0; // the next demand slab's runner, alternating (level 1 and 2)

    int64_t n_spread = 0; // demand slabs sent to a runner chosen by the alternation, whole
    int64_t n_halved = 0; // demand slabs cut in two halves (two jobs each)
};

// one read job's share of a slab: bytes [offs, offs + len) of it, on runner `drive` (-1: pick_alt at the read)
struct llama_moe_split_piece {
    uint32_t offs  = 0;
    uint32_t len   = 0; // 0 = the whole slab
    int8_t   drive = -1;
};

// The jobs one demand slab becomes at `level`: the slab is nb bytes at file offset file_offs. Writes 1 or 2
// pieces to out and returns how many. Level 0 is one whole-slab piece with no drive (today's job); level 1 one
// whole piece on runner rr & 1; level 2 two pieces split at the 4096-byte file boundary nearest below the
// middle, on runners rr & 1 and the other (a slab with no such boundary strictly inside it splits at its middle;
// a slab under 2 bytes stays whole, as at level 1). Levels 1 and 2 advance rr by one.
int llama_moe_split_slab(int32_t level, size_t file_offs, size_t nb, uint32_t & rr, llama_moe_split_piece out[2]);

// Queues a demand book: the reserved desk slot `slot` of floor sl, book `expert`, one job per slab (or per
// piece, at level 2), each woken separately; slot_pending counts the jobs. `lent`/`save` are the lent belt's
// restore and keep (llama-moe-room-lend-ops.cpp); a restore is a copy from RAM, so it keeps whole slabs and no
// drive. Under mgr.mtx.
void llama_moe_stream_queue_demand_locked(llama_moe_stream & mgr, llama_moe_stream_layer & sl, int32_t expert,
        int32_t slot, uint64_t lent = 0, uint64_t save = 0);

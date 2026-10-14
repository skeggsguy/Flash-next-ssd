// The keep rule (LLAMA_MOE_STREAM_LOOKAHEAD_KEEP=1): a book the lookahead guessed for floor L, one floor ahead
// or two, stays on L's desk until L's own stop, whether L reads it or not.
//
// Why: a guessed book keeps the low route hotness it came with, so without the rule it is the desk's next
// victim, and the next guess to land on the same floor (L-1's one-floor guesses, after L-2's two-floor ones)
// puts it back before L ever reads it. The simulator put that at 59-65% of two-floor guesses (#30 step 4).
//
// The rule, on floor L's desk:
//   - a guess never takes a kept slot (pick_victim_locked with spare_kept); the issue loop stops when no
//     victim is left, as it always has.
//   - L's own need (the remap's demand loads) takes an unkept slot when there is one, and a kept one rather
//     than wait (n_keep_forced). It never waits where the desk without the rule would not: kept slots are
//     RESIDENT or LOADING, and a LOADING one was never a victim, so the second pick is the old rule exactly.
//     Waiting instead would hang the floor when kept books fill its desk (llama-moe-stream-wave-stage.cpp).
//   - every kept slot of L is released at L's stop (llama_moe_stream_remap_out, after the demand loop, for the
//     locked path and the quick path alike), and at a wave's or the reading room's stop on L, which never
//     reach remap_out; the guesses only come from the single-wave remap, so those two only clear leftovers.
//   - reserve_slot_locked forgets a slot's tags, so a reused slot is never kept or counted for a book it no
//     longer holds.
//
// Off (the default without the trained guesser; with it named, on unless "0": llama-flashnext.h), the desk's
// choices are the old ones byte for byte: the tags and counters below are kept
// either way (one-floor and two-floor "used" and "put back unread"), only the victim rule ignores them.
// The GPU slot servicer (LLAMA_MOE_STREAM_GPU_SLOT) chooses victims on the GPU, which knows nothing of kept
// slots, so the rule is off there.

#include "llama-moe-stream.h"

#include "llama-flashnext.h"
#include "llama-impl.h"
#include "llama-moe-stream-guesser.h"

#include <cstdlib>
#include <cstring>

bool llama_moe_stream_lookahead_keep_env() {
    return llama_flashnext_lookahead_keep(std::getenv("LLAMA_MOE_STREAM_LOOKAHEAD_KEEP"), llama_moe_stream_guesser_env() != nullptr);
}

// the victim for a book floor sl itself needs now: unkept first, then a kept one rather than wait. -1 only
// when the old rule gives -1 (every allowed slot is LOADING), and the caller waits as before.
int32_t llama_moe_stream::pick_victim_need_locked(llama_moe_stream_layer & sl, const uint8_t * keep) {
    if (!la_keep) {
        return pick_victim_locked(sl, keep);
    }
    const int32_t v = pick_victim_locked(sl, keep, /*spare_kept =*/ true);
    if (v >= 0) {
        return v;
    }
    const int32_t f = pick_victim_locked(sl, keep);
    if (f >= 0) {
        stats.n_keep_forced++; // only kept slots were left (or LOADING ones, never victims): take one
    }
    return f;
}

// a guess has just reserved `slot` for `book`: tag it, so its floor can count it read, and keep it until the
// floor stops
void llama_moe_stream::keep_mark_locked(llama_moe_stream_layer & sl, int32_t slot, int32_t book, bool two_ahead) {
    if (sl.slot_kept.size() != (size_t) sl.n_slots) {
        return; // a layer made by hand (tests) without the keep state
    }
    if (two_ahead) {
        sl.slot_la2[slot] = book;
        stats.n_la2_issued++;
    } else {
        sl.slot_la1[slot] = book;
        stats.n_la1_issued++;
    }
    if (!sl.slot_kept[slot]) {
        sl.n_slot_kept++;
    }
    sl.slot_kept[slot] = two_ahead ? 2 : 1;
}

// `slot` is about to hold another book: a guess still kept there is put back unread before its floor stopped
void llama_moe_stream::keep_forget_locked(llama_moe_stream_layer & sl, int32_t slot) {
    if (sl.slot_kept.size() != (size_t) sl.n_slots) {
        return;
    }
    if (sl.slot_kept[slot]) {
        if (sl.slot_kept[slot] == 2) {
            stats.n_la2_put_back++;
        } else {
            stats.n_la1_put_back++;
        }
        sl.slot_kept[slot] = 0;
        sl.n_slot_kept--;
    }
    sl.slot_la1[slot] = -1;
    sl.slot_la2[slot] = -1;
}

void llama_moe_stream_keep_release(llama_moe_stream_layer & sl) {
    if (sl.n_slot_kept == 0) {
        return; // the common case: nothing guessed for this floor since its last stop
    }
    for (auto & k : sl.slot_kept) {
        k = 0;
    }
    sl.n_slot_kept = 0;
}

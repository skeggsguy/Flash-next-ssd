#include "llama-moe-stream.h"
#include "llama-moe-stream-impl.h"
#include "llama-moe-room.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// The lookahead: the remap op for floor L also starts loading floor L+1's predicted books
// (llama_moe_stream_lookahead in llama-moe-stream-layer.h). Moved out of llama-moe-stream-remap.cpp.

// Prefetch the next layer's predicted experts. Never waits and never evicts anything this call
// needs - a wrong guess costs one slab read, which the drive has headroom for (decode uses ~1.2 of
// ~6.9 GB/s). Called from the remap op, so it adds no graph split of its own. With the lent belt
// (llama-moe-room-lend-ops.cpp) a guess keeps the book it puts back and is copied back from the belt
// when its book is there; the guesses' keeps are copied out by the next floor's remap op, or by the
// guess's own worker at its gate, whichever comes first.
// the top_k predicted books of one token's row of next-floor logits, most likely first
static void llama_moe_stream_lookahead_top(llama_moe_stream_lookahead * la, const float * logits,
        std::vector<int32_t> & picks, uint32_t n, const std::vector<float> & bias, uint32_t top_k) {
    picks.clear();
    la->score.resize(n);
    for (uint32_t e = 0; e < n; e++) {
        // same selection rule as build_moe_ffn for this arch: sqrt(softplus(x)) then the bias
        const float x = logits[e];
        float p = x > 20.0f ? x : log1pf(expf(x));   // softplus, guarded for large x
        p = sqrtf(p);
        la->score[e] = p + (bias.empty() ? 0.0f : bias[e]);
    }

    for (uint32_t k = 0; k < top_k; k++) {
        uint32_t best = 0;
        float    bv   = -INFINITY;
        for (uint32_t e = 0; e < n; e++) {
            if (la->score[e] > bv) { bv = la->score[e]; best = e; }
        }
        la->score[best] = -INFINITY;   // consume
        picks.push_back((int32_t) best);
    }
}

// start loading the listed books in order; stops when the backlog is full or no slot is free
static void llama_moe_stream_lookahead_issue(llama_moe_stream_layer & sl, const std::vector<int32_t> & books,
        bool two_ahead = false) {
    auto * mgr = sl.mgr;

    llama_moe_room * lender = mgr->room && mgr->room->lend_on ? mgr->room.get() : nullptr;
    uint32_t n_kept = 0;

    for (const int32_t best : books) {
        if (sl.expert_slot.find((int32_t) best) != sl.expert_slot.end()) {
            continue;                  // already resident or in flight - the common case
        }
        if (mgr->q_spec.size() >= mgr->q_spec_max) {
            break;                     // backlog already deeper than the drive will clear in time
        }
        const int32_t v = mgr->pick_victim_locked(sl, nullptr);
        if (v < 0) {
            break;                     // every slot busy; the layer's own remap will demand-load it
        }
        const uint64_t lent = lender ? lender->lend_find_locked(sl, (int32_t) best, true) : 0; // pinned before any keep
        const uint64_t save = lender ? lender->lend_keep_locked(sl, v, n_kept) : 0;             // before reserve forgets the old book
        mgr->reserve_slot_locked(sl, (int32_t) best, v);
        sl.slot_pending[v] = (uint8_t) sl.weights.size();
        for (size_t wi = 0; wi < sl.weights.size(); wi++) {
            llama_moe_stream_work w = { &sl, (int32_t) best, v, (int32_t) wi, sl.slot_gen[v] };
            w.lent = lent;
            w.save = save;
            mgr->q_spec.push_back(w);
            mgr->cv_work.notify_one();
        }
        mgr->stats.n_preload_issued++;
        if (two_ahead) {
            if (sl.slot_la2.size() != (size_t) sl.n_slots) {
                sl.slot_la2.assign((size_t) sl.n_slots, -1);
            }
            sl.slot_la2[v] = best;
            mgr->stats.n_la2_issued++;
        }
    }
}

// the next floor's guesses for the batch's last token
static void llama_moe_stream_prefetch_next(llama_moe_stream_lookahead * la, const float * logits, std::vector<int32_t> & books) {
    llama_moe_stream_lookahead_top(la, logits, books, la->sl_next->n_expert, la->bias, la->top_k);
}

// LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2=K: floor L also fetches floor L+2's K most likely books for the batch's
// last token, predicted by floor L+2's router on floor L's input (one more small GEMM in the graph, joined
// to L+1's logits). Two floors of attention and FFN are skipped, so the guess is weaker than L+1's and K is
// kept small; it queues after L+1's guesses. Each slot it fills remembers the book, and the floor's remap
// counts the ones it reads ("lookahead 2 floors" in the stats), so the guess's accuracy is measured.
uint32_t llama_moe_stream_lookahead_depth2_env() {
    const char * s = getenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2");
    return s != nullptr ? (uint32_t) std::max(0, atoi(s)) : 0;
}

// the rows are [n1 + n2] per token (L+1's logits, then L+2's); with LLAMA_MOE_STREAM_LOOKAHEAD_ALL a small
// batch (the apprentice's check) fetches two floors ahead for every token, rank by rank from the last,
// each book once, as prefetch_next_all does one floor ahead; otherwise the last token only
static void llama_moe_stream_prefetch_next2(llama_moe_stream_lookahead * la, const float * rows, int64_t n_tok,
        int64_t stride, int64_t n1, std::vector<int32_t> & books) {
    if (!la->bias2_read) {
        la->bias2_read = true;
        if (la->bias_src2) {
            la->bias2.resize(ggml_nelements(la->bias_src2));
            ggml_backend_tensor_get(la->bias_src2, la->bias2.data(), 0, ggml_nbytes(la->bias_src2));
        }
    }
    const uint32_t n2 = la->sl_next2->n_expert;
    const int64_t first = (n_tok > 1 && n_tok <= 16 && la->all) ? 0 : n_tok - 1;
    std::vector<std::vector<int32_t>> picks((size_t) n_tok);
    for (int64_t t = first; t < n_tok; t++) {
        llama_moe_stream_lookahead_top(la, rows + t*stride + n1, picks[(size_t) t], n2, la->bias2, la->top_k2);
    }
    std::vector<uint8_t> seen(n2, 0);
    books.clear();
    for (uint32_t k = 0; k < la->top_k2; k++) {
        for (int64_t t = n_tok - 1; t >= first; t--) {
            const int32_t e = picks[(size_t) t][k];
            if (!seen[e]) {
                seen[e] = 1;
                books.push_back(e);
            }
        }
    }
}

// LLAMA_MOE_STREAM_LOOKAHEAD_ALL=1: a small batch (the apprentice's check, 2..16 tokens) prefetches the
// next floor's predicted books for every token, not only the last: rank 1 of each token (the last
// token first), then rank 2, and so on, each book once. Unset: the last token only, as before.
uint32_t llama_moe_stream_lookahead_all_ranks_env() {
    const char * s = getenv("LLAMA_MOE_STREAM_LOOKAHEAD_ALL_RANKS");
    return s != nullptr ? (uint32_t) std::max(0, atoi(s)) : 0;
}

bool llama_moe_stream_lookahead_all_env() {
    const char * s = getenv("LLAMA_MOE_STREAM_LOOKAHEAD_ALL");
    return s != nullptr && *s != '\0' && strcmp(s, "0") != 0;
}

static void llama_moe_stream_prefetch_next_all(llama_moe_stream_lookahead * la, const float * rows,
        int64_t n_tok, int64_t stride, std::vector<int32_t> & books) {
    std::vector<std::vector<int32_t>> picks((size_t) n_tok);
    for (int64_t t = 0; t < n_tok; t++) {
        llama_moe_stream_lookahead_top(la, rows + t*stride, picks[(size_t) t], la->sl_next->n_expert, la->bias, la->top_k);
    }
    std::vector<uint8_t> seen(la->sl_next->n_expert, 0);
    books.clear();
    const uint32_t ranks = la->all_ranks > 0 ? std::min(la->all_ranks, la->top_k) : la->top_k;
    for (uint32_t k = 0; k < la->top_k; k++) {
        for (int64_t t = n_tok - 1; t >= 0; t--) {
            if (t != n_tok - 1 && k >= ranks) {
                continue; // an earlier token: only its most likely books
            }
            const int32_t e = picks[(size_t) t][k];
            if (!seen[e]) {
                seen[e] = 1;
                books.push_back(e);
            }
        }
    }
}

// every guess already on the desk or on its way: the issue loop would skip them all
static bool llama_moe_stream_lookahead_all_on_desk(const llama_moe_stream_layer & sl, const std::vector<int32_t> & books) {
    for (const int32_t e : books) {
        if (sl.expert_slot.find(e) == sl.expert_slot.end()) {
            return false;
        }
    }
    return true;
}

void llama_moe_stream_remap_la(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata) {
    auto * la = (llama_moe_stream_lookahead *) userdata;

    llama_moe_stream_remap(dst, a, ith, nth, la->sl);

    if (ith != 0 || la->sl_next == nullptr || la->top_k == 0) {
        return;
    }

    // b is [n_expert, n_tokens] of predicted next-layer logits (with depth 2, [2*n_expert, n_tokens]: the
    // next layer's, then the one after's); use the last token's row, which is the one whose routing the
    // next layer will actually resolve first
    const int64_t n_tok = b->ne[1] > 0 ? b->ne[1] : 1;
    const float * logits = (const float *) b->data + (n_tok - 1)*b->ne[0];

    if (!la->bias_read) {
        la->bias_read = true;
        if (la->bias_src) {
            la->bias.resize(ggml_nelements(la->bias_src));
            ggml_backend_tensor_get(la->bias_src, la->bias.data(), 0, ggml_nbytes(la->bias_src));
        }
    }

    // The guesses depend only on the logits, never on the desk, so with LLAMA_MOE_STREAM_NOLOCK they are picked
    // before the lock, and when every one is already on the desk or on its way (the issue loop would skip them
    // all) the lock is not taken at all (llama-moe-stream-quick.cpp says why reading the desk's index here is
    // safe). Off: the lock first, as before.
    auto * mgr = la->sl_next->mgr;
    std::unique_lock<std::mutex> lk(mgr->mtx, std::defer_lock);
    if (!mgr->nolock) {
        lk.lock();
        if (mgr->load_failed) {
            return;
        }
    }
    std::vector<int32_t> books1, books2;
    if (n_tok > 1 && n_tok <= 16 && la->all) {
        llama_moe_stream_prefetch_next_all(la, (const float *) b->data, n_tok, b->ne[0], books1);
    } else {
        llama_moe_stream_prefetch_next(la, logits, books1);
    }
    const int64_t n1 = la->sl_next->n_expert;
    const bool two = la->sl_next2 && la->top_k2 > 0 && b->ne[0] == n1 + la->sl_next2->n_expert;
    if (two) {
        llama_moe_stream_prefetch_next2(la, (const float *) b->data, n_tok, b->ne[0], n1, books2);
    }
    if (mgr->nolock) {
        if (llama_moe_stream_lookahead_all_on_desk(*la->sl_next, books1) &&
                (!two || llama_moe_stream_lookahead_all_on_desk(*la->sl_next2, books2))) {
            mgr->n_la_quick++;
            return;
        }
        lk.lock();
        if (mgr->load_failed) {
            return;
        }
    }
    llama_moe_stream_lookahead_issue(*la->sl_next, books1);
    if (two) {
        llama_moe_stream_lookahead_issue(*la->sl_next2, books2, true);
    }
}

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
        std::vector<int32_t> & picks) {
    const uint32_t n = la->sl_next->n_expert;

    picks.clear();
    la->score.resize(n);
    for (uint32_t e = 0; e < n; e++) {
        // same selection rule as build_moe_ffn for this arch: sqrt(softplus(x)) then the bias
        const float x = logits[e];
        float p = x > 20.0f ? x : log1pf(expf(x));   // softplus, guarded for large x
        p = sqrtf(p);
        la->score[e] = p + (la->bias.empty() ? 0.0f : la->bias[e]);
    }

    for (uint32_t k = 0; k < la->top_k; k++) {
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
static void llama_moe_stream_lookahead_issue(llama_moe_stream_lookahead * la, const std::vector<int32_t> & books) {
    llama_moe_stream_layer & sl = *la->sl_next;
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
    }
}

static void llama_moe_stream_prefetch_next(llama_moe_stream_lookahead * la, const float * logits) {
    std::vector<int32_t> picks;
    llama_moe_stream_lookahead_top(la, logits, picks);
    llama_moe_stream_lookahead_issue(la, picks);
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
        int64_t n_tok, int64_t stride) {
    std::vector<std::vector<int32_t>> picks((size_t) n_tok);
    for (int64_t t = 0; t < n_tok; t++) {
        llama_moe_stream_lookahead_top(la, rows + t*stride, picks[(size_t) t]);
    }
    std::vector<uint8_t> seen(la->sl_next->n_expert, 0);
    std::vector<int32_t> books;
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
    llama_moe_stream_lookahead_issue(la, books);
}

void llama_moe_stream_remap_la(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata) {
    auto * la = (llama_moe_stream_lookahead *) userdata;

    llama_moe_stream_remap(dst, a, ith, nth, la->sl);

    if (ith != 0 || la->sl_next == nullptr || la->top_k == 0) {
        return;
    }

    // b is [n_expert, n_tokens] of predicted next-layer logits; use the last token's row, which is
    // the one whose routing the next layer will actually resolve first
    const int64_t n_tok = b->ne[1] > 0 ? b->ne[1] : 1;
    const float * logits = (const float *) b->data + (n_tok - 1)*b->ne[0];

    if (!la->bias_read) {
        la->bias_read = true;
        if (la->bias_src) {
            la->bias.resize(ggml_nelements(la->bias_src));
            ggml_backend_tensor_get(la->bias_src, la->bias.data(), 0, ggml_nbytes(la->bias_src));
        }
    }

    std::unique_lock<std::mutex> lk(la->sl_next->mgr->mtx);
    if (!la->sl_next->mgr->load_failed) {
        if (n_tok > 1 && n_tok <= 16 && la->all) {
            llama_moe_stream_prefetch_next_all(la, (const float *) b->data, n_tok, b->ne[0]);
        } else {
            llama_moe_stream_prefetch_next(la, logits);
        }
    }
}

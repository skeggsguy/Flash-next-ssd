#pragma once

// The ceiling tax: write only the undo copies a rollback can reach (CEILING-TAX-PLAN.md, #ceiling-tax).
//
// The 36 floors that keep a fixed summary (the recurrent state: the delta-net state and its conv history)
// keep one undo copy of it per possible rollback (rollback snapshots): 1 + n_rs_seq copies, n_rs_seq being
// the apprentice's ceiling (its most drafted words a round). Slot r holds the summary as it was r tokens
// before the round's end, so the check's rollback of r rejected words reads slot r.
//
// Today every round writes all of them. A round of n tokens (the draft plus the library's own word)
// produces slots 0..min(n, K)-1 (K = 1 + n_rs_seq); the slots past it are filled with the round's start
// state ("best-effort": exact for a rollback of exactly n, a wrong state for anything deeper), and
// qwen4exp's conv histories write every slot too. A higher ceiling therefore makes every round dearer,
// even a round with no draft at all: ~3.5 MiB a floor a slot (C-tax-probe, +9.5 ms a round at ceiling 10
// against 5).
//
// The only rollback llama-server makes is the check's, right after a round, and it rolls back at most the
// rejected words: r <= n - 1. With the switch on a round writes only slots 0..min(n, K)-1, and the memory
// remembers how far back this seq's last round can roll (its reach, min(n - 1, n_rs_seq)); a rollback past
// the reach is refused without touching the memory (the caller falls back to its bookmark copy, as it does
// past n_rs_seq today). A slot that is skipped is a slot nothing reads.
//
// Not reading a slot is not enough on Metal, so qwen4exp's delta-net conv history (cache_r_l) keeps every
// slot with the switch on too. Its copies are small (~120 KiB a floor a slot, a twenty-fifth of the
// state's), but without them Metal's graph order (ggml_graph_optimize) packs a short round's nodes
// differently, its fused kernels round differently, and the real model's words drift after ~80 tokens
// (gate-prose at n3, 2026-10-10: 46750edd against off's 05da8e0c; on and off agree with fusion off, or
// with this history's slots kept). The PLE conv history's slots and the state's clamp still skip.
//
// LLAMA_RS_SKIP_SLOTS (read when the context is made, so per context): unset or "1" skips (on by default from
// 2026-10-15, the fork's proven defaults, llama-flashnext.h); anything else, "0" included, is the old graph and
// memory exactly.

#include <cstdint>

// LLAMA_RS_SKIP_SLOTS's value (nullptr: unset) as on/off: unset and "1" are on
bool llama_rs_skip_slots_parse(const char * value);

// the context's setting, with its start-up line (at WARN, from every context, with its n_rs_seq)
bool llama_rs_skip_slots_init(const char * value, uint32_t n_rs_seq);

// how many undo slots (0..n-1) a round of n_seq_tokens writes, out of K = 1 + n_rs_seq
inline int64_t llama_rs_slots_written(bool skip, int64_t n_seq_tokens, int64_t K) {
    if (!skip) {
        return K;
    }
    return n_seq_tokens < K ? n_seq_tokens : K;
}

// the deepest rollback a round of n_seq_tokens leaves exact with the switch on: slot r is written for
// r < min(n, K), so r <= min(n - 1, n_rs_seq)
inline uint32_t llama_rs_reach(uint32_t n_seq_tokens, uint32_t n_rs_seq) {
    const uint32_t deepest = n_seq_tokens > 0 ? n_seq_tokens - 1 : 0;
    return deepest < n_rs_seq ? deepest : n_rs_seq;
}

// whether a rollback of `rollback` tokens may proceed: 1..n_rs_seq today, and with the switch on also no
// deeper than the last round's reach
inline bool llama_rs_rollback_ok(bool skip, uint32_t rollback, uint32_t reach, uint32_t n_rs_seq) {
    if (rollback < 1 || rollback > n_rs_seq) {
        return false;
    }
    return !skip || rollback <= reach;
}

#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

// The state log (LLAMA_MOE_STREAM_STATE_LOG=<path>): at every floor stop of a single-wave call (writing, and a
// read-in short enough to take one wave), one record per word per floor of what the floor's router saw and said,
// plus the shared expert's output. It is the training data for a guesser that reads floor n-2's router input and
// scores floor n's books (TRAINED-LOOKAHEAD-PLAN.md in the study). Unset: no file and no extra graph node, so the
// words are byte-identical; set, the floor's shared expert runs before its stop (its output is an input of the
// stop's op) and nothing else moves.
//
// File layout, little-endian, all sizes in the header so a reader needs nothing else:
//
//     header (header_bytes = 40):
//         char magic[4] = "MSSL", u32 version = 1, u32 header_bytes, u32 record_bytes,
//         u32 n_floor  (the book manager's floors: the model's layers, the apprentice's floor too when loaded),
//         u32 n_slip   (books a slip names: n_expert_used),
//         u32 n_cur    (the router input's width: n_embd),
//         u32 n_scores (the router's scores: n_expert),
//         u32 n_shexp  (the shared expert's output width: n_embd),
//         u32 value_type = 1 (f16 for cur, scores and shexp)
//     then one record of record_bytes per word per floor:
//         u32 call     the decode call this word belongs to, counted from 0 in this file
//         u32 n_tokens the call's words (1 = a written word; more = a short read-in or the apprentice's check)
//         u16 row      the word's index in its call
//         u8  floor
//         u8  zero
//         i16 slip[n_slip]       the books the router picked, in its order
//         f16 cur[n_cur]         the router input (hc_mixed)
//         f16 scores[n_scores]   the router's raw scores (logits, before the softmax; its top n_slip are the slip)
//         f16 shexp[n_shexp]     the shared expert's output, gated (what the floor adds beside its books)
//
// A floor's records are written in row order, floors in the order they run, so the records of one call are
// contiguous. The last floor of a read-in sees only the rows that produce output, so it is not logged when it
// sees fewer rows than its call has; writing (every row an output) logs every floor. The graph's warmup pass
// (a slip of every book) is skipped. A server killed mid-run leaves a truncated last record: drop it. A write that
// fails (the drive full) closes the log and says so once in the server log; what was written before stays readable.
struct llama_moe_stream_statelog {
    static constexpr uint32_t VERSION      = 1;
    static constexpr uint32_t HEADER_BYTES = 40;
    static constexpr uint32_t RECORD_HEAD  = 12; // call, n_tokens, row, floor, zero

    llama_moe_stream_statelog(std::string path, uint32_t n_floor);
    ~llama_moe_stream_statelog(); // closes the file and says what it wrote

    // one floor stop: ids I32 [n_slip, n_rows], scores F32 [n_scores, n_rows], cur and shexp F32 [n, n_rows];
    // rows contiguous (nb[0] = 4), any row stride. n_slip is the routing width (the manager's n_expert_used,
    // known only after the manager is made); a stop with ids of another width is skipped. Thread-safe.
    void put(int32_t il, uint32_t n_slip, const ggml_tensor * ids, const ggml_tensor * scores,
            const ggml_tensor * cur, const ggml_tensor * shexp);

    // bytes of one record for these widths
    static size_t record_bytes(uint32_t n_slip, uint32_t n_cur, uint32_t n_scores, uint32_t n_shexp);

    const std::string path;
    const uint32_t    n_floor;

    // counters, for the closing line and the tests
    int64_t n_records = 0;
    int64_t n_calls   = 0;
    int64_t n_skipped = 0; // stops whose slip was not n_slip wide (the warmup sweep) or whose shapes disagree
    bool    failed    = false; // could not open, or a write failed (the drive full): off from then on, said once

private:
    std::mutex mtx;
    FILE *     file   = nullptr;
    uint32_t   n_slip = 0, n_cur = 0, n_scores = 0, n_shexp = 0; // fixed by the first record
    int64_t    call    = -1;
    int32_t    last_il = INT32_MAX;
    int64_t    since_flush = 0;
    std::vector<uint8_t> rec;    // scratch: one record

    bool open_locked(uint32_t n_slip, uint32_t n_cur, uint32_t n_scores, uint32_t n_shexp);
    void fail_locked(const char * what); // a write failed: close, say so once, and stay off
};

// LLAMA_MOE_STREAM_STATE_LOG, read where the book manager reads its other switches; empty = off
const char * llama_moe_stream_state_log_env();

// the stop's op when the state log is on: src[0] the slip (I32 ids), src[1] the router's scores, src[2] the router
// input, src[3] the shared expert's output, src[4] the lookahead's logits or absent. Logs, then does what the
// stop does without the log (llama_moe_stream_remap_la, or llama_moe_stream_remap). userdata is the floor's
// llama_moe_stream_layer.
void llama_moe_stream_remap_log(ggml_tensor * dst, int ith, int nth, void * userdata);

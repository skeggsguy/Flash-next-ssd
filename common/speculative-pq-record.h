#pragma once

// The p/q record: LLAMA_SPEC_PQ_RECORD=<path> (unset, "" or "0": off, no file). FAIR-CHECK-PLAN step 1 (#41).
//
// What the fair check (keep a drafted word with chance min(1, p/q)) would be priced from: per round, the
// library's own chances p at each checked position and DFlash 2's own chances q (its selector's lattice).
// Its own file, apart from the rate trace (speculative-rate-trace.h), because it holds token ids (numbers
// only, never text); the study keeps the files out of git (`*.pqr`). A recording switch: off by default, and
// with it on not one word changes (tests/test-speculative-pq-record-mtp.cpp, the DFlash SAME gates):
//   - the draft side (the DFlash 2 branch of common/speculative.cpp) copies the lattice rows of every
//     drafted position before the next draft decode reuses the buffer: 16 candidate ids and the 16 x 16
//     score table (row j: the scores when the previous position took candidate j), and the walk's words;
//   - the check side is `accept_n`, the library's accept loop (common_sampler_sample_and_accept_n,
//     grammar_first false) call for call, which after each draw reads the chain's candidates
//     (common_sampler_get_candidates: post-chain p, it only sorts; the next draw overwrites them) and the
//     sampler's draw state (a grammar applying, its words refused among the candidates, a grammar-first
//     redraw, the reasoning budget's state). Past a miss, the positions the check decoded but never drew
//     are read from a clone of the sampler that took the drafted word instead (`x`; their p follow the
//     draft, the clone's draws are thrown away), unless a grammar applies;
//   - a word written with no draft (`plain`) records its one position the same way;
//   - the draft side, MTP head (the MTP branch of common/speculative.cpp, FAIR-CHECK-PLAN step 3b): every position
//     the head decoded, its own chances q, the softmax of its logits over the whole vocabulary at temperature 1 (its
//     top TOP_Q; not the draft sampler's top-k 10, renormalised), the word it drafted there (the sampler's top) and
//     that word's share of the sampler's 10 (`conf`, what p-min and the chain stop cut on). The position a cut
//     stopped at was decoded, so it is written too; the drafted words follow the greedy picks, so q at a position is
//     conditioned on the drafted words before it.
// The server's hooks are thin: accept_n in place of the library's loop, check_end once it knows whether the
// check restored a checkpoint (a restore is checked again as a `replay`), plain after a plain draw, and
// sampling when a slot takes a request.
//
// Lines (JSON, one per line, appended):
//   {"record":"spec-pq","version":1,"n_seq":..,"max_cand":..,"wall_us":..}           a session opens
//   {"answer":A,"seq":s,"n_prompt":N,"wall_us":..,<the request's sampling>}          common_speculative_begin
//   {"line":"check"|"plain","i":..,"answer":A,"seq":s,"pos":P,"replay":0|1,"restore":0|1,"wall_us":..,
//    "n":drafted words checked,"acc":words the check returned,"kept":..,"draft":[ids],"words":[ids],
//    "p":[[drawn,g,b,refused,size,id,p,id,p,..],..],"x":[[-1,g,b,refused,size,id,p,..],..],
//    "lat":{"pos0":P,"k":16,"walk":[ids],"ids":[[16 ids],..],"s":[[256 scores],..]}
//    | "mtp":{"pos0":P,"cap":C,"walk":[ids],"q":[[pick,conf,id,q,id,q,..],..]}}
//   {"end":"spec-pq","lines":..,"orphans":..,"record_us":..}                           the record closes
// `pos` is the anchor's position (the word the round drafts from, the draft's pos0); p[i] is the chain's
// chances for the word at pos + 1 + i. g: 1 a grammar applies, 2 it is lazy, 4 this draw was redone grammar
// first; b: the reasoning budget's state (-1 none). `kept` is the rate trace's count for the round a check
// closes (a replay counts one less, as the server does); a restoring check's `kept` is what it kept before
// the restore. `lat` rides on the check or plain line of the draft's own position; a draft no line took
// (another position, or overwritten) counts as an orphan; so does `mtp`, whose `cap` is the depth the draft was
// allowed (0: the measured depth's shadow guess, drafted and never checked) and `q` one row a decoded position.
// Read by the study's `sim/pq_record.py`.

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct common_sampler;
struct common_speculative;
struct common_dflash2_lattice;
struct common_params_sampling;

class common_spec_pq_record {
public:
    static constexpr int MAX_CAND = 32; // candidates written a position (the library's top-k 20 fits)
    static constexpr int TOP_Q    = 20; // the MTP head's chances written a position

    // one drawn (or read) position
    struct position {
        llama_token drawn   = -1; // -1: read past a miss (no draw kept)
        int         g       = 0;
        int         b       = -1;
        int         refused = 0;
        int         size    = 0;  // candidates the chain left (only the first MAX_CAND are written)
        std::vector<std::pair<llama_token, float>> cand;
    };

    // nullptr when the variable is unset, empty or "0"; a path that cannot be opened is reported once and
    // gives nullptr, so a typo never stops the server
    static std::unique_ptr<common_spec_pq_record> open_from_env(const char * env_name, uint32_t n_seq);

    common_spec_pq_record(FILE * f, uint32_t n_seq, int64_t wall_us);
    ~common_spec_pq_record(); // writes the closing line and closes

    common_spec_pq_record(const common_spec_pq_record &) = delete;
    common_spec_pq_record & operator=(const common_spec_pq_record &) = delete;

    // a slot took a request (its sampling, written on the next answer line) / an answer begins
    void sampling(int seq, const common_params_sampling & params);
    void answer_begin(int seq, int n_prompt);

    // the draft side: copies the lattice's rows and the walk's words (before the trim)
    void draft(int seq, int pos0, const common_dflash2_lattice & lat, const std::vector<llama_token> & words);

    // the draft side, MTP: a draft from `pos0` begins; the head decoded a position (its logits at `idx` of ctx_dft;
    // `pick` the word the sampler put first, `conf` its share of the sampler's candidates); the draft ends, `cap` its
    // allowed depth and `words` what it drafted (before the server's trim)
    void mtp_begin(int seq, int pos0);
    void mtp_step(int seq, llama_context * ctx_dft, int idx, llama_token pick, float conf);
    void mtp_end(int seq, int cap, const std::vector<llama_token> & words);

    // the top k (id, chance) of the softmax of n logits, highest first (ties: the lower id first); a word more than
    // SKIP below the best is left out of the sum (each under e^-30 of the best's, so the chances are short by at most
    // n e^-30 of the total)
    static constexpr float SKIP = 30.0f;
    static std::vector<std::pair<llama_token, float>> softmax_top(const float * logits, int n, int k);

    // the check side: the library's accept loop, recorded; `pos` is the anchor's position
    std::vector<llama_token> accept_n(common_sampler * smpl, llama_context * ctx, const std::vector<int32_t> & idxs,
            const std::vector<llama_token> & draft, int seq, int pos, bool replay);
    void check_end(int seq, bool restore); // writes the check's line

    // a word written with no draft, right after its draw
    void plain(int seq, int pos, common_sampler * smpl, llama_token id);

    // what a draw left in the sampler; r0: its resample count before the draw
    static position read_position(common_sampler * smpl, llama_token drawn, uint64_t r0);

    int64_t n_written() const { return n_lines; }
    int64_t n_orphans() const { return orphans; }
    int64_t self_us()   const { return t_self_us; }

private:
    struct pending_draft {
        bool                     active = false;
        int                      pos0   = 0;
        int                      k      = 0;
        int                      n_pos  = 0;
        std::vector<float>       rows;  // n_pos x (k ids + k x k scores)
        std::vector<llama_token> walk;
        bool                     mtp = false; // an MTP draft: the rows below, not the lattice's
        int                      cap = 0;
        std::vector<llama_token> pick;        // a decoded position each
        std::vector<float>       conf;
        std::vector<std::vector<std::pair<llama_token, float>>> q;
    };
    struct pending_check {
        bool                     active  = false;
        int                      pos     = 0;
        bool                     replay  = false;
        std::vector<llama_token> draft;
        std::vector<llama_token> words;
        std::vector<position>    p;
        std::vector<position>    x;
    };

    void read_past_miss(common_sampler * smpl, llama_context * ctx, const std::vector<int32_t> & idxs,
            const std::vector<llama_token> & draft, size_t miss, pending_check & pc) const;
    void write_line(const char * kind, int seq, const pending_check & pc, bool restore, int kept);
    static void add_draft(std::string & s, const pending_draft & d); // the draft's "lat" or "mtp" field
    void put(const std::string & line);
    bool valid(int seq) const { return seq >= 0 && seq < (int) drafts.size(); }

    std::mutex                 mtx;
    FILE *                     f;
    std::vector<pending_draft> drafts;    // [n_seq]
    std::vector<pending_check> checks;    // [n_seq]
    std::vector<int64_t>       answer_of; // [n_seq]
    std::vector<std::string>   sampling_of; // [n_seq] the request's sampling, as JSON fields
    std::vector<uint64_t>      resample_seen; // [n_seq] the sampler's resample count at its last recorded draw
    int64_t                    n_answers = 0;
    int64_t                    n_lines   = 0;
    int64_t                    orphans   = 0;
    int64_t                    t_self_us = 0;
};

// the speculative wrapper's record (nullptr when off, or with no wrapper)
common_spec_pq_record * common_speculative_pq_record(const common_speculative * spec);

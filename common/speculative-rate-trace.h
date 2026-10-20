#pragma once

// The draft's checking-cycle trace: LLAMA_SPEC_RATE_TRACE=<path> (unset, "" or "0": off, no file).
//
// One JSON line per checking cycle, numbers only: the ceiling in force, draft steps run, guesses drafted /
// sent to be checked / kept, each step's top confidence, the confidences past the stop (`tail`), why the
// draft stopped, the position it drafted from, whether the kept count came from the server's replay of a
// draft cut short, and three steady-clock times from the start of the draft call: the draft's end, the
// accept, and the next draft call of the same answer (the whole cycle; -1 when the answer ended first). No
// token ids and no text, so a trace cannot be turned back into words; the study keeps trace files out of
// git (`*.srtr`).
//
// Two writers, never both on one file: today's apprentice (draft-mtp, draft-mtp-adaptive) keeps its own
// hooks in its draft loop; any other draft is traced by the speculative wrapper (common_speculative_draft),
// which opens the line before the draft call and closes it after the wrapper's trim, so "checked" is what
// went to the check. Such a draft only reports each word's confidence and its stop (the impl's note_*
// helpers); the wrapper's guard opens no trace when an impl writes its own.
//
// A header line opens each session (the file is appended to), and an answer line opens each answer
// (`common_speculative_begin`), carrying the prompt length and the wall clock so a replay can join the
// answers to a rung's steps by time. A closing line (`"end"`) gives the cycles written and the time the
// trace itself took. Read by the study's `sim/spec_trace.py`, which reads version 1 too.
// The header's `rate_mode` names the depth rule: 0 fixed or hand-tuned, 1 the measured depth
// (LLAMA_SPEC_ADAPTIVE_RATE=1), 2 the priced depth (=2, speculative-rate2.h), whose stops are priced, not
// p-min. Version 2 adds `draft` (the draft type) and `block` (a block draft's trained block, else 0).
//
// While the C1 phase log is on (GGML_METAL_CBLOG, common/spec-phase-log.h) each cycle line also carries the
// log's `"round"`, so a cycle joins its phase lines; with that log off the line is unchanged.
//
// Every hook is one null check when the trace is off.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

struct common_speculative_rate_trace {
    static constexpr int MAX_STEPS = 16; // the deepest draft any setting runs (common_speculative_rate::K_MAX)

    // why a draft stopped: today's apprentice derives it from what it ran (stop_of); other drafts report it.
    // chain: the chain stop (CHAIN-PLAN step 2); trim: the wrapper cut the draft to the server's ceiling
    enum stop_reason { STOP_CAP, STOP_PMIN, STOP_ERROR, STOP_CHAIN, STOP_TRIM };

    struct cycle {
        bool    active    = false;
        int64_t index     = 0;   // cycle number in this session
        int64_t answer    = -1;  // answer the cycle belongs to
        int64_t round     = -1;  // the C1 phase log's round (common/spec-phase-log.h); written only when >= 0
        int     pos       = 0;   // position the draft started from
        int     cap       = 0;   // the ceiling in force (0: a measured-depth shadow cycle, nothing checked)
        int     drafted   = 0;   // guesses that passed the cut
        int     checked   = 0;   // guesses sent to the library to check
        int     kept      = 0;   // guesses the library kept
        int     stop      = -1;  // a reported stop_reason; -1: derived from steps, drafted and the ceiling
        bool    replay    = false; // kept came from the server's replay of a draft cut short (kept = n - 1)
        bool    ended     = false; // draft_end seen
        int64_t t0_us     = 0;
        int64_t draft_us  = -1;
        int64_t accept_us = -1;
        std::vector<float> conf; // top confidence of each step run, the one the cut dropped included
        std::vector<float> tail; // confidences past the stop, walked for the trace only (nothing drafted)
    };

    // nullptr when the variable is unset, empty or "0"; otherwise a trace appending to the path (a path
    // that cannot be opened is reported once and gives nullptr, so a typo never stops the server)
    static std::unique_ptr<common_speculative_rate_trace> open_from_env(
            const char * env_name, uint32_t n_seq, int n_max, float p_min, bool adaptive, int rate_mode,
            const std::string & draft = "", int block = 0);

    common_speculative_rate_trace(FILE * f, uint32_t n_seq, int n_max, float p_min, bool adaptive, int rate_mode,
            int64_t wall_us, const std::string & draft = "", int block = 0);
    ~common_speculative_rate_trace(); // writes every unfinished cycle (cycle_us -1), the closing line, and closes

    common_speculative_rate_trace(const common_speculative_rate_trace &) = delete;
    common_speculative_rate_trace & operator=(const common_speculative_rate_trace &) = delete;

    static int64_t now_us();      // steady clock
    static int64_t wall_now_us(); // system clock, for joining answers to a rung's steps
    static stop_reason stop_of(int steps, int drafted, int cap);
    static const char * stop_name(stop_reason r);
    static std::string format_cycle(const cycle & c, int seq, int64_t cycle_us);

    // hooks, in the order a cycle sees them
    void answer_begin(int seq, int n_prompt, int64_t now, int64_t wall);
    void draft_begin(int seq, int pos, int64_t now, int64_t round = -1);
    void draft_step(int seq, float conf);
    void draft_end(int seq, int cap, int drafted, int checked, int64_t now);
    void accepted(int seq, int n_accepted, int64_t now, bool replay = false);

    // the wrapper's hooks for a draft that reports (between draft_begin and draft_close)
    void draft_restart(int seq);                 // the next draft type tries: forget what the last one reported
    void draft_tail(int seq, float conf);        // a confidence past the stop
    void draft_stop(int seq, stop_reason why);   // why the draft stopped
    // closes the draft: n_result words before the wrapper's trim, n_checked after it. drafted is derived from
    // the reported steps and stop (a draft that reported nothing: n_result), and a trim that cut words is
    // the stop
    void draft_close(int seq, int cap, int n_result, int n_checked, int64_t now);

    void add_self_us(int64_t us) { t_self_us += us; } // time the trace's own work took outside this file

    int64_t n_written() const { return n_cycles_written; }
    int64_t self_us()   const { return t_self_us; }

private:
    void finish(int seq, int64_t cycle_us);
    void put(const std::string & line);
    bool valid(int seq) const { return seq >= 0 && seq < (int) open.size(); }

    FILE *             f;
    std::vector<cycle> open; // [n_seq] the cycle each sequence is in
    std::vector<int64_t> answer_of; // [n_seq]
    int64_t            n_answers        = 0;
    int64_t            n_cycles         = 0;
    int64_t            n_cycles_written = 0;
    int64_t            t_self_us        = 0; // time spent formatting and writing lines, plus add_self_us
};

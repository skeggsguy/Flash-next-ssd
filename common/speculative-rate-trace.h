#pragma once

// The apprentice's checking-cycle trace: LLAMA_SPEC_RATE_TRACE=<path> (unset, "" or "0": off, no file).
//
// One JSON line per checking cycle of the MTP draft (draft-mtp and draft-mtp-adaptive), numbers only:
// the ceiling in force, draft steps run, guesses drafted / sent to be checked / kept, each step's top
// confidence, why the draft stopped, the position it drafted from, and three steady-clock times from
// the start of the draft call: the draft's end, the accept, and the next draft call of the same answer
// (the whole cycle; -1 when the answer ended first). No token ids and no text, so a trace cannot be
// turned back into words; the study still keeps trace files out of git (`*.srtr`).
//
// A header line opens each session (the file is appended to), and an answer line opens each answer
// (`common_speculative_begin`), carrying the prompt length and the wall clock so a replay can join the
// answers to a rung's steps by time. Read by the study's `sim/spec_replay.py`.
//
// The hooks in common/speculative.cpp are one null check each when the trace is off.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

struct common_speculative_rate_trace {
    static constexpr int MAX_STEPS = 16; // the deepest draft any setting runs (common_speculative_rate::K_MAX)

    // why a draft stopped, derived from what it ran
    enum stop_reason { STOP_CAP, STOP_PMIN, STOP_ERROR };

    struct cycle {
        bool    active    = false;
        int64_t index     = 0;   // cycle number in this session
        int64_t answer    = -1;  // answer the cycle belongs to
        int     pos       = 0;   // position the draft started from
        int     cap       = 0;   // the ceiling in force (0: a measured-depth shadow cycle, nothing checked)
        int     drafted   = 0;   // guesses that passed p-min
        int     checked   = 0;   // guesses sent to the library to check
        int     kept      = 0;   // guesses the library kept
        bool    ended     = false; // draft_end seen
        int64_t t0_us     = 0;
        int64_t draft_us  = -1;
        int64_t accept_us = -1;
        std::vector<float> conf; // top confidence of each step run, the one p-min dropped included
    };

    // nullptr when the variable is unset, empty or "0"; otherwise a trace appending to the path (a path
    // that cannot be opened is reported once and gives nullptr, so a typo never stops the server)
    static std::unique_ptr<common_speculative_rate_trace> open_from_env(
            const char * env_name, uint32_t n_seq, int n_max, float p_min, bool adaptive, bool rate_mode);

    common_speculative_rate_trace(FILE * f, uint32_t n_seq, int n_max, float p_min, bool adaptive, bool rate_mode,
            int64_t wall_us);
    ~common_speculative_rate_trace(); // writes every unfinished cycle (cycle_us -1) and closes the file

    common_speculative_rate_trace(const common_speculative_rate_trace &) = delete;
    common_speculative_rate_trace & operator=(const common_speculative_rate_trace &) = delete;

    static int64_t now_us();      // steady clock
    static int64_t wall_now_us(); // system clock, for joining answers to a rung's steps
    static stop_reason stop_of(int steps, int drafted, int cap);
    static const char * stop_name(stop_reason r);
    static std::string format_cycle(const cycle & c, int seq, int64_t cycle_us);

    // hooks, in the order a cycle sees them
    void answer_begin(int seq, int n_prompt, int64_t now, int64_t wall);
    void draft_begin(int seq, int pos, int64_t now);
    void draft_step(int seq, float conf);
    void draft_end(int seq, int cap, int drafted, int checked, int64_t now);
    void accepted(int seq, int n_accepted, int64_t now);

    int64_t n_written() const { return n_cycles_written; }

private:
    void finish(int seq, int64_t cycle_us);
    void put(const std::string & line);

    FILE *             f;
    std::vector<cycle> open; // [n_seq] the cycle each sequence is in
    std::vector<int64_t> answer_of; // [n_seq]
    int64_t            n_answers        = 0;
    int64_t            n_cycles         = 0;
    int64_t            n_cycles_written = 0;
};

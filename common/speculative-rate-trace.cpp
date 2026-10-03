#include "speculative-rate-trace.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>

std::unique_ptr<common_speculative_rate_trace> common_speculative_rate_trace::open_from_env(
        const char * env_name, uint32_t n_seq, int n_max, float p_min, bool adaptive, int rate_mode) {
    const char * path = std::getenv(env_name);
    if (path == nullptr || *path == '\0' || std::strcmp(path, "0") == 0) {
        return nullptr;
    }
    FILE * f = std::fopen(path, "a");
    if (f == nullptr) {
        fprintf(stderr, "%s: cannot open '%s' (%s); no apprentice trace\n", env_name, path, std::strerror(errno));
        return nullptr;
    }
    fprintf(stderr, "apprentice trace: every checking cycle to %s (%s)\n", path, env_name);
    return std::make_unique<common_speculative_rate_trace>(f, n_seq, n_max, p_min, adaptive, rate_mode, wall_now_us());
}

common_speculative_rate_trace::common_speculative_rate_trace(FILE * f, uint32_t n_seq, int n_max, float p_min,
        bool adaptive, int rate_mode, int64_t wall_us)
    : f(f), open(n_seq), answer_of(n_seq, -1) {
    char buf[256];
    snprintf(buf, sizeof(buf),
            "{\"trace\":\"spec-rate\",\"version\":1,\"n_seq\":%u,\"n_max\":%d,\"p_min\":%.4f,"
            "\"adaptive\":%d,\"rate_mode\":%d,\"wall_us\":%" PRId64 "}\n",
            n_seq, n_max, (double) p_min, adaptive ? 1 : 0, rate_mode, wall_us);
    put(buf);
}

common_speculative_rate_trace::~common_speculative_rate_trace() {
    for (int s = 0; s < (int) open.size(); s++) {
        finish(s, -1);
    }
    if (f != nullptr) {
        std::fclose(f);
    }
}

int64_t common_speculative_rate_trace::now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

int64_t common_speculative_rate_trace::wall_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

// the draft loop stops a sequence on a guess under p-min (that step ran, its guess was dropped), on
// reaching its ceiling, or on a failed decode; an end token never stops it
common_speculative_rate_trace::stop_reason common_speculative_rate_trace::stop_of(int steps, int drafted, int cap) {
    if (steps > drafted) {
        return STOP_PMIN;
    }
    if (drafted >= cap && drafted > 0) {
        return STOP_CAP;
    }
    return STOP_ERROR;
}

const char * common_speculative_rate_trace::stop_name(stop_reason r) {
    switch (r) {
        case STOP_CAP:  return "cap";
        case STOP_PMIN: return "pmin";
        default:        return "error";
    }
}

std::string common_speculative_rate_trace::format_cycle(const cycle & c, int seq, int64_t cycle_us) {
    const int steps = (int) c.conf.size();
    // a shadow cycle (cap 0) drafts one guess at a ceiling of one
    const stop_reason why = stop_of(steps, c.drafted, std::max(1, c.cap));
    std::string line;
    line.reserve(256 + 8 * c.conf.size());
    char buf[320];
    snprintf(buf, sizeof(buf),
            "{\"cycle\":%" PRId64 ",\"answer\":%" PRId64 ",\"seq\":%d,",
            c.index, c.answer, seq);
    line += buf;
    if (c.round >= 0) {
        snprintf(buf, sizeof(buf), "\"round\":%" PRId64 ",", c.round);
        line += buf;
    }
    snprintf(buf, sizeof(buf),
            "\"pos\":%d,\"cap\":%d,\"steps\":%d,"
            "\"drafted\":%d,\"checked\":%d,\"kept\":%d,\"stop\":\"%s\",\"conf\":[",
            c.pos, c.cap, steps, c.drafted, c.checked, c.kept,
            c.ended ? stop_name(why) : "error");
    line += buf;
    for (int i = 0; i < steps; i++) {
        snprintf(buf, sizeof(buf), i == 0 ? "%.4f" : ",%.4f", (double) c.conf[i]);
        line += buf;
    }
    snprintf(buf, sizeof(buf), "],\"draft_us\":%" PRId64 ",\"accept_us\":%" PRId64 ",\"cycle_us\":%" PRId64 "}\n",
            c.draft_us, c.accept_us, cycle_us);
    line += buf;
    return line;
}

void common_speculative_rate_trace::put(const std::string & line) {
    if (f == nullptr) {
        return;
    }
    // flushed per line: a server stopped by a signal loses at most the cycle in flight (~2 us a cycle of
    // ~50-250 ms, and only with the trace on)
    std::fwrite(line.data(), 1, line.size(), f);
    std::fflush(f);
}

void common_speculative_rate_trace::finish(int seq, int64_t cycle_us) {
    cycle & c = open[seq];
    if (!c.active) {
        return;
    }
    put(format_cycle(c, seq, cycle_us));
    n_cycles_written++;
    c = cycle();
}

void common_speculative_rate_trace::answer_begin(int seq, int n_prompt, int64_t now, int64_t wall) {
    if (seq < 0 || seq >= (int) open.size()) {
        return;
    }
    finish(seq, -1); // the answer before ended after its last check: no next draft call to time to
    answer_of[seq] = n_answers++;
    char buf[200];
    snprintf(buf, sizeof(buf),
            "{\"answer\":%" PRId64 ",\"seq\":%d,\"n_prompt\":%d,\"t_us\":%" PRId64 ",\"wall_us\":%" PRId64 "}\n",
            answer_of[seq], seq, n_prompt, now, wall);
    put(buf);
}

void common_speculative_rate_trace::draft_begin(int seq, int pos, int64_t now, int64_t round) {
    if (seq < 0 || seq >= (int) open.size()) {
        return;
    }
    if (open[seq].active) {
        finish(seq, now - open[seq].t0_us);
    }
    cycle & c = open[seq];
    c.active = true;
    c.index  = n_cycles++;
    c.answer = answer_of[seq];
    c.round  = round;
    c.pos    = pos;
    c.t0_us  = now;
    c.conf.reserve(MAX_STEPS);
}

void common_speculative_rate_trace::draft_step(int seq, float conf) {
    if (seq < 0 || seq >= (int) open.size() || !open[seq].active || (int) open[seq].conf.size() >= MAX_STEPS) {
        return;
    }
    open[seq].conf.push_back(conf);
}

void common_speculative_rate_trace::draft_end(int seq, int cap, int drafted, int checked, int64_t now) {
    if (seq < 0 || seq >= (int) open.size() || !open[seq].active) {
        return;
    }
    cycle & c = open[seq];
    c.cap      = cap;
    c.drafted  = drafted;
    c.checked  = checked;
    c.ended    = true;
    c.draft_us = now - c.t0_us;
}

void common_speculative_rate_trace::accepted(int seq, int n_accepted, int64_t now) {
    if (seq < 0 || seq >= (int) open.size() || !open[seq].active) {
        return;
    }
    cycle & c = open[seq];
    c.kept      = n_accepted; // as reported: a replay reads kept > checked as a hook fault
    c.accept_us = now - c.t0_us;
}

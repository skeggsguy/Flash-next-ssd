#include "speculative-rate-trace.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>

std::unique_ptr<common_speculative_rate_trace> common_speculative_rate_trace::open_from_env(
        const char * env_name, uint32_t n_seq, int n_max, float p_min, bool adaptive, int rate_mode,
        const std::string & draft, int block, double chain_p) {
    const char * path = std::getenv(env_name);
    if (path == nullptr || *path == '\0' || std::strcmp(path, "0") == 0) {
        return nullptr;
    }
    FILE * f = std::fopen(path, "a");
    if (f == nullptr) {
        fprintf(stderr, "%s: cannot open '%s' (%s); no apprentice trace\n", env_name, path, std::strerror(errno));
        return nullptr;
    }
    fprintf(stderr, "apprentice trace: every checking cycle to %s (%s, %s)\n", path, env_name,
            draft.empty() ? "draft" : draft.c_str());
    return std::make_unique<common_speculative_rate_trace>(f, n_seq, n_max, p_min, adaptive, rate_mode,
            wall_now_us(), draft, block, chain_p);
}

// the shortest text that reads back as exactly `v` (a threshold typed as "0.3" prints as 0.3, not 0.29999999999999999)
static std::string shortest_double(double v) {
    char buf[32];
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(buf, sizeof(buf), "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v) {
            break;
        }
    }
    return buf;
}

common_speculative_rate_trace::common_speculative_rate_trace(FILE * f, uint32_t n_seq, int n_max, float p_min,
        bool adaptive, int rate_mode, int64_t wall_us, const std::string & draft, int block, double chain_p)
    : f(f), open(n_seq), answer_of(n_seq, -1) {
    std::string name;
    for (char ch : draft) { // a type name; anything a JSON string would need escaped is dropped
        if (ch >= ' ' && ch != '"' && ch != '\\') {
            name += ch;
        }
    }
    char buf[384];
    snprintf(buf, sizeof(buf),
            "{\"trace\":\"spec-rate\",\"version\":2,\"n_seq\":%u,\"n_max\":%d,\"p_min\":%.4f,"
            "\"adaptive\":%d,\"rate_mode\":%d,\"draft\":\"%.64s\",\"block\":%d,\"chain_p\":%s,"
            "\"wall_us\":%" PRId64 "}\n",
            n_seq, n_max, (double) p_min, adaptive ? 1 : 0, rate_mode, name.c_str(), block, shortest_double(chain_p).c_str(), wall_us);
    put(buf);
}

common_speculative_rate_trace::~common_speculative_rate_trace() {
    for (int s = 0; s < (int) open.size(); s++) {
        finish(s, -1);
    }
    char buf[160];
    snprintf(buf, sizeof(buf), "{\"end\":\"spec-rate\",\"cycles\":%" PRId64 ",\"trace_us\":%" PRId64 "}\n",
            n_cycles_written, t_self_us);
    put(buf);
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

// the apprentice's draft loop stops a sequence on a guess under p-min (that step ran, its guess was
// dropped), on reaching its ceiling, or on a failed decode; an end token never stops it
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
        case STOP_CAP:   return "cap";
        case STOP_PMIN:  return "pmin";
        case STOP_CHAIN: return "chain";
        case STOP_TRIM:  return "trim";
        default:         return "error";
    }
}

static void put_floats(std::string & line, const std::vector<float> & v) {
    char buf[32];
    for (size_t i = 0; i < v.size(); i++) {
        snprintf(buf, sizeof(buf), i == 0 ? "%.9g" : ",%.9g", (double) v[i]); // a float read back exactly
        line += buf;
    }
}

std::string common_speculative_rate_trace::format_cycle(const cycle & c, int seq, int64_t cycle_us) {
    const int steps = (int) c.conf.size();
    // a shadow cycle (cap 0) drafts one guess at a ceiling of one
    const stop_reason why = c.stop >= 0 ? (stop_reason) c.stop : stop_of(steps, c.drafted, std::max(1, c.cap));
    std::string line;
    line.reserve(256 + 16 * (c.conf.size() + c.tail.size()));
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
    put_floats(line, c.conf);
    line += "],\"tail\":[";
    put_floats(line, c.tail);
    snprintf(buf, sizeof(buf), "],\"replay\":%d,\"draft_us\":%" PRId64 ",\"accept_us\":%" PRId64
            ",\"cycle_us\":%" PRId64 "}\n", c.replay ? 1 : 0, c.draft_us, c.accept_us, cycle_us);
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
    const int64_t t0 = now_us();
    put(format_cycle(c, seq, cycle_us));
    n_cycles_written++;
    c = cycle();
    t_self_us += now_us() - t0;
}

void common_speculative_rate_trace::answer_begin(int seq, int n_prompt, int64_t now, int64_t wall) {
    if (!valid(seq)) {
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
    if (!valid(seq)) {
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
    if (!valid(seq) || !open[seq].active || (int) open[seq].conf.size() >= MAX_STEPS) {
        return;
    }
    open[seq].conf.push_back(conf);
}

void common_speculative_rate_trace::draft_tail(int seq, float conf) {
    if (!valid(seq) || !open[seq].active || (int) open[seq].tail.size() >= MAX_STEPS) {
        return;
    }
    open[seq].tail.push_back(conf);
}

void common_speculative_rate_trace::draft_restart(int seq) {
    if (!valid(seq) || !open[seq].active) {
        return;
    }
    open[seq].conf.clear();
    open[seq].tail.clear();
    open[seq].stop = -1;
}

void common_speculative_rate_trace::draft_stop(int seq, stop_reason why) {
    if (!valid(seq) || !open[seq].active) {
        return;
    }
    open[seq].stop = (int) why;
}

void common_speculative_rate_trace::draft_end(int seq, int cap, int drafted, int checked, int64_t now) {
    if (!valid(seq) || !open[seq].active) {
        return;
    }
    cycle & c = open[seq];
    c.cap      = cap;
    c.drafted  = drafted;
    c.checked  = checked;
    c.ended    = true;
    c.draft_us = now - c.t0_us;
}

void common_speculative_rate_trace::draft_close(int seq, int cap, int n_result, int n_checked, int64_t now) {
    if (!valid(seq) || !open[seq].active) {
        return;
    }
    cycle & c = open[seq];
    const int steps = (int) c.conf.size();
    int drafted = n_result;
    if (c.stop >= 0) {
        // the word the cut dropped was walked, not drafted
        drafted = steps - (c.stop == STOP_PMIN || c.stop == STOP_CHAIN ? 1 : 0);
    } else if (steps > 0) {
        c.stop  = STOP_ERROR; // it reported its words but no stop: it broke off
        drafted = steps;
    } else {
        c.stop  = n_result > 0 ? STOP_CAP : STOP_ERROR; // a draft that reports nothing
    }
    if (n_checked < n_result) {
        c.stop = STOP_TRIM;
    }
    draft_end(seq, cap, drafted, n_checked, now);
}

void common_speculative_rate_trace::accepted(int seq, int n_accepted, int64_t now, bool replay) {
    if (!valid(seq) || !open[seq].active) {
        return;
    }
    cycle & c = open[seq];
    // as reported (a replay reads kept > checked as a hook fault); the server's replay re-checks the kept
    // words plus its own, so it counts one less, as the server's n_accepted-- does
    c.kept      = replay && n_accepted > 0 ? n_accepted - 1 : n_accepted;
    c.replay    = replay;
    c.accept_us = now - c.t0_us;
}

// The draft's checking-cycle trace (common/speculative-rate-trace.h): off means no file, and on it
// writes one line per cycle with the numbers a replay needs, each cycle timed to the next draft call of the
// same answer and the last cycle of an answer left open (-1). Scripted sessions with injected clocks, read
// back line by line. Version 2 (CHAIN-PLAN step 1): the header names the draft and its block, confidences
// print at %.9g, a reporting draft passes its stop in (chain and trim are new), the tail holds the
// confidences past the stop, the server's replay counts one word less, and a closing line gives the trace's
// own time.

#include "speculative-rate-trace.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_failures++; } } while (0)

using trace_t = common_speculative_rate_trace;

static std::string tmp_path(const char * tag) {
    const char * dir = std::getenv("TMPDIR");
    return std::string(dir && *dir ? dir : "/tmp") + "/test-spec-rate-trace-" + tag + "-" +
           std::to_string((long) getpid()) + ".srtr";
}

static std::vector<std::string> read_lines(const std::string & path) {
    std::vector<std::string> out;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        out.push_back(line);
    }
    return out;
}

static bool has(const std::string & line, const std::string & part) {
    return line.find(part) != std::string::npos;
}

static bool file_exists(const std::string & path) {
    return access(path.c_str(), F_OK) == 0;
}

// off: unset, empty and "0" give no trace and create no file; a path appends a header per opening
static void test_env() {
    const std::string path = tmp_path("env");
    std::remove(path.c_str());
    const char * var = "TEST_SPEC_RATE_TRACE";

    unsetenv(var);
    CHECK(trace_t::open_from_env(var, 1, 3, 0.3f, false, false) == nullptr, "unset must give no trace");
    setenv(var, "", 1);
    CHECK(trace_t::open_from_env(var, 1, 3, 0.3f, false, false) == nullptr, "empty must give no trace");
    setenv(var, "0", 1);
    CHECK(trace_t::open_from_env(var, 1, 3, 0.3f, false, false) == nullptr, "\"0\" must give no trace");
    CHECK(!file_exists("0"), "\"0\" must not create a file named 0");
    setenv(var, "/nonexistent-dir-for-trace/x.srtr", 1);
    CHECK(trace_t::open_from_env(var, 1, 3, 0.3f, false, false) == nullptr, "an unopenable path gives no trace");

    setenv(var, path.c_str(), 1);
    { auto t = trace_t::open_from_env(var, 1, 8, 0.0f, false, false); CHECK(t != nullptr, "a path must trace"); }
    { auto t = trace_t::open_from_env(var, 2, 5, 0.3f, true, true);   CHECK(t != nullptr, "a path must trace"); }
    { auto t = trace_t::open_from_env(var, 1, 8, 0.3f, true, 2);      CHECK(t != nullptr, "a path must trace"); }
    { auto t = trace_t::open_from_env(var, 1, 7, 0.0f, false, 0, "draft-dflash", 8); CHECK(t != nullptr, "a path must trace"); }
    { auto t = trace_t::open_from_env(var, 1, 7, 0.0f, false, 0, "draft-dflash", 8, 0.3); CHECK(t != nullptr, "a path must trace"); }
    { auto t = trace_t::open_from_env(var, 1, 5, 0.3f, true, 1, "draft-mtp", 0, 1.0 / 3.0); CHECK(t != nullptr, "a path must trace"); }
    const auto lines = read_lines(path);
    CHECK(lines.size() == 12, "six openings append a header and a closing line each, got %zu lines", lines.size());
    if (lines.size() == 12) {
        CHECK(has(lines[4], "\"n_max\":8") && has(lines[4], "\"rate_mode\":2"), "header 3 (RATE=2): %s", lines[4].c_str());
        CHECK(has(lines[0], "{\"trace\":\"spec-rate\",\"version\":2,") && has(lines[0], "\"n_max\":8") &&
              has(lines[0], "\"p_min\":0.0000") && has(lines[0], "\"rate_mode\":0") &&
              has(lines[0], "\"draft\":\"\",\"block\":0,"), "header 1: %s", lines[0].c_str());
        CHECK(lines[1] == "{\"end\":\"spec-rate\",\"cycles\":0,\"trace_us\":0}", "closing line 1: %s", lines[1].c_str());
        CHECK(has(lines[2], "\"n_seq\":2") && has(lines[2], "\"n_max\":5") && has(lines[2], "\"p_min\":0.3000") &&
              has(lines[2], "\"adaptive\":1") && has(lines[2], "\"rate_mode\":1"), "header 2: %s", lines[2].c_str());
        CHECK(has(lines[6], "\"n_max\":7,") && has(lines[6], "\"draft\":\"draft-dflash\",\"block\":8,\"chain_p\":0,\"wall_us\":"),
              "header 4 (a block draft, the chain stop off): %s", lines[6].c_str());
        CHECK(has(lines[8], "\"block\":8,\"chain_p\":0.3,\"wall_us\":"), "header 5 (the chain stop at 0.3): %s",
              lines[8].c_str());
        CHECK(has(lines[10], "\"chain_p\":0.3333333333333333,"), "header 6 (a threshold that needs 16 digits to read "
              "back exactly): %s", lines[10].c_str());
    }
    unsetenv(var);
    std::remove(path.c_str());
}

static void test_stop_of() {
    CHECK(trace_t::stop_of(3, 3, 3) == trace_t::STOP_CAP,  "3 steps, 3 drafted at ceiling 3: cap");
    CHECK(trace_t::stop_of(2, 1, 3) == trace_t::STOP_PMIN, "2 steps, 1 drafted: p-min dropped the second");
    CHECK(trace_t::stop_of(1, 0, 8) == trace_t::STOP_PMIN, "an empty draft is a p-min stop at the first step");
    CHECK(trace_t::stop_of(2, 2, 8) == trace_t::STOP_ERROR, "stopped short of the ceiling with no drop: error");
    CHECK(trace_t::stop_of(0, 0, 1) == trace_t::STOP_ERROR, "no step ran: error");
}

// one sequence, two answers: a full draft, a p-min stop, an empty draft (never accepted), a shadow cycle,
// then the answer ends and the destructor writes the last open cycle
static void test_session() {
    const std::string path = tmp_path("session");
    std::remove(path.c_str());
    FILE * f = std::fopen(path.c_str(), "a");
    CHECK(f != nullptr, "open %s", path.c_str());
    if (!f) { return; }
    {
        trace_t t(f, 1, 3, 0.3f, true, true, 1000);
        t.answer_begin(0, 17000, 5000, 1790000000000000);

        // cycle 0: ceiling 3, three steps all over p-min, two kept; draft 40 us, accept 90 us, cycle 120 us
        t.draft_begin(0, 17000, 10000);
        t.draft_step(0, 0.91f); t.draft_step(0, 0.8f); t.draft_step(0, 0.55f);
        t.draft_end(0, 3, 3, 3, 10040);
        t.accepted(0, 2, 10090);

        // cycle 1: the second step falls under p-min; one checked, none kept
        t.draft_begin(0, 17003, 10120);
        t.draft_step(0, 0.7f); t.draft_step(0, 0.1f);
        t.draft_end(0, 3, 1, 1, 10150);
        t.accepted(0, 0, 10200);

        // cycle 2: empty draft, so the server checks nothing and never calls accept
        t.draft_begin(0, 17004, 10260);
        t.draft_step(0, 0.2f);
        t.draft_end(0, 3, 0, 0, 10270);

        // cycle 3: a measured-depth shadow cycle: one guess drafted, nothing checked
        t.draft_begin(0, 17005, 10300);
        t.draft_step(0, 0.95f);
        t.draft_end(0, 0, 1, 0, 10320);
        t.accepted(0, 0, 10330); // the server does not, but a stray accept must not invent kept guesses

        // a new answer: cycle 3 ends with the old answer, its whole-cycle time unknown
        t.answer_begin(0, 18000, 20000, 1790000000100000);
        t.draft_begin(0, 18000, 20100);
        t.draft_step(0, 0.99f);
        t.draft_end(0, 3, 1, 1, 20110); // a decode that failed after one step at ceiling 3
        t.accepted(0, 1, 20150);
        CHECK(t.n_written() == 4, "4 cycles written before the destructor, got %lld", (long long) t.n_written());
    }
    const auto l = read_lines(path);
    CHECK(l.size() == 9, "header + 2 answers + 5 cycles + the closing line = 9 lines, got %zu", l.size());
    if (l.size() != 9) { std::remove(path.c_str()); return; }

    CHECK(has(l[1], "{\"answer\":0,\"seq\":0,\"n_prompt\":17000,\"t_us\":5000,\"wall_us\":1790000000000000}"), "%s", l[1].c_str());
    CHECK(l[2] == "{\"cycle\":0,\"answer\":0,\"seq\":0,\"pos\":17000,\"cap\":3,\"steps\":3,\"drafted\":3,\"checked\":3,"
                  "\"kept\":2,\"stop\":\"cap\",\"conf\":[0.910000026,0.800000012,0.550000012],\"tail\":[],\"replay\":0,"
                  "\"draft_us\":40,\"accept_us\":90,\"cycle_us\":120}",
          "cycle 0: %s", l[2].c_str());
    CHECK(has(l[3], "\"cycle\":1,") && has(l[3], "\"steps\":2,\"drafted\":1,\"checked\":1,\"kept\":0,\"stop\":\"pmin\"") &&
          has(l[3], "\"conf\":[0.699999988,0.100000001],\"tail\":[]") && has(l[3], "\"draft_us\":30,\"accept_us\":80,\"cycle_us\":140}"),
          "cycle 1: %s", l[3].c_str());
    CHECK(has(l[4], "\"cycle\":2,") && has(l[4], "\"steps\":1,\"drafted\":0,\"checked\":0,\"kept\":0,\"stop\":\"pmin\"") &&
          has(l[4], "\"accept_us\":-1,\"cycle_us\":40}"), "cycle 2 (empty, never accepted): %s", l[4].c_str());
    CHECK(has(l[5], "\"cycle\":3,\"answer\":0,") && has(l[5], "\"cap\":0,\"steps\":1,\"drafted\":1,\"checked\":0,\"kept\":0,\"stop\":\"cap\"") &&
          has(l[5], "\"draft_us\":20,\"accept_us\":30,\"cycle_us\":-1}"), "cycle 3 (shadow, last of its answer): %s", l[5].c_str());
    CHECK(has(l[6], "{\"answer\":1,\"seq\":0,\"n_prompt\":18000,"), "answer 1: %s", l[6].c_str());
    CHECK(has(l[7], "\"cycle\":4,\"answer\":1,") && has(l[7], "\"stop\":\"error\"") && has(l[7], "\"kept\":1,") &&
          has(l[7], "\"cycle_us\":-1}"), "cycle 4 (destructor): %s", l[7].c_str());
    CHECK(has(l[8], "{\"end\":\"spec-rate\",\"cycles\":5,\"trace_us\":"), "closing line: %s", l[8].c_str());
    std::remove(path.c_str());
}

// two sequences interleaved: each cycle is timed to its own sequence's next draft call, answers numbered
// across sequences, and hooks for a sequence outside the trace are ignored
static void test_two_seqs() {
    const std::string path = tmp_path("seqs");
    std::remove(path.c_str());
    FILE * f = std::fopen(path.c_str(), "a");
    if (!f) { CHECK(false, "open"); return; }
    {
        trace_t t(f, 2, 2, 0.0f, false, false, 0);
        t.answer_begin(1, 100, 0, 0);
        t.answer_begin(0, 200, 0, 0);
        t.draft_begin(0, 200, 1000); t.draft_begin(1, 100, 1000);
        t.draft_step(0, 0.5f); t.draft_step(1, 0.6f); t.draft_step(1, 0.4f);
        t.draft_end(0, 2, 1, 1, 1010); t.draft_end(1, 2, 2, 2, 1010);
        t.accepted(1, 2, 1100); t.accepted(0, 1, 1100);
        t.draft_begin(1, 103, 1500);
        t.draft_begin(0, 202, 1700);
        t.draft_begin(7, 0, 1800); t.draft_step(-1, 0.5f); t.answer_begin(9, 1, 0, 0); // ignored
    }
    const auto l = read_lines(path);
    CHECK(l.size() == 8, "header + 2 answers + 4 cycles + the closing line, got %zu", l.size());
    if (l.size() == 8) {
        CHECK(has(l[1], "{\"answer\":0,\"seq\":1,") && has(l[2], "{\"answer\":1,\"seq\":0,"), "answers numbered in order");
        CHECK(has(l[3], "\"cycle\":1,\"answer\":0,\"seq\":1,") && has(l[3], "\"kept\":2,") && has(l[3], "\"cycle_us\":500}"),
              "seq 1's first cycle ends at its own next draft: %s", l[3].c_str());
        CHECK(has(l[4], "\"cycle\":0,\"answer\":1,\"seq\":0,") && has(l[4], "\"stop\":\"error\"") && has(l[4], "\"cycle_us\":700}"),
              "seq 0's first cycle ends at its own next draft: %s", l[4].c_str());
        CHECK(has(l[5], "\"seq\":0,") && has(l[5], "\"cycle_us\":-1}") && has(l[6], "\"seq\":1,") && has(l[6], "\"cycle_us\":-1}"),
              "open cycles written by the destructor");
    }
    std::remove(path.c_str());
}

// version 2's wrapper hooks, for a draft that reports (a DFlash draft): its stop passed in, the tail, the trim,
// a second draft type's report replacing the first's, and the server's replay counting one word less
static void test_reported_stops() {
    const std::string path = tmp_path("reported");
    std::remove(path.c_str());
    FILE * f = std::fopen(path.c_str(), "a");
    if (!f) { CHECK(false, "open"); return; }
    CHECK(std::string(trace_t::stop_name(trace_t::STOP_CHAIN)) == "chain" &&
          std::string(trace_t::stop_name(trace_t::STOP_TRIM)) == "trim", "the new stop names");
    {
        trace_t t(f, 1, 7, 0.3f, false, 0, 0, "draft-dflash", 8);
        t.answer_begin(0, 100, 0, 0);
        // cycle 0: p-min drops the fourth word; the trace walks three more into the tail
        t.draft_begin(0, 100, 1000);
        for (float p : {0.9f, 0.8f, 0.7f, 0.2f}) { t.draft_step(0, p); }
        for (float p : {0.6f, 0.5f, 0.4f}) { t.draft_tail(0, p); }
        t.draft_stop(0, trace_t::STOP_PMIN);
        t.draft_close(0, 7, 3, 3, 1010);
        t.accepted(0, 2, 1050);
        // cycle 1: the chain stop drops the second word; the server's replay of it keeps nothing (n 0 stays 0)
        t.draft_begin(0, 103, 2000);
        t.draft_step(0, 0.9f); t.draft_step(0, 0.25f);
        t.draft_stop(0, trace_t::STOP_CHAIN);
        t.draft_close(0, 7, 1, 1, 2010);
        t.accepted(0, 0, 2050, /*replay*/ true);
        // cycle 2: seven drafted, the server's ceiling of 2 trims five; a replay accepts 3, so 2 were kept
        t.draft_begin(0, 104, 3000);
        for (int i = 0; i < 7; i++) { t.draft_step(0, 0.5f); }
        t.draft_stop(0, trace_t::STOP_CAP);
        t.draft_close(0, 2, 7, 2, 3010);
        t.accepted(0, 3, 3050, /*replay*/ true);
        // cycle 3: one draft type reports and fails, the next drafts four and reports nothing
        t.draft_begin(0, 107, 4000);
        t.draft_step(0, 0.1f); t.draft_stop(0, trace_t::STOP_PMIN);
        t.draft_restart(0);
        t.draft_close(0, 7, 4, 4, 4010);
        t.accepted(0, 4, 4050);
        // cycle 4: confidences reported, no stop, no words: the draft broke off
        t.draft_begin(0, 112, 5000);
        t.draft_step(0, 0.9f); t.draft_step(0, 0.9f);
        t.draft_close(0, 7, 0, 0, 5010);
        // cycle 5: n_min cleared a draft p-min stopped at two words
        t.draft_begin(0, 112, 6000);
        for (float p : {0.9f, 0.9f, 0.1f}) { t.draft_step(0, p); }
        t.draft_stop(0, trace_t::STOP_PMIN);
        t.draft_close(0, 7, 0, 0, 6010);
        // cycle 6: nothing reported and nothing drafted
        t.draft_begin(0, 112, 7000);
        t.draft_close(0, 7, 0, 0, 7010);
        t.add_self_us(5000);
    }
    const auto l = read_lines(path);
    CHECK(l.size() == 10, "header + answer + 7 cycles + the closing line, got %zu", l.size());
    if (l.size() != 10) { std::remove(path.c_str()); return; }
    CHECK(l[2] == "{\"cycle\":0,\"answer\":0,\"seq\":0,\"pos\":100,\"cap\":7,\"steps\":4,\"drafted\":3,\"checked\":3,"
                  "\"kept\":2,\"stop\":\"pmin\",\"conf\":[0.899999976,0.800000012,0.699999988,0.200000003],"
                  "\"tail\":[0.600000024,0.5,0.400000006],\"replay\":0,\"draft_us\":10,\"accept_us\":50,\"cycle_us\":1000}",
          "cycle 0 (p-min, tail): %s", l[2].c_str());
    CHECK(has(l[3], "\"steps\":2,\"drafted\":1,\"checked\":1,\"kept\":0,\"stop\":\"chain\",\"conf\":[0.899999976,0.25],"
                    "\"tail\":[],\"replay\":1,"), "cycle 1 (chain, a replay of nothing): %s", l[3].c_str());
    CHECK(has(l[4], "\"cap\":2,\"steps\":7,\"drafted\":7,\"checked\":2,\"kept\":2,\"stop\":\"trim\",") &&
          has(l[4], "\"replay\":1,"), "cycle 2 (trim, a replay of 3 keeps 2): %s", l[4].c_str());
    CHECK(has(l[5], "\"steps\":0,\"drafted\":4,\"checked\":4,\"kept\":4,\"stop\":\"cap\",\"conf\":[],\"tail\":[],\"replay\":0,"),
          "cycle 3 (the first type's report forgotten): %s", l[5].c_str());
    CHECK(has(l[6], "\"steps\":2,\"drafted\":2,\"checked\":0,\"kept\":0,\"stop\":\"error\",") && has(l[6], "\"accept_us\":-1,"),
          "cycle 4 (broke off): %s", l[6].c_str());
    CHECK(has(l[7], "\"steps\":3,\"drafted\":2,\"checked\":0,\"kept\":0,\"stop\":\"pmin\","), "cycle 5 (n_min): %s", l[7].c_str());
    CHECK(has(l[8], "\"steps\":0,\"drafted\":0,\"checked\":0,\"kept\":0,\"stop\":\"error\",") && has(l[8], "\"cycle_us\":-1}"),
          "cycle 6 (nothing): %s", l[8].c_str());
    long long cycles = -1, us = -1;
    CHECK(std::sscanf(l[9].c_str(), "{\"end\":\"spec-rate\",\"cycles\":%lld,\"trace_us\":%lld}", &cycles, &us) == 2 &&
          cycles == 7 && us >= 5000, "closing line counts 7 cycles and the time added: %s", l[9].c_str());
    std::remove(path.c_str());
}

int main() {
    test_env();
    test_stop_of();
    test_reported_stops();
    test_session();
    test_two_seqs();
    if (g_failures) {
        fprintf(stderr, "test-speculative-rate-trace: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-speculative-rate-trace: OK\n");
    return 0;
}

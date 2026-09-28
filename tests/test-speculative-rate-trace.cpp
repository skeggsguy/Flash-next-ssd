// The apprentice's checking-cycle trace (common/speculative-rate-trace.h): off means no file, and on it
// writes one line per cycle with the numbers a replay needs, each cycle timed to the next draft call of the
// same answer and the last cycle of an answer left open (-1). Scripted sessions with injected clocks, read
// back line by line.

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
    const auto lines = read_lines(path);
    CHECK(lines.size() == 2, "two openings append two headers, got %zu lines", lines.size());
    if (lines.size() == 2) {
        CHECK(has(lines[0], "\"trace\":\"spec-rate\"") && has(lines[0], "\"n_max\":8") &&
              has(lines[0], "\"p_min\":0.0000") && has(lines[0], "\"rate_mode\":0"), "header 1: %s", lines[0].c_str());
        CHECK(has(lines[1], "\"n_seq\":2") && has(lines[1], "\"n_max\":5") && has(lines[1], "\"p_min\":0.3000") &&
              has(lines[1], "\"adaptive\":1") && has(lines[1], "\"rate_mode\":1"), "header 2: %s", lines[1].c_str());
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
    CHECK(l.size() == 8, "header + 2 answers + 5 cycles = 8 lines, got %zu", l.size());
    if (l.size() != 8) { std::remove(path.c_str()); return; }

    CHECK(has(l[1], "{\"answer\":0,\"seq\":0,\"n_prompt\":17000,\"t_us\":5000,\"wall_us\":1790000000000000}"), "%s", l[1].c_str());
    CHECK(l[2] == "{\"cycle\":0,\"answer\":0,\"seq\":0,\"pos\":17000,\"cap\":3,\"steps\":3,\"drafted\":3,\"checked\":3,"
                  "\"kept\":2,\"stop\":\"cap\",\"conf\":[0.9100,0.8000,0.5500],\"draft_us\":40,\"accept_us\":90,\"cycle_us\":120}",
          "cycle 0: %s", l[2].c_str());
    CHECK(has(l[3], "\"cycle\":1,") && has(l[3], "\"steps\":2,\"drafted\":1,\"checked\":1,\"kept\":0,\"stop\":\"pmin\"") &&
          has(l[3], "\"conf\":[0.7000,0.1000]") && has(l[3], "\"draft_us\":30,\"accept_us\":80,\"cycle_us\":140}"),
          "cycle 1: %s", l[3].c_str());
    CHECK(has(l[4], "\"cycle\":2,") && has(l[4], "\"steps\":1,\"drafted\":0,\"checked\":0,\"kept\":0,\"stop\":\"pmin\"") &&
          has(l[4], "\"accept_us\":-1,\"cycle_us\":40}"), "cycle 2 (empty, never accepted): %s", l[4].c_str());
    CHECK(has(l[5], "\"cycle\":3,\"answer\":0,") && has(l[5], "\"cap\":0,\"steps\":1,\"drafted\":1,\"checked\":0,\"kept\":0,\"stop\":\"cap\"") &&
          has(l[5], "\"draft_us\":20,\"accept_us\":30,\"cycle_us\":-1}"), "cycle 3 (shadow, last of its answer): %s", l[5].c_str());
    CHECK(has(l[6], "{\"answer\":1,\"seq\":0,\"n_prompt\":18000,"), "answer 1: %s", l[6].c_str());
    CHECK(has(l[7], "\"cycle\":4,\"answer\":1,") && has(l[7], "\"stop\":\"error\"") && has(l[7], "\"kept\":1,") &&
          has(l[7], "\"cycle_us\":-1}"), "cycle 4 (destructor): %s", l[7].c_str());
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
    CHECK(l.size() == 7, "header + 2 answers + 4 cycles, got %zu", l.size());
    if (l.size() == 7) {
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

int main() {
    test_env();
    test_stop_of();
    test_session();
    test_two_seqs();
    if (g_failures) {
        fprintf(stderr, "test-speculative-rate-trace: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-speculative-rate-trace: OK\n");
    return 0;
}

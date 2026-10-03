// The C1 phase log (common/spec-phase-log.h): its clock is the Metal timeline's (mach_absolute_time) to within a
// microsecond; N records come out as exactly N lines after the anchor, written at exit by the process's own log;
// off (GGML_METAL_CBLOG unset, or GGML_METAL_CBLOG_SIDECARS=0) means no log, no file and no buffer; rounds move on
// one at a time and round-only phases are dropped outside a round. The `.phases` format is restated here by hand,
// so a writer that drifts drifts away from this test too (the reader is the study's runner/c1_attrib.py).

#include "spec-phase-log.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

static int g_failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_failures++; } } while (0)

static std::string tmp_path(const char * tag) {
    const char * dir = std::getenv("TMPDIR");
    std::string d = dir && *dir ? dir : "/tmp";
    if (d.back() != '/') {
        d += '/';
    }
    return d + "test-spec-phase-log-" + tag + "-" + std::to_string((long) getpid()) + ".txt";
}

static std::vector<std::string> read_lines(const std::string & path) {
    std::vector<std::string> out;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        out.push_back(line);
    }
    return out;
}

static bool exists(const std::string & path) {
    return access(path.c_str(), F_OK) == 0;
}

// runs `body` in a child process that then calls exit(0), so the process log's atexit write runs for real
template <typename F>
static int in_child(F body) {
    fflush(stdout);
    fflush(stderr);
    const pid_t pid = fork();
    if (pid == 0) {
        body();
        exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// the clock is mach_absolute_time in nanoseconds: within 1 us, and never sleep-counting CLOCK_MONOTONIC
static void test_clock() {
#if defined(__APPLE__)
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double best = 1e18;
    for (int i = 0; i < 64; i++) {
        const int64_t a = common_spec_phase_now_ns();
        const int64_t m = (int64_t) ((__uint128_t) mach_absolute_time() * tb.numer / tb.denom);
        const int64_t b = common_spec_phase_now_ns();
        best = std::fmin(best, std::fabs((double) m - 0.5 * (double) (a + b)) + 0.5 * (double) (b - a));
    }
    CHECK(best <= 1000.0, "the phase clock is %.0f ns from mach_absolute_time (at most 1000)", best);
#endif
    const int64_t t0 = common_spec_phase_now_ns();
    const int64_t t1 = common_spec_phase_now_ns();
    CHECK(t1 >= t0, "the clock went back: %" PRId64 " then %" PRId64, t0, t1);
}

static void test_off() {
    const char * cb = "TEST_SPEC_PHASE_CBLOG";
    const char * side = "TEST_SPEC_PHASE_SIDECARS";
    unsetenv(cb);
    unsetenv(side);
    CHECK(common_spec_phase_log::open_from_env(cb, side) == nullptr, "unset must give no log");
    setenv(cb, "", 1);
    CHECK(common_spec_phase_log::open_from_env(cb, side) == nullptr, "an empty path must give no log");
    const std::string path = tmp_path("off");
    setenv(cb, path.c_str(), 1);
    setenv(side, "0", 1);
    CHECK(common_spec_phase_log::open_from_env(cb, side) == nullptr, "sidecars 0 must give no log");
    setenv(side, "1", 1);
    auto on = common_spec_phase_log::open_from_env(cb, side);
    CHECK(on != nullptr && on->path() == path + ".phases", "a path must give <path>.phases");
    CHECK(on && on->n_chunks() == 0 && on->n_records() == 0, "a new log holds no buffer");
    CHECK(!exists(path + ".phases"), "nothing on disk before the write");
    on.reset(); // written on destruction: the anchor alone
    CHECK(read_lines(path + ".phases").size() == 1, "an empty log writes its anchor only");
    std::remove((path + ".phases").c_str());

    // the process's own log, off: hooks do nothing, and exit writes nothing
    const std::string p2 = tmp_path("off-process");
    const int rc = in_child([&] {
        setenv("GGML_METAL_CBLOG", p2.c_str(), 1);
        setenv("GGML_METAL_CBLOG_SIDECARS", "0", 1);
        if (common_spec_phase_log::get() != nullptr) {
            exit(3);
        }
        common_spec_phase_round_begin();
        common_spec_phase_scope s(COMMON_SPEC_PHASE_CHECK, 2);
        if (s.log != nullptr || s.t0_ns != 0 || common_spec_phase_round() != -1) {
            exit(4); // a hook that is off takes no clock and keeps no round
        }
    });
    CHECK(rc == 0, "the off process log: child exit %d", rc);
    CHECK(!exists(p2 + ".phases"), "an off log must leave no file");
}

// N records -> the anchor and exactly N lines, in order, from the process log's exit write; more than one block
static void test_exit_write() {
    const std::string path = tmp_path("exit");
    const int N = (int) common_spec_phase_log::CHUNK + 1234;
    const int rc = in_child([&] {
        setenv("GGML_METAL_CBLOG", path.c_str(), 1);
        unsetenv("GGML_METAL_CBLOG_SIDECARS");
        common_spec_phase_log * log = common_spec_phase_log::get();
        if (log == nullptr || log != common_spec_phase_log::get()) {
            exit(3);
        }
        log->put(COMMON_SPEC_PHASE_OPEN_TGT, 1000000000LL, 1500000000LL, 0);
        log->round_begin();
        for (int i = 1; i < N; i++) {
            log->put((common_spec_phase) (i % COMMON_SPEC_PHASE_COUNT), 2000000000LL + i * 1000LL,
                    2000000000LL + i * 1000LL + 999, i);
        }
    });
    CHECK(rc == 0, "child exit %d", rc);
    const auto l = read_lines(path + ".phases");
    CHECK((int) l.size() == N + 1, "anchor + %d records, got %zu lines", N, l.size());
    if ((int) l.size() == N + 1) {
        double up = 0, unix_s = 0;
        CHECK(sscanf(l[0].c_str(), "A %lf %lf", &up, &unix_s) == 2 && up > 0 && unix_s > 1.7e9, "anchor: %s", l[0].c_str());
        CHECK(l[1] == "P -1 open_tgt 1.000000000 1.500000000 0", "an opening is round -1: %s", l[1].c_str());
        CHECK(l[2] == "P 0 open_dft 2.000001000 2.000001999 1", "record 1: %s", l[2].c_str());
        CHECK(l[7] == "P 0 check 2.000006000 2.000006999 6", "record 6: %s", l[7].c_str());
        char want[160];
        const int i = N - 1;
        snprintf(want, sizeof(want), "P 0 %s %.9f %.9f %d", common_spec_phase_name((common_spec_phase) (i % COMMON_SPEC_PHASE_COUNT)),
                (2000000000LL + i * 1000LL) * 1e-9, (2000000000LL + i * 1000LL + 999) * 1e-9, i);
        CHECK(l[N] == want, "the last record, past the first block: %s (want %s)", l[N].c_str(), want);
    }
    std::remove((path + ".phases").c_str());
}

// rounds rise one at a time; round-only phases are dropped between rounds; a scope's end comes after its start
static void test_rounds_and_scopes() {
    const std::string path = tmp_path("rounds");
    const int rc = in_child([&] {
        setenv("GGML_METAL_CBLOG", path.c_str(), 1);
        unsetenv("GGML_METAL_CBLOG_SIDECARS");
        if (common_spec_phase_round() != -1) {
            exit(3);
        }
        { common_spec_phase_scope s(COMMON_SPEC_PHASE_OPEN_TGT); }
        { common_spec_phase_scope s(COMMON_SPEC_PHASE_CHECK, 9, true); } // reading in: not a round's
        common_spec_phase_round_begin();
        { common_spec_phase_scope s(COMMON_SPEC_PHASE_DRAFT_STEP, 0); }
        { common_spec_phase_scope s(COMMON_SPEC_PHASE_CHECK, 2, true); }
        common_spec_phase_round_end();
        { common_spec_phase_scope s(COMMON_SPEC_PHASE_PROCESS, 2, true); } // between rounds: dropped
        common_spec_phase_round_begin();
        {
            common_spec_phase_scope s(COMMON_SPEC_PHASE_ACCEPT, 0, true);
            s.arg = 3;
            s.end();
            s.end(); // once only
        }
        if (common_spec_phase_round() != 1) {
            exit(4);
        }
    });
    CHECK(rc == 0, "child exit %d", rc);
    const auto l = read_lines(path + ".phases");
    CHECK(l.size() == 5, "anchor + 4 phases, got %zu", l.size());
    if (l.size() == 5) {
        CHECK(l[1].rfind("P -1 open_tgt ", 0) == 0, "%s", l[1].c_str());
        CHECK(l[2].rfind("P 0 draft_step ", 0) == 0 && l[2].substr(l[2].size() - 2) == " 0", "%s", l[2].c_str());
        CHECK(l[3].rfind("P 0 check ", 0) == 0 && l[3].substr(l[3].size() - 2) == " 2", "%s", l[3].c_str());
        CHECK(l[4].rfind("P 1 accept ", 0) == 0 && l[4].substr(l[4].size() - 2) == " 3", "%s", l[4].c_str());
        for (size_t i = 1; i < l.size(); i++) {
            long long r = 0, arg = 0;
            char name[32];
            double t0 = 0, t1 = 0;
            CHECK(sscanf(l[i].c_str(), "P %lld %31s %lf %lf %lld", &r, name, &t0, &t1, &arg) == 5 && t1 >= t0 && t0 > 0,
                  "start, then end: %s", l[i].c_str());
        }
    }
    std::remove((path + ".phases").c_str());
}

int main() {
    test_clock();
    test_off();
    test_exit_write();
    test_rounds_and_scopes();
    if (g_failures) {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("OK\n");
    return 0;
}

// The book manager's floor-stop log (src/llama-moe-stream-stops.h): its clock is the Metal timeline's
// (mach_absolute_time) to within a microsecond; a stop's lookup is what is left of the op after send and wait;
// N stops come out as exactly N lines after the anchor, written at exit; off (GGML_METAL_CBLOG unset, or
// GGML_METAL_CBLOG_SIDECARS=0) means no log, no file and no buffer. The `.stops` format is restated here by hand
// (the reader is the study's runner/c1_attrib.py). Which graph a stop names is the server test's
// (tests/test-c1-phase-server.py, on the MTP fixture), since only a Metal graph sets it.

#include "../src/llama-moe-stream-stops.h"

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

using stops_t = llama_moe_stream_stops;

static std::string tmp_path(const char * tag) {
    const char * dir = std::getenv("TMPDIR");
    std::string d = dir && *dir ? dir : "/tmp";
    if (d.back() != '/') {
        d += '/';
    }
    return d + "test-moe-stream-stops-" + tag + "-" + std::to_string((long) getpid()) + ".txt";
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

static void test_clock() {
#if defined(__APPLE__)
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double best = 1e18;
    for (int i = 0; i < 64; i++) {
        const int64_t a = stops_t::now_ns();
        const int64_t m = (int64_t) ((__uint128_t) mach_absolute_time() * tb.numer / tb.denom);
        const int64_t b = stops_t::now_ns();
        best = std::fmin(best, std::fabs((double) m - 0.5 * (double) (a + b)) + 0.5 * (double) (b - a));
    }
    CHECK(best <= 1000.0, "the stop clock is %.0f ns from mach_absolute_time (at most 1000)", best);
#endif
}

static void test_format_and_split() {
    const stops_t::rec r = { nullptr, 0, 1234567890123LL, 1500, 2000, 3000, 7, 10, 2 };
    CHECK(stops_t::format(r) == "F 0x0 0 7 1234.567890123 10 2 1.500 2.000 3.000\n", "format: %s", stops_t::format(r).c_str());

    const std::string path = tmp_path("split");
    {
        stops_t s(path + ".stops");
        CHECK(s.n_chunks() == 0, "a new log holds no buffer");
        // a 100 us stop with 30 us sending and 50 us waiting: 20 us of lookup
        s.put(3, 5000000000LL, 5000100000LL, 10, 4, 30000, 50000);
        CHECK(s.n_records() == 1 && s.n_chunks() == 1, "one record, one block");
    }
    const auto l = read_lines(path + ".stops");
    CHECK(l.size() == 2, "anchor + 1 stop, got %zu", l.size());
    if (l.size() == 2) {
        // no Metal graph ran on this thread: the stop names none
        CHECK(l[1] == "F 0x0 0 3 5.000000000 10 4 20.000 30.000 50.000", "the stop: %s", l[1].c_str());
    }
    std::remove((path + ".stops").c_str());
}

static void test_off() {
    const char * cb = "TEST_MOE_STOPS_CBLOG";
    const char * side = "TEST_MOE_STOPS_SIDECARS";
    unsetenv(cb);
    unsetenv(side);
    CHECK(stops_t::open_from_env(cb, side) == nullptr, "unset must give no log");
    setenv(cb, "", 1);
    CHECK(stops_t::open_from_env(cb, side) == nullptr, "an empty path must give no log");
    setenv(cb, "/x/cblog.txt", 1);
    setenv(side, "0", 1);
    CHECK(stops_t::open_from_env(cb, side) == nullptr, "sidecars 0 must give no log");

    const std::string p2 = tmp_path("off-process");
    const int rc = in_child([&] {
        setenv("GGML_METAL_CBLOG", p2.c_str(), 1);
        setenv("GGML_METAL_CBLOG_SIDECARS", "0", 1);
        exit(stops_t::get() == nullptr ? 0 : 3);
    });
    CHECK(rc == 0, "the off process log: child exit %d", rc);
    CHECK(!exists(p2 + ".stops"), "an off log must leave no file");
}

static void test_exit_write() {
    const std::string path = tmp_path("exit");
    const int N = (int) stops_t::CHUNK * 2 + 17;
    const int rc = in_child([&] {
        setenv("GGML_METAL_CBLOG", path.c_str(), 1);
        unsetenv("GGML_METAL_CBLOG_SIDECARS");
        stops_t * s = stops_t::get();
        if (s == nullptr || s != stops_t::get()) {
            exit(3);
        }
        for (int i = 0; i < N; i++) {
            s->put(i % 48, 1000000000LL + i * 100000LL, 1000000000LL + i * 100000LL + 9000, 10, i % 3, 1000, 2000);
        }
    });
    CHECK(rc == 0, "child exit %d", rc);
    const auto l = read_lines(path + ".stops");
    CHECK((int) l.size() == N + 1, "anchor + %d stops, got %zu lines", N, l.size());
    if ((int) l.size() == N + 1) {
        double up = 0, unix_s = 0;
        CHECK(sscanf(l[0].c_str(), "A %lf %lf", &up, &unix_s) == 2 && up > 0 && unix_s > 1.7e9, "anchor: %s", l[0].c_str());
        CHECK(l[1] == "F 0x0 0 0 1.000000000 10 0 6.000 1.000 2.000", "first: %s", l[1].c_str());
        char want[160];
        const int i = N - 1;
        snprintf(want, sizeof(want), "F 0x0 0 %d %.9f 10 %d 6.000 1.000 2.000", i % 48, (1000000000LL + i * 100000LL) * 1e-9, i % 3);
        CHECK(l[N] == want, "last, past two blocks: %s (want %s)", l[N].c_str(), want);
    }
    std::remove((path + ".stops").c_str());
}

int main() {
    test_clock();
    // the forked tests first: the format test's log asks the backend registry for Metal in this process
    test_off();
    test_exit_write();
    test_format_and_split();
    if (g_failures) {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("OK\n");
    return 0;
}

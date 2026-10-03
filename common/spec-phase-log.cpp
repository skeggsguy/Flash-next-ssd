#include "spec-phase-log.h"

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static const char * const PHASE_NAMES[COMMON_SPEC_PHASE_COUNT] = {
    "open_tgt", "open_dft", "save_dft", "draft_step", "load_dft",
    "save_tgt", "check", "accept", "process", "restore_tgt",
};

const char * common_spec_phase_name(common_spec_phase ph) {
    return ph >= 0 && ph < COMMON_SPEC_PHASE_COUNT ? PHASE_NAMES[ph] : "?";
}

int64_t common_spec_phase_now_ns() {
#if defined(__APPLE__)
    // = mach_absolute_time in nanoseconds: the clock of the Metal timeline. CLOCK_MONOTONIC(_RAW) count sleep here
    return (int64_t) clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
}

int64_t common_spec_phase_unix_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

std::unique_ptr<common_spec_phase_log> common_spec_phase_log::open_from_env(
        const char * env_cblog, const char * env_sidecars) {
    const char * cblog = std::getenv(env_cblog);
    if (cblog == nullptr || *cblog == '\0') {
        return nullptr;
    }
    const char * side = std::getenv(env_sidecars);
    if (side != nullptr && std::strcmp(side, "0") == 0) {
        return nullptr;
    }
    return std::make_unique<common_spec_phase_log>(std::string(cblog) + ".phases");
}

static common_spec_phase_log * g_phase_log = nullptr;

static void common_spec_phase_log_at_exit() {
    if (g_phase_log != nullptr) {
        g_phase_log->write();
    }
}

common_spec_phase_log * common_spec_phase_log::get() {
    // leaked on purpose: written by the atexit handler, so it outlives every static destructor that might log
    static common_spec_phase_log * log = [] {
        g_phase_log = open_from_env("GGML_METAL_CBLOG", "GGML_METAL_CBLOG_SIDECARS").release();
        if (g_phase_log != nullptr) {
            fprintf(stderr, "spec phase log: the apprentice's round, phase by phase, to %s at exit\n",
                    g_phase_log->path().c_str());
            std::atexit(common_spec_phase_log_at_exit);
        }
        return g_phase_log;
    }();
    return log;
}

common_spec_phase_log::common_spec_phase_log(std::string path)
    : out_path(std::move(path)), anchor_ns(common_spec_phase_now_ns()), anchor_unix_us(common_spec_phase_unix_us()) {
}

common_spec_phase_log::~common_spec_phase_log() {
    write();
}

void common_spec_phase_log::put(common_spec_phase ph, int64_t t0_ns, int64_t t1_ns, int64_t arg) {
    std::lock_guard<std::mutex> lk(mtx);
    if (written) {
        return; // after the exit write: a late record has nowhere to go
    }
    if (n == chunks.size() * CHUNK) {
        chunks.emplace_back(new rec[CHUNK]);
    }
    chunks[n / CHUNK][n % CHUNK] = { cur_round, t0_ns, t1_ns, arg, (int32_t) ph };
    n++;
}

void common_spec_phase_log::round_begin() {
    std::lock_guard<std::mutex> lk(mtx);
    cur_round++;
    open_round = true;
}

void common_spec_phase_log::round_end() {
    std::lock_guard<std::mutex> lk(mtx);
    open_round = false;
}

bool common_spec_phase_log::in_round() const {
    std::lock_guard<std::mutex> lk(mtx);
    return open_round;
}

int64_t common_spec_phase_log::round() const {
    std::lock_guard<std::mutex> lk(mtx);
    return cur_round;
}

size_t common_spec_phase_log::n_records() const {
    std::lock_guard<std::mutex> lk(mtx);
    return n;
}

size_t common_spec_phase_log::n_chunks() const {
    std::lock_guard<std::mutex> lk(mtx);
    return chunks.size();
}

std::string common_spec_phase_log::format(const rec & r) {
    char buf[160];
    snprintf(buf, sizeof(buf), "P %" PRId64 " %s %.9f %.9f %" PRId64 "\n", r.round,
            common_spec_phase_name((common_spec_phase) r.phase), r.t0_ns * 1e-9, r.t1_ns * 1e-9, r.arg);
    return buf;
}

bool common_spec_phase_log::write() {
    std::lock_guard<std::mutex> lk(mtx);
    if (written) {
        return false;
    }
    written = true;
    FILE * f = std::fopen(out_path.c_str(), "w");
    if (f == nullptr) {
        fprintf(stderr, "spec phase log: cannot write %s (%s)\n", out_path.c_str(), std::strerror(errno));
        return false;
    }
    std::setvbuf(f, nullptr, _IOFBF, 1 << 20);
    fprintf(f, "A %.9f %.6f\n", anchor_ns * 1e-9, anchor_unix_us * 1e-6);
    for (size_t i = 0; i < n; i++) {
        const std::string line = format(chunks[i / CHUNK][i % CHUNK]);
        std::fwrite(line.data(), 1, line.size(), f);
    }
    std::fclose(f);
    return true;
}

void common_spec_phase_round_begin() {
    if (auto * log = common_spec_phase_log::get()) {
        log->round_begin();
    }
}

void common_spec_phase_round_end() {
    if (auto * log = common_spec_phase_log::get()) {
        log->round_end();
    }
}

int64_t common_spec_phase_round() {
    auto * log = common_spec_phase_log::get();
    return log != nullptr ? log->round() : -1;
}

common_spec_phase_scope::common_spec_phase_scope(common_spec_phase ph, int64_t arg, bool round_only)
    : log(common_spec_phase_log::get()), ph(ph), arg(arg) {
    if (log != nullptr && round_only && !log->in_round()) {
        log = nullptr;
    }
    if (log != nullptr) {
        t0_ns = common_spec_phase_now_ns();
    }
}

void common_spec_phase_scope::end() {
    if (log == nullptr) {
        return;
    }
    log->put(ph, t0_ns, common_spec_phase_now_ns(), arg);
    log = nullptr;
}

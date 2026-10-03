#include "llama-moe-stream-stops.h"

#include "ggml-backend.h"

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

int64_t llama_moe_stream_stops::now_ns() {
#if defined(__APPLE__)
    // = mach_absolute_time in nanoseconds: the clock of the Metal timeline. CLOCK_MONOTONIC(_RAW) count sleep here
    return (int64_t) clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
}

std::unique_ptr<llama_moe_stream_stops> llama_moe_stream_stops::open_from_env(
        const char * env_cblog, const char * env_sidecars) {
    const char * cblog = std::getenv(env_cblog);
    if (cblog == nullptr || *cblog == '\0') {
        return nullptr;
    }
    const char * side = std::getenv(env_sidecars);
    if (side != nullptr && std::strcmp(side, "0") == 0) {
        return nullptr;
    }
    return std::make_unique<llama_moe_stream_stops>(std::string(cblog) + ".stops");
}

static llama_moe_stream_stops * g_stops = nullptr;

static void llama_moe_stream_stops_at_exit() {
    if (g_stops != nullptr) {
        g_stops->write();
    }
}

llama_moe_stream_stops * llama_moe_stream_stops::get() {
    // leaked on purpose: written by the atexit handler, after the last stop
    static llama_moe_stream_stops * stops = [] {
        g_stops = open_from_env("GGML_METAL_CBLOG", "GGML_METAL_CBLOG_SIDECARS").release();
        if (g_stops != nullptr) {
            fprintf(stderr, "moe stream: every floor stop to %s at exit\n", g_stops->path().c_str());
            std::atexit(llama_moe_stream_stops_at_exit);
        }
        return g_stops;
    }();
    return stops;
}

llama_moe_stream_stops::llama_moe_stream_stops(std::string path)
    : out_path(std::move(path)), anchor_ns(now_ns()),
      anchor_unix_us(std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()) {
    // the Metal timeline's last graph on this thread (ggml-metal-context.m); absent without Metal
    if (ggml_backend_reg_t reg = ggml_backend_reg_by_name("MTL")) {
        last_graph = (last_graph_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_cblog_last");
    }
}

llama_moe_stream_stops::~llama_moe_stream_stops() {
    write();
}

void llama_moe_stream_stops::put(int32_t il, int64_t t0_ns, int64_t t1_ns, uint32_t n_asked, uint32_t n_trips,
        int64_t send_ns, int64_t wait_ns) {
    void *   ctx = nullptr;
    uint64_t seq = 0;
    if (last_graph != nullptr) {
        last_graph(&ctx, &seq);
    }
    const int64_t lookup_ns = (t1_ns - t0_ns) - send_ns - wait_ns;
    put_rec({ ctx, seq, t0_ns, lookup_ns, send_ns, wait_ns, il, n_asked, n_trips });
}

void llama_moe_stream_stops::put_rec(const rec & r) {
    std::lock_guard<std::mutex> lk(mtx);
    if (written) {
        return;
    }
    if (n == chunks.size() * CHUNK) {
        chunks.emplace_back(new rec[CHUNK]);
    }
    chunks[n / CHUNK][n % CHUNK] = r;
    n++;
}

size_t llama_moe_stream_stops::n_records() const {
    std::lock_guard<std::mutex> lk(mtx);
    return n;
}

size_t llama_moe_stream_stops::n_chunks() const {
    std::lock_guard<std::mutex> lk(mtx);
    return chunks.size();
}

std::string llama_moe_stream_stops::format(const rec & r) {
    char buf[200];
    // %p as the Metal timeline prints its ctx, so the two files name a graph with the same string
    snprintf(buf, sizeof(buf), "F %p %" PRIu64 " %d %.9f %u %u %.3f %.3f %.3f\n", r.ctx, r.seq, r.il,
            r.t0_ns * 1e-9, r.n_asked, r.n_trips, r.lookup_ns * 1e-3, r.send_ns * 1e-3, r.wait_ns * 1e-3);
    return buf;
}

bool llama_moe_stream_stops::write() {
    std::lock_guard<std::mutex> lk(mtx);
    if (written) {
        return false;
    }
    written = true;
    FILE * f = std::fopen(out_path.c_str(), "w");
    if (f == nullptr) {
        fprintf(stderr, "moe stream: cannot write %s (%s)\n", out_path.c_str(), std::strerror(errno));
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

#pragma once

// The book manager's floor stops (the study's C1-PROFILE-PLAN.md): `<GGML_METAL_CBLOG>.stops`, one line per call
// of the remap op (llama-moe-stream-remap.cpp), the CPU's turn between two Metal graphs while writing.
//
// Off unless GGML_METAL_CBLOG=<path> is set and GGML_METAL_CBLOG_SIDECARS is not "0"; when off the remap pays one
// pointer check: no file, no buffer, no clock call.
//
//   A <uptime_raw_s> <unix_s>                                       clock anchor, once (the first line)
//   F <metal_ctx> <seq> <il> <t_start> <n_asked> <n_trips> <lookup_us> <send_us> <wait_us>
//
// metal_ctx and seq name the last graph this thread handed to Metal before the stop (the Metal timeline's `G`
// line: same pointer, same number), so the stop sits in the gap after that graph; "0x0 0" when the timeline is
// off or there is no Metal backend. t_start is the op's entry in seconds of CLOCK_UPTIME_RAW (the timeline's
// clock). n_asked: distinct books the floor's slip named; n_trips: books not on the desk, sent for now.
// wait: the CPU blocked until this floor's books were on the desk (a book still in flight from the lookahead
// counts; the runners' drive time ahead of the stop does not). send: reserving slots and handing the trips to the
// runners (and the lent belt's copies). lookup: the rest of the op (the lock, the slip, the desk lookups, the
// relabel).

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct llama_moe_stream_stops {
    struct rec {
        const void * ctx;
        uint64_t     seq;
        int64_t      t0_ns;
        int64_t      lookup_ns;
        int64_t      send_ns;
        int64_t      wait_ns;
        int32_t      il;
        uint32_t     n_asked;
        uint32_t     n_trips;
    };

    static constexpr size_t CHUNK = 1 << 14;

    // the process's log: nullptr when off; the first call opens it and arranges the write at exit
    static llama_moe_stream_stops * get();
    static std::unique_ptr<llama_moe_stream_stops> open_from_env(const char * env_cblog, const char * env_sidecars);

    static int64_t now_ns(); // CLOCK_UPTIME_RAW on Apple

    explicit llama_moe_stream_stops(std::string path);
    ~llama_moe_stream_stops(); // writes, if not written yet

    llama_moe_stream_stops(const llama_moe_stream_stops &) = delete;
    llama_moe_stream_stops & operator=(const llama_moe_stream_stops &) = delete;

    // one floor stop; the graph before it is asked of the Metal backend here
    void put(int32_t il, int64_t t0_ns, int64_t t1_ns, uint32_t n_asked, uint32_t n_trips, int64_t send_ns,
             int64_t wait_ns);
    void put_rec(const rec & r);

    size_t n_records() const;
    size_t n_chunks()  const;
    const std::string & path() const { return out_path; }

    bool write();

    static std::string format(const rec & r);

private:
    using last_graph_fn = void (*)(void ** ctx, uint64_t * seq);

    mutable std::mutex mtx;
    std::string        out_path;
    int64_t            anchor_ns;
    int64_t            anchor_unix_us;
    last_graph_fn      last_graph = nullptr;
    std::vector<std::unique_ptr<rec[]>> chunks;
    size_t             n = 0;
    bool               written = false;
};

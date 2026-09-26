#include "llama-moe-stream.h"
#include "llama-moe-room.h"

#include "llama-impl.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>       // fopen/fprintf/rename, for the route-hotness json dump
#include <functional>   // std::greater, for the route-hotness sort
#include <numeric>      // std::accumulate, for the route-hotness totals
#include <string>       // std::string, for the json path
#include <vector>

// dump counter deltas since the last dump, if the interval has elapsed. caller holds mtx.
//
// The number this exists to produce is `stall`: wave-staging wait time as a percentage of wall
// time in the window. Prefill has been measured as neither FLOP-bound (cutting 40% of expert-GEMM
// work made it slower) nor bandwidth-bound (halving per-token expert bytes via -ub changed nothing)
// nor queue-depth-bound (8/12/16 I/O threads are flat), so the open question is whether the time is
// going into waiting on expert loads at all. A high stall% says yes and points at the load path;
// a low one says the cost is elsewhere - the CPU-side staging, the custom op, or the GEMM itself.
void llama_moe_stream::maybe_dump_stats_locked() {
    if (stats_dump_us <= 0) {
        return;
    }

    const int64_t now = ggml_time_us();
    if (stats_t_last_us == 0) {
        stats_t_last_us = now;
        stats_prev      = stats;
        return;
    }
    const int64_t dt = now - stats_t_last_us;
    if (dt < stats_dump_us) {
        return;
    }
    busy_flush_locked(now);

    const int64_t d_calls   = stats.n_calls          - stats_prev.n_calls;
    const int64_t d_hit     = stats.n_hit            - stats_prev.n_hit;
    const int64_t d_miss    = stats.n_miss           - stats_prev.n_miss;
    const int64_t d_waves   = stats.n_waves_run      - stats_prev.n_waves_run;
    const int64_t d_pre_i   = stats.n_preload_issued - stats_prev.n_preload_issued;
    const int64_t d_pre_r   = stats.n_preload_ready  - stats_prev.n_preload_ready;
    const int64_t d_hit_l   = stats.n_hit_loading    - stats_prev.n_hit_loading;
    const int64_t d_hit_r   = stats.n_hit_ready      - stats_prev.n_hit_ready;
    const int64_t d_stall   = (stats.t_stall_us      - stats_prev.t_stall_us) +
                              (stats.t_stall_wave_us - stats_prev.t_stall_wave_us);
    const int64_t d_op      = (stats.t_wave_op_us    - stats_prev.t_wave_op_us) +
                              (stats.t_remap_op_us   - stats_prev.t_remap_op_us);
    const int64_t d_cpu     = d_op > d_stall ? d_op - d_stall : 0; // staging work, excluding the wait
    const int64_t d_touched = d_hit + d_miss;

    // only report windows that did work, so idle time between requests does not emit noise
    if (d_calls > 0 || d_waves > 0) {
        LLAMA_LOG_WARN("%s: moe stream: %6.2f s | remaps %5" PRId64 " waves %4" PRId64
                       " | miss %5" PRId64 "/%-6" PRId64 " (%5.1f%%) | hit rdy/late %5" PRId64 "/%-5" PRId64
                       " | preload %4" PRId64 "->%-4" PRId64
                       " | stall %5.1f%%  cpu-op %5.1f%%  rest(gpu) %5.1f%%\n",
                __func__, dt/1e6, d_calls, d_waves, d_miss, d_touched,
                d_touched > 0 ? 100.0*d_miss/d_touched : 0.0,
                d_hit_r, d_hit_l,
                d_pre_i, d_pre_r,
                dt > 0 ? 100.0*d_stall/dt      : 0.0,
                dt > 0 ? 100.0*d_cpu/dt        : 0.0,
                dt > 0 ? 100.0*(dt - d_op)/dt  : 0.0);

        // per-miss anatomy: is a miss read-bound (parallelise the slab reads) or upload-bound
        // (pipeline read against upload)?
        const int64_t d_rd = stats.t_io_read_us   - stats_prev.t_io_read_us;
        const int64_t d_up = stats.t_io_upload_us - stats_prev.t_io_upload_us;
        const int64_t d_sl = stats.n_slabs_read   - stats_prev.n_slabs_read;
        if (d_sl > 0) {
            LLAMA_LOG_WARN("%s: moe stream: slabs %5" PRId64 " | read %7.2f ms (%5.3f ms/slab) | "
                           "upload %7.2f ms (%5.3f ms/slab) | read %4.1f%% of miss\n",
                    __func__, d_sl, d_rd/1000.0, d_rd/1000.0/d_sl, d_up/1000.0, d_up/1000.0/d_sl,
                    (d_rd + d_up) > 0 ? 100.0*d_rd/(d_rd + d_up) : 0.0);

            // reads faster than the first bound cannot have come from the drive, so they measure
            // the page cache acting as an L2 behind the slot cache
            char hist[256];
            int  off = 0;
            int64_t fast = 0;
            for (int b = 0; b < MOE_STREAM_READ_BUCKETS; b++) {
                const int64_t d = stats.n_read_bucket[b] - stats_prev.n_read_bucket[b];
                if (b == 0) {
                    fast = d;
                }
                off += snprintf(hist + off, sizeof(hist) - off, "%s%" PRId64,
                        b ? "/" : "", d);
            }
            LLAMA_LOG_WARN("%s: moe stream: read us <100/<250/<500/<1k/<2k/<4k/<8k/more = %s | "
                           "page-cache L2 %4.1f%%\n",
                    __func__, hist, d_sl > 0 ? 100.0*fast/d_sl : 0.0);

            // which runner served the window's bytes. A 53/47 split that reads 100/0 is a
            // misconfiguration, and no throughput number would name it.
            const int64_t d_bf = stats.n_bytes_file - stats_prev.n_bytes_file;
            const int64_t d_ba = stats.n_bytes_alt  - stats_prev.n_bytes_alt;
            const int64_t d_bt = d_bf + d_ba;
            LLAMA_LOG_WARN("%s: moe stream: expert bytes file/alt = %7.2f/%7.2f MiB (%4.1f%%/%4.1f%%)\n",
                    __func__, d_bf/1048576.0, d_ba/1048576.0,
                    d_bt > 0 ? 100.0*d_bf/d_bt : 0.0,
                    d_bt > 0 ? 100.0*d_ba/d_bt : 0.0);

            // how busy each runner was, and how fast it read while busy
            const int64_t d_tf = stats.t_busy_file_us - stats_prev.t_busy_file_us;
            const int64_t d_ta = stats.t_busy_alt_us  - stats_prev.t_busy_alt_us;
            LLAMA_LOG_WARN("%s: moe stream: drives busy file/alt = %5.1f%%/%5.1f%% | %5.2f/%5.2f GB/s\n",
                    __func__, dt > 0 ? 100.0*d_tf/dt : 0.0, dt > 0 ? 100.0*d_ta/dt : 0.0,
                    d_tf > 0 ? d_bf/1e3/d_tf : 0.0, d_ta > 0 ? d_ba/1e3/d_ta : 0.0);
            const int64_t d_la2_i = stats.n_la2_issued - stats_prev.n_la2_issued;
            const int64_t d_la2_u = stats.n_la2_used   - stats_prev.n_la2_used;
            if (d_la2_i > 0 || d_la2_u > 0) {
                LLAMA_LOG_WARN("%s: moe stream: lookahead 2 floors: fetched %lld, used %lld (%.1f%%)\n", __func__,
                        (long long) d_la2_i, (long long) d_la2_u, d_la2_i > 0 ? 100.0*d_la2_u/d_la2_i : 0.0);
            }
        }

        if (room) {
            room->dump_stats_locked(dt);
        }

        if (n_slot_chk > 0) {
            LLAMA_LOG_WARN("%s: moe stream: gpu slot resolve verified %" PRId64 " calls, %" PRId64 " mismatches\n",
                    __func__, n_slot_chk, n_slot_bad);
        }

        if (stats.chunk_util_max > 0) {
            LLAMA_LOG_WARN("%s: moe stream: chunk utilisation worst = %" PRId64 "%% (100%% = abort)\n",
                    __func__, stats.chunk_util_max);
        }

        if (stats.pair_over_max > 0) {
            // running maximum, not a delta: it sizes the graph's chunk slack, so what matters is the
            // worst case the run has produced so far, not the worst in this particular window
            LLAMA_LOG_WARN("%s: moe stream: pair imbalance worst = +%" PRId64 "%% over mean\n",
                    __func__, stats.pair_over_max);
        }
    }

    if (dump_hotness) {
        dump_route_hotness_locked();
    }

    stats_t_last_us = now;
    stats_prev      = stats;
}

// How concentrated is expert routing? For each streamed layer the selection counts are sorted and
// the cumulative share held by the hottest 12.5%/25%/50% of experts is computed, then averaged over
// layers. Flat traffic (12.5% of experts taking ~12.5% of selections) says a per-expert precision
// scheme has nothing to exploit; a steep curve says it might. Caller holds mtx.
void llama_moe_stream::dump_route_hotness_locked() const {
    const size_t frac_n = 3;
    const double fracs[frac_n] = { 0.125, 0.25, 0.5 };
    double   share_sum[frac_n] = { 0.0, 0.0, 0.0 };
    int64_t  n_layers_seen = 0;
    uint64_t total_all     = 0;

    for (const auto & sl : layers) {
        if (sl == nullptr || sl->route_hotness.empty()) {
            continue;
        }
        std::vector<uint32_t> h = sl->route_hotness;
        const uint64_t total = std::accumulate(h.begin(), h.end(), (uint64_t) 0);
        if (total == 0) {
            continue;
        }
        total_all += total;
        std::sort(h.begin(), h.end(), std::greater<uint32_t>());
        for (size_t f = 0; f < frac_n; f++) {
            const size_t k   = std::max<size_t>(1, (size_t)(h.size()*fracs[f]));
            uint64_t     cum = 0;
            for (size_t i = 0; i < k && i < h.size(); i++) {
                cum += h[i];
            }
            share_sum[f] += 100.0*cum/total;
        }
        n_layers_seen++;
    }

    if (n_layers_seen == 0) {
        LLAMA_LOG_WARN("%s: moe stream: route hotness: no selections recorded yet\n", __func__);
        return;
    }

    LLAMA_LOG_WARN("%s: moe stream: route hotness over %" PRId64 " layers (%" PRIu64 " selections, %s): "
                   "top 12.5%% of experts = %4.1f%% of traffic | top 25%% = %4.1f%% | top 50%% = %4.1f%% "
                   "(flat would be 12.5/25/50)\n",
            __func__, n_layers_seen, total_all,
            hot_decay_interval > 0 ? "DECAYED, recent-weighted - set LLAMA_MOE_STREAM_HOT_DECAY=0 for cumulative"
                                   : "cumulative",
            share_sum[0]/n_layers_seen, share_sum[1]/n_layers_seen, share_sum[2]/n_layers_seen);

    // one concrete layer, so the shape is visible and not just summarised
    for (const auto & sl : layers) {
        if (sl == nullptr || sl->route_hotness.empty()) {
            continue;
        }
        std::vector<std::pair<uint32_t, int32_t>> v;
        v.reserve(sl->route_hotness.size());
        for (size_t e = 0; e < sl->route_hotness.size(); e++) {
            v.emplace_back(sl->route_hotness[e], (int32_t) e);
        }
        std::sort(v.begin(), v.end(), std::greater<std::pair<uint32_t, int32_t>>());
        char buf[256];
        int  off = 0;
        for (size_t i = 0; i < 8 && i < v.size() && off < (int) sizeof(buf); i++) {
            const int n = snprintf(buf + off, sizeof(buf) - off, "%s#%d:%u",
                    i ? " " : "", v[i].second, v[i].first);
            // snprintf returns the length it WANTED, so off must not absorb it blindly:
            // past the end, sizeof(buf) - off underflows and the next write leaves buf.
            if (n < 0 || n >= (int) (sizeof(buf) - off)) {
                off = (int) sizeof(buf) - 1;
                break;
            }
            off += n;
        }
        LLAMA_LOG_WARN("%s: moe stream: layer %d hottest experts: %s | coldest count = %u\n",
                __func__, sl->il, buf, v.back().first);
        break;
    }

    // Full per-layer counts for offline use (building a keep-manifest). Rewritten each dump and
    // written via a temp file + rename, so a reader never sees a half-written file if the server
    // is killed mid-write.
    if (!hotness_json.empty()) {
        // A model and its MTP draft each own a llama_moe_stream, and both would write this path -
        // the draft (1 layer) clobbering the model (48). Give each instance its own file by
        // suffixing the streamed-layer count: counts.json -> counts.48L.json / counts.1L.json.
        std::string path = hotness_json;
        const size_t dot = path.find_last_of('.');
        const std::string suffix = "." + std::to_string(n_layers_seen) + "L";
        path = (dot == std::string::npos || path.find('/', dot) != std::string::npos)
             ? path + suffix
             : path.substr(0, dot) + suffix + path.substr(dot);

        const std::string tmp = path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "w");
        if (f == nullptr) {
            LLAMA_LOG_WARN("%s: moe stream: cannot write %s\n", __func__, tmp.c_str());
            return;
        }
        fprintf(f, "{\n  \"decayed\": %s,\n  \"selections_total\": %" PRIu64 ",\n  \"layers\": {",
                hot_decay_interval > 0 ? "true" : "false", total_all);
        bool first_layer = true;
        for (const auto & sl : layers) {
            if (sl == nullptr || sl->route_hotness.empty()) {
                continue;
            }
            fprintf(f, "%s\n    \"%d\": [", first_layer ? "" : ",", sl->il);
            first_layer = false;
            for (size_t e = 0; e < sl->route_hotness.size(); e++) {
                fprintf(f, "%s%u", e ? "," : "", sl->route_hotness[e]);
            }
            fprintf(f, "]");
        }
        fprintf(f, "\n  },\n  \"hits\": {");
        // Per-layer hit/miss/cold-miss beside the selection counts. Unlike "layers" these never
        // decay, so they are the run's whole history whatever LLAMA_MOE_STREAM_HOT_DECAY says.
        first_layer = true;
        for (const auto & sl : layers) {
            if (sl == nullptr || sl->route_hotness.empty()) {
                continue;
            }
            fprintf(f, "%s\n    \"%d\": [%" PRId64 ",%" PRId64 ",%" PRId64 "]",
                    first_layer ? "" : ",", sl->il, sl->n_hit, sl->n_miss, sl->n_miss_cold);
            first_layer = false;
        }
        fprintf(f, "\n  }\n}\n");
        fclose(f);
        if (rename(tmp.c_str(), path.c_str()) != 0) {
            LLAMA_LOG_WARN("%s: moe stream: cannot rename %s -> %s\n",
                    __func__, tmp.c_str(), path.c_str());
        }
    }
}

void llama_moe_stream::print_stats() {
    std::lock_guard<std::mutex> lock(mtx);

    trace_flush_locked();

    const int64_t n_touched = stats.n_hit + stats.n_miss;
    LLAMA_LOG_WARN("%s: moe stream: remap calls = %" PRId64 ", expert hits = %" PRId64 ", misses = %" PRId64 " (%" PRId64 " cold), hit rate = %.2f%%\n",
            __func__, stats.n_calls, stats.n_hit, stats.n_miss, stats.n_miss_cold,
            n_touched > 0 ? 100.0*stats.n_hit/n_touched : 0.0);
    LLAMA_LOG_WARN("%s: moe stream: load stall = %.2f ms total (%.3f ms per remap call)\n",
            __func__, stats.t_stall_us/1000.0, stats.n_calls > 0 ? stats.t_stall_us/1000.0/stats.n_calls : 0.0);
    if (stats.n_wave_calls > 0) {
        LLAMA_LOG_WARN("%s: moe stream: waves = %" PRId64 " (%" PRId64 " non-empty), preloads issued = %" PRId64 " (ready on arrival = %" PRId64 "), wave stall = %.2f ms\n",
                __func__, stats.n_wave_calls, stats.n_waves_run, stats.n_preload_issued, stats.n_preload_ready, stats.t_stall_wave_us/1000.0);
    }
    if (n_slot_chk > 0) {
        LLAMA_LOG_WARN("%s: moe stream: gpu slot resolve = %" PRId64 " calls verified, %" PRId64 " mismatches\n",
                __func__, n_slot_chk, n_slot_bad);
    }
    if (room) {
        room->print_stats_locked();
    }
    {
        const int64_t n_bytes = stats.n_bytes_file + stats.n_bytes_alt;
        LLAMA_LOG_WARN("%s: moe stream: expert bytes read = %.2f GiB file + %.2f GiB alt (%4.1f%%/%4.1f%%)\n",
                __func__, stats.n_bytes_file/1073741824.0, stats.n_bytes_alt/1073741824.0,
                n_bytes > 0 ? 100.0*stats.n_bytes_file/n_bytes : 0.0,
                n_bytes > 0 ? 100.0*stats.n_bytes_alt/n_bytes  : 0.0);
    }
    if (stats.n_slabs_read > 0) {
        for (int b = 0; b < MOE_STREAM_READ_BUCKETS; b++) {
            const int64_t lo = b ? MOE_STREAM_READ_BUCKET_US[b-1] : 0;
            if (b == MOE_STREAM_READ_BUCKETS - 1) {
                LLAMA_LOG_WARN("%s: moe stream: read %6" PRId64 " us +      : %8" PRId64 " slabs (%4.1f%%)\n",
                        __func__, lo, stats.n_read_bucket[b], 100.0*stats.n_read_bucket[b]/stats.n_slabs_read);
            } else {
                LLAMA_LOG_WARN("%s: moe stream: read %6" PRId64 " - %6" PRId64 " us: %8" PRId64 " slabs (%4.1f%%)\n",
                        __func__, lo, MOE_STREAM_READ_BUCKET_US[b], stats.n_read_bucket[b],
                        100.0*stats.n_read_bucket[b]/stats.n_slabs_read);
            }
        }
    }
}

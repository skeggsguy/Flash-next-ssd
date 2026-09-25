#pragma once

#include <cstdint>

// The book manager's counters: llama_moe_stream::stats, and stats_prev, the snapshot of the last dump.

// Read-latency histogram for expert slab reads: upper bounds in microseconds, plus one bucket for
// everything above the last bound.
//
// The mean read time cannot separate a page-cache hit from an SSD read. A hit is a ~3 MB memcpy
// (~0.2 ms), an SSD read is 1-3 ms, so the distribution is bimodal and the mean sits between the
// modes - 3.19 ms/slab is consistent with anything from 0% to 40% hits. This histogram is what
// sizes the page cache as an L2 tier behind the slot cache, which decides whether it is worth
// protecting from prefill (prefill streams ~94 GB of experts through a page cache of a few GB).
static const int64_t MOE_STREAM_READ_BUCKET_US[] = { 100, 250, 500, 1000, 2000, 4000, 8000 };
static const int     MOE_STREAM_READ_BUCKETS     = 8;

struct llama_moe_stream_stats {
    int64_t n_calls     = 0; // remap invocations
    int64_t n_hit       = 0; // touched experts already resident or loading
    int64_t n_miss      = 0; // demand loads issued
    int64_t n_miss_cold = 0; // first-ever touch of an expert
    int64_t t_stall_us  = 0; // wait time in miss handling

    int64_t n_wave_calls     = 0; // wave-ids invocations (>= n_calls under multi-pass prefill)
    int64_t n_waves_run      = 0; // non-empty waves
    int64_t n_preload_issued = 0; // next-wave loads started during a wave's compute
    int64_t n_preload_ready  = 0; // wave experts already resident from the previous preload

    // Decode has no waves, so n_preload_ready above is unreachable there and reads as 0 - it
    // cannot say whether the lookahead prefetch is working. These two split a demand HIT by the
    // slot's state, which distinguishes the two failure modes:
    //   hit while LOADING  -> the prefetch picked the right expert but issued it too late
    //   miss (not counted here) -> the prefetch picked the wrong expert, or never issued
    int64_t n_hit_loading = 0; // hit on a slot still loading (a preload that did not land in time)
    int64_t n_hit_ready   = 0; // hit on a slot already resident
    int64_t t_stall_wave_us  = 0; // wait time in wave miss handling

    // Total wall time inside the streaming custom ops, stall included. These ops run on the CPU
    // as graph dependencies, so the GPU is idle for their duration and the time is on the
    // critical path. (op - stall) is therefore the CPU-side staging cost - planning waves,
    // picking victims, and emit_wave_slots' n_tok*n_ids id rewrite - as opposed to waiting on I/O.
    // Needed to answer what the ~92% of non-stall prefill actually is: Metal GEMM, or this.
    int64_t t_wave_op_us  = 0; // in llama_moe_stream_wave_ids  (prefill, multi-wave path)
    int64_t t_remap_op_us = 0; // in llama_moe_stream_remap     (decode, single-wave path)

    // Pair partitioning only. The graph fixes the per-wave pair chunk before the router runs, so
    // the slack it carries over the mean has to cover whatever imbalance the planner is left with
    // after LPT - and that slack is pure waste, the one cost the partition path still pays. This
    // is the measurement that sizes it: the worst (max wave load / mean - 1) seen, in percent.
    int64_t pair_over_max = 0;

    // worst wave load as a % of plan_pair_chunk. The margin to 100% IS the crash margin, and
    // unlike imbalance-over-mean it accounts for the n_tokens floor on the chunk.
    int64_t chunk_util_max = 0;

    // how often a hot expert had to be split across waves to fit the static chunk
    int64_t n_pair_splits = 0;

    // per-miss anatomy: SSD read vs Metal upload, and how many slabs were moved
    int64_t t_io_read_us   = 0;
    int64_t t_io_upload_us = 0;
    int64_t n_slabs_read   = 0;

    // Expert bytes read, split by which of the two runners served them: n_bytes_file is the
    // set opened from the model's own shards, n_bytes_alt the set opened from
    // --moe-stream-alt-path. With one runner n_bytes_alt stays 0. The point of the split is
    // that it is the only in-engine evidence that striping is actually striping: a 53/47 flag
    // that reads 100/0 is a misconfiguration no throughput number would name.
    int64_t n_bytes_file = 0;
    int64_t n_bytes_alt  = 0;

    // Time each runner had at least one read in flight. Against the window it is how busy the drive
    // was; against its bytes, how fast it went while busy. The reading room is meant to keep both
    // runners busy through a read-in, and this is where that shows.
    int64_t t_busy_file_us = 0;
    int64_t t_busy_alt_us  = 0;

    // read latency distribution; see MOE_STREAM_READ_BUCKET_US
    int64_t n_read_bucket[MOE_STREAM_READ_BUCKETS] = {0};
};

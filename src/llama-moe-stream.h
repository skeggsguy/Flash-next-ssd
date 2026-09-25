#pragma once

#include "llama-mmap.h"

#include "ggml-cpp.h"
#include "llama-moe-stream-layer.h"
#include "llama-moe-stream-stats.h"

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// LLAMA_MOE_STREAM_PARTITION: give each (token, expert) pair to exactly one wave instead of running
// every wave over every pair and masking the rest to zero. ON by default; set 0 to disable.
// It only engages for multi-wave ubatches - a single wave has nothing to mask against, so decode
// keeps the plain path. Read in both llama-moe-stream.cpp and llama-graph.cpp, which must agree:
// the graph sizes the gathered tensors that the planner fills.
bool moe_stream_partition();

// SSD streaming of MoE routed expert weights
//
// Streamed layers do not materialize their ffn_*_exps tensors; instead each weight gets a
// device-side cache tensor of n_slots expert slabs, filled on demand from the GGUF file by an
// id-remapping custom op that runs on the CPU right after the router top-k. The remap only
// changes which cache slot an expert id resolves to - it never changes which experts the router
// selected, so streaming affects latency, not outputs.
//
// Missing experts are loaded by a pool of I/O threads while the remap op waits; eviction is by
// decaying route hotness with an LRU tiebreak. Reads are buffered by default, or O_DIRECT with
// LLAMA_MOE_STREAM_DIRECT=1 (bypasses the page cache; recommended when the model far exceeds RAM).
//
// note: multiple contexts decoding the same streamed model concurrently are not supported -
// one context can evict slots referenced by the other's in-flight graph.

struct llama_moe_stream {
    uint32_t n_slots      = 0; // expert cache slots per streamed layer
    int32_t  n_io_threads = 0;

    std::vector<std::unique_ptr<llama_moe_stream_layer>> layers; // [n_layer], null = not streamed

    llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct);
    ~llama_moe_stream();

    llama_moe_stream_layer * layer(int32_t il) const {
        return il >= 0 && (size_t) il < layers.size() ? layers[il].get() : nullptr;
    }

    std::vector<llama_moe_stream_hash_router> hash_routers;
    uint32_t hash_n_used = 0; // n_expert_used, the row stride of every tid2eid

    // idempotent; called at graph build, which is the first point the tensor pointer is known
    void register_hash_router(int32_t il, ggml_tensor * tid2eid, uint32_t n_expert_used);

    // registers a streamed weight of layer il and returns its cache tensor
    ggml_tensor * create_cache_tensor(
            int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
            uint16_t file_idx, size_t offs);

    // allocate the cache tensor buffers (after all create_cache_tensor calls)
    void alloc_bufs(bool no_alloc);

    // reopen the GGUF files for streaming reads
    void open_files(const std::vector<std::string> & paths);

    size_t size_bufs() const;

    void print_stats();

    // LLAMA_MOE_STREAM_NO_ZEROCOPY: stage reads through a bounce buffer even when the backend offers
    // a host pointer. Kept as a switch because reading straight into the cache measured no better.
    bool no_zerocopy = false;

    bool use_direct_io = false; // O_DIRECT streaming reads (LLAMA_MOE_STREAM_DIRECT), no page cache

    llama_files files; // privately reopened GGUF files, same indices as the loader's

    // Two runners. A second, byte-identical copy of the model lives on another drive and
    // files_alt holds its shards, opened with the same indices as `files` - the shards are copies,
    // so a weight's file_idx and offs are the same in both and nothing about the layout changes.
    // An expert is read from one set or the other by its ID: the first alt_split percent of a
    // layer's expert ids from `files`, the rest from `files_alt`. Splitting by id rather than by
    // hotness keeps the proportion stable - hot and cold experts are spread across the range, so
    // the byte share follows the id share whatever the workload routes to. The percentage is the
    // two drives' measured throughput share (53/47 on this box), not a guess.
    llama_files files_alt;
    std::string alt_path;         // the alt copy's FIRST shard; empty = one runner
    int32_t     alt_split = 100;  // percent of expert ids served from `files`

    // which of the two sets serves expert `expert` of a layer that has `n_expert` experts
    bool use_alt(int32_t expert, uint32_t n_expert) const {
        return !files_alt.empty() && n_expert > 0 &&
               ((int64_t) expert*100) / (int64_t) n_expert >= (int64_t) alt_split;
    }

    size_t  max_nb_expert      = 0;
    int64_t hot_decay_interval = 0; // remap calls between route-hotness halvings (0 = no decay)

    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context_ptr>> ctxs; // one per buft, or per chunk
    std::vector<size_t> ctx_bytes; // cache bytes placed in ctxs[i], same index
    std::vector<ggml_backend_buffer_ptr> bufs;

    // The gentle opening. Each backend buffer is pinned the moment it is created (Metal adds it to
    // a residency set and requests residency at once), so a desk allocated as one buffer asks the
    // OS for all of it in one step and the OS has to make room just as fast - by compressing other
    // processes' memory, or, if it cannot keep up, by swapping. LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB
    // caps each cache buffer at that size, and LLAMA_MOE_STREAM_ALLOC_PAUSE_MS (default 500 when
    // chunking) waits between them, so the desk is claimed in steps. 0 = one buffer per buft, as
    // before. Only where the slabs live changes, never their contents.
    size_t  alloc_chunk_bytes = 0;
    int32_t alloc_pause_ms    = 0;

    // The GPU slot tables get their own context and buffer, so that turning the feature on leaves
    // the expert-cache buffer byte-for-byte as it is with the feature off. Sharing the cache
    // context would interleave a small I32 tensor between a layer's expert slabs.
    ggml_backend_buffer_type_t buft_state = nullptr;
    ggml_context_ptr         ctx_state;
    ggml_backend_buffer_ptr  buf_state;

    // load pool (queue and all layer residency state guarded by mtx)
    mutable std::mutex      mtx;
    std::condition_variable cv_work; // queued work or shutdown
    std::condition_variable cv_done; // a load committed or failed

    std::deque<llama_moe_stream_work> q_demand;

    // Speculative loads (lookahead and hash prefetch) live in their OWN queue, drained only when
    // nothing is being waited on. With a single FIFO a wide prefetch delays the demand read a layer
    // is blocked on: measured at lookahead K=16, misses/remap fell 0.251 -> 0.172 but read latency
    // rose 1.69 -> 2.17 ms/slab and decode fell 8.45 -> 5.73 t/s. Separating them lets prefetch
    // width cost bandwidth (of which the drive has ~6x spare) instead of demand latency.
    std::deque<llama_moe_stream_work> q_spec;

    // cap the speculative backlog: a prefetch that is still queued when its layer arrives has done
    // nothing but reserve a slot and burn bandwidth
    size_t q_spec_max = 0;

    std::vector<std::thread> workers;
    bool workers_started = false;
    bool shutting_down   = false;
    bool load_failed     = false;

    bool debug = false;

    // GPU slot resolution (LLAMA_MOE_STREAM_GPU_SLOT): 0 off, 1 verify (the CPU remap still decides
    // and its answer is what reaches the GEMM, the kernel only checks itself against it), 2 publish
    // the table but insert no graph node - isolates the cost of the state buffer from the kernel.
    // LLAMA_MOE_STREAM_LRU: drop route hotness and evict purely by last use. Prerequisite for
    // moving residency to the GPU - hotness has no cheap kernel form, an argmin over 113 slots
    // does. Simulated at +11% misses.
    bool pure_lru     = false;

    int  gpu_slot     = 0;
    int64_t n_slot_chk = 0; // remap calls whose table the kernel checked
    int64_t n_slot_bad = 0; // of those, disagreements with the CPU - must stay 0
    int     n_slot_warn = 0; // warnings printed, capped

    llama_moe_stream_stats stats;

    // Periodic delta dump (LLAMA_MOE_STREAM_STATS_MS=<ms>, unset = off).
    //
    // print_stats() alone cannot answer "how much of prefill is spent waiting for experts": it is
    // cumulative over the run, so prefill and decode are summed together, it is LLAMA_LOG_INFO which
    // llama-server filters out, and it only runs on clean shutdown - a pkill loses it entirely.
    // This dumps the deltas since the last dump, at WARN, on a wall-clock interval, so each line
    // covers one phase and carries the ratio that matters: stall time over elapsed time.
    int64_t                stats_dump_us    = 0; // interval, 0 = disabled
    int64_t                stats_t_last_us  = 0;
    llama_moe_stream_stats stats_prev;
    void maybe_dump_stats_locked();

    // Per-expert selection counts, to answer "is expert usage skewed enough to be worth exploiting?"
    // (e.g. giving rarely-routed experts fewer bits). Off unless LLAMA_MOE_STREAM_HOTNESS is set;
    // rides the LLAMA_MOE_STREAM_STATS_MS dump so it needs no new trigger. NOTE: route_hotness decays
    // (halves every hot_decay_interval), so this measures RECENT usage - set
    // LLAMA_MOE_STREAM_HOT_DECAY=0 for cumulative counts.
    void dump_route_hotness_locked() const;

    bool dump_hotness = false;

    // LLAMA_MOE_STREAM_TRACE=<path>: the borrowing log proper - every slip, in routing order, as it
    // is written. The hotness counts above say how often a book was wanted over a whole run; they
    // cannot say in what ORDER, so they cannot drive a cache simulator or size a reserve shelf
    // against a replacement policy. This writes the raw selected ids instead and leaves the
    // arithmetic to sim/ (see sim/README.md for the format and sim/trace_reader.py for a reader).
    //
    // Little-endian, packed, no alignment padding. Header once at file start:
    //     char magic[4] = "MSTR", u32 version = 1, u32 n_expert_used, u32 n_layer
    // then one record per routing call:
    //     u8 il, u32 n_tokens, u8 kind (0 = a single token, 1 = a multi-token ubatch),
    //     i16 ids[n_tokens*n_expert_used]
    //
    // ~40 MB per 40K-token turn at n_expert_used = 10, which is cheap enough to leave on for a
    // whole rung. Two call sites write, both under mtx: llama_moe_stream_remap (the single-wave
    // path: one decode token is kind 0, a ubatch that fits the cache in one pass is kind 1) and
    // stage_wave_for_op at wave 0 (the multi-wave path a large ubatch takes; always kind 1, once
    // per layer per ubatch, not once per wave). The wave path has to be here because route_hotness
    // is decode-only: without it the prefill sweep would be invisible to the borrowing log. kind
    // says how many tokens the call routed, not why: with the MTP draft on, a verification ubatch
    // is several tokens and is written as kind 1.
    std::string trace_path;                // LLAMA_MOE_STREAM_TRACE, empty = off
    FILE *      trace_file    = nullptr;
    int64_t     trace_records = 0;         // records written since the last flush
    int64_t     trace_skipped = 0;         // calls whose routing width was not n_expert_used
    std::vector<int16_t> trace_ids;        // scratch: one record's ids, narrowed to i16

    // The model's routing width (n_expert_used), set at load time. The graph runs its warmup pass
    // with n_expert_used = n_expert, so a call's own id count cannot be trusted to be the width -
    // this is what tells a real slip from the 512-wide warmup sweep.
    uint32_t n_expert_used = 0;

    // open on first use (n_expert_used is only known once a router has run), append one record,
    // flush. All three are called with mtx held, except trace_close which runs at shutdown.
    void trace_open_locked(uint32_t n_expert_used);
    void trace_record_locked(int32_t il, uint32_t n_tokens, uint8_t kind, const int32_t * ids, int64_t n);
    void trace_flush_locked();
    void trace_close();

    // streamed layers, i.e. non-null entries of `layers` - the suffix that keeps a model and its
    // draft head from writing over each other's trace and hotness files
    int64_t n_streamed_layers() const;

    // LLAMA_MOE_STREAM_HOTNESS_JSON=<path>: also write the FULL per-layer selection counts there,
    // rewritten on every dump so the last write is the run's final state. Feeds the keep-manifest
    // that gguf_prune_experts.py consumes ({"<layer>": [expert ids]}), via a small python step.
    std::string hotness_json;

    // internals
    void start_workers_locked();
    void worker_loop();
    int32_t pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const;
    void reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot);
    void promote_slot_locked(llama_moe_stream_layer & sl, int32_t slot);

    // multi-pass prefill helpers (called by llama_moe_stream_wave_ids, all under mtx)
    void plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n);
    void plan_pairs_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n); // wave 0: build the plan
    void emit_wave_pairs(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int32_t w, uint32_t n_ids, int64_t n_pairs, int64_t chunk); // the 4 index rows
    void stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids); // make wave w resident + preload next
    void emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int32_t w, uint32_t n_ids, int64_t n_tok); // write the slot ids
};

// The alt copy's shard list, derived from its first shard exactly as the split loader derives -m's
// siblings (llama_get_list_splits). Declared here so a test can hold it to that promise: the two
// sets are indexed by the same file_idx, so a disagreement would read an expert from the wrong
// shard and change the words without crashing. Throws if `first` is not the first shard of an
// n_split-way split.
std::vector<std::string> llama_moe_stream_alt_paths(const std::string & first, size_t n_split);

// Metal servicer entry point (LLAMA_MOE_STREAM_GPU_SLOT=3). Runs on Metal's listener queue with
// the GPU stalled, so it must do the reads and return. user_data is the llama_moe_stream.
void llama_moe_stream_service_gpu(void * user_data, void * state_host, int32_t layer);

// how many tokens a ubatch may carry for the GPU to own residency. Decode and speculative
// verification only: the kernel resolves the pair list serially in one thread, which is the right
// trade for a handful of pairs and the wrong one for a 4096-token prefill. Prefill keeps the CPU
// path, where one graph serves the whole ubatch and the split costs almost nothing anyway.
static const int32_t LLAMA_MOE_GPU_SLOT_MAX_TOKENS = 32;

// callback of the id-remapping custom op inserted by build_moe_ffn
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// remap for THIS layer (src a, as above) plus a lookahead prefetch for the next layer driven by
// this layer's router input (src b, f32 [n_embd] already multiplied by the next layer's gate_inp,
// i.e. b holds the next layer's predicted logits). userdata is a llama_moe_stream_lookahead.
void llama_moe_stream_remap_la(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata);

// Identity on the ubatch token ids, with a side effect: start the loads for every hash-routed
// layer. The hash layers take their tid2eid get_rows index from this op's output, which is what
// orders it before layer 0. Never waits.
void llama_moe_stream_prefetch_hash(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// callbacks of the multi-pass prefill custom ops inserted by build_moe_ffn when a ubatch touches
// more experts than the cache holds; each src[0] is the contiguous selected ids
//   wave_ids:   makes wave w's expert slice resident and emits slot ids (masked pairs park on a pool)
//   wave_mask:  emits 1.0 for pairs belonging to wave w, 0.0 otherwise
//   wave_pairs: same staging, but emits the index rows of wave w's dense pair list (partition path)
void llama_moe_stream_wave_ids  (ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_moe_stream_wave_mask (ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_moe_stream_wave_pairs(ggml_tensor * dst, int ith, int nth, void * userdata);

// index rows of the wave_pairs output, an I32 [chunk, LLAMA_MOE_PAIR_ROWS] tensor
enum llama_moe_pair_row {
    LLAMA_MOE_PAIR_TOK  = 0, // token index, to gather the GEMM input row
    LLAMA_MOE_PAIR_PAIR = 1, // flat pair index t*n_ids + k, to gather (weight_before_ffn) and scatter
    LLAMA_MOE_PAIR_SLOT = 2, // cache slot the GEMM indexes
    LLAMA_MOE_PAIR_EXP  = 3, // original expert id, for the biases and per-expert scales
    LLAMA_MOE_PAIR_ROWS = 4,
};

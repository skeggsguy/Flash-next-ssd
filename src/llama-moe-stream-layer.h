#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

// Per-floor state of the book manager (llama_moe_stream, llama-moe-stream.h): each floor's desk
// slots, the userdata its custom ops carry, and one queued book load.

struct llama_moe_stream;

enum llama_moe_stream_slot_state : uint8_t {
    LLAMA_MOE_STREAM_SLOT_EMPTY    = 0,
    LLAMA_MOE_STREAM_SLOT_LOADING  = 1, // reserved, load queued or in flight
    LLAMA_MOE_STREAM_SLOT_RESIDENT = 2,
};

// one streamed weight tensor (gate/up/down or fused gate_up) of one layer
struct llama_moe_stream_weight {
    ggml_tensor * cache = nullptr; // cache tensor {ne0, ne1, n_slots}

    uint16_t file_idx  = 0; // GGUF split file index
    size_t   offs      = 0; // file offset of the full exps tensor data
    size_t   nb_expert = 0; // bytes per expert slab
};

struct llama_moe_stream_layer;

// Element layout of a layer's GPU residency buffer. Kept in one place because the Metal kernel
// indexes the same buffer with the same arithmetic and the two must not drift.
//
//   [0,           n_slots)            slot -> expert id, -1 = empty
//   [n_slots,   2*n_slots)            slot -> last-use stamp (compare as a wrapping difference)
//   [2*n_slots, 2*n_slots + n_expert) expert -> slot, -1 = not resident
//   tail                              MOE_SLOT_TAIL_COUNT counters
//   requests                          up to n_slots (expert, slot) pairs the CPU must load
struct llama_moe_slot_state_layout {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    int32_t off_slot_expert() const { return 0; }
    int32_t off_last_use()    const { return n_slots; }
    int32_t off_expert_slot() const { return 2*n_slots; }
    int32_t off_tail()        const { return 2*n_slots + n_expert; }
    int32_t off_requests()    const { return off_tail() + 4; }
    int32_t size()            const { return off_requests() + 2*n_slots; }
};

enum llama_moe_slot_tail {
    MOE_SLOT_TAIL_CLOCK = 0, // monotonic use counter; the GPU owns it at mode 3
    MOE_SLOT_TAIL_NREQ  = 1, // (expert, slot) pairs the kernel emitted this call
    MOE_SLOT_TAIL_NBAD  = 2, // verify mismatches, mode 1 only
    MOE_SLOT_TAIL_SEQ   = 3, // call sequence, so a servicer can spot a missed handshake
};

// Userdata for the remap op when one-layer-ahead prefetch is on. The op keeps its normal job for
// THIS layer and additionally predicts the NEXT layer's routing from this layer's router input,
// issuing those loads a layer early.
//
// The prediction is deliberately the cheap one: topk(W_next . x_L). It skips the attention and FFN
// terms between the two layers, so it is a lower bound - measured 72.1% top-6 overlap, and 46.3% of
// the experts that actually STALL at K=6, for 0.35 wasted reads per layer-token. The exact version
// would need layer L+1's attention output, i.e. running attention twice per layer (5 GB/token), so
// it is not worth having.
struct llama_moe_stream_lookahead {
    llama_moe_stream_layer * sl      = nullptr; // this layer, remapped as usual
    llama_moe_stream_layer * sl_next = nullptr; // layer to prefetch into, null = plain remap
    uint32_t                 top_k   = 0;       // how many predicted experts to fetch
    bool                     all     = false;   // LLAMA_MOE_STREAM_LOOKAHEAD_ALL: every token of a small batch
    ggml_tensor *            bias_src = nullptr;// next layer's exp_probs_b, may be null
    bool                     bias_read = false;
    std::vector<float>       bias;              // host copy of bias_src, filled on first use
    std::vector<float>       score;             // scratch [n_expert]
};

struct llama_moe_stream_layer;

// userdata of one wave's custom ops (multi-pass prefill): identifies which pass this is
struct llama_moe_stream_wave {
    llama_moe_stream_layer * sl   = nullptr;
    int32_t                  wave = -1;
};

// per-layer streaming state - also the userdata of the id-remapping custom op
struct llama_moe_stream_layer {
    llama_moe_stream * mgr = nullptr;

    int32_t  il       = -1;
    uint32_t n_expert = 0;
    uint32_t n_slots  = 0;

    std::vector<llama_moe_stream_weight> weights; // 2 (fused gate_up + down) or 3 entries

    // residency state, guarded by mgr->mtx
    std::vector<int32_t>                 slot_expert;   // [n_slots] expert id or -1
    std::vector<uint8_t>                 slot_state;    // [n_slots] llama_moe_stream_slot_state
    std::vector<uint8_t>                 slot_pending;  // [n_slots] slabs still in flight for this slot
    std::vector<uint64_t>                slot_gen;      // [n_slots] reservation generation
    std::vector<int64_t>                 slot_last_use; // [n_slots] LRU stamps
    std::unordered_map<int32_t, int32_t> expert_slot;   // RESIDENT and LOADING entries

    std::vector<uint32_t> route_hotness; // [n_expert] decayed selection counts, for eviction
    std::vector<uint8_t>  seen;          // [n_expert] for cold-miss attribution
    int64_t use_counter = 0;

    // Per-layer borrowing log. The global counters in llama_moe_stream_stats sum these, but a
    // whole-model hit rate cannot say WHICH layers miss - and the miss load is known to be wildly
    // uneven between layers (see llama_moe_stream_hash_router). Incremented at exactly the sites
    // that increment the global counters, so sum over layers == global at every moment.
    // Never decayed, unlike route_hotness: these are cumulative over the run.
    int64_t n_hit       = 0;
    int64_t n_miss      = 0;
    int64_t n_miss_cold = 0;

    // scratch for the remap callback
    std::vector<int32_t> uniq;
    std::vector<uint8_t> touched;
    std::vector<uint8_t> keep;         // [n_slots] slots the current call must not evict
    std::vector<int32_t> demand_slots; // slots the current call waits on

    // wave plan for multi-pass prefill (guarded by mgr->mtx): the touched experts are split into
    // plan_n_waves passes of at most plan_capacity experts each, run one pass at a time
    uint32_t plan_capacity  = 0;  // experts per wave, set at graph build
    uint32_t plan_n_waves   = 0;  // waves of the current call
    int32_t  plan_next_wave = -1; // wave expected to run next (ordering guard)
    std::vector<uint8_t> expert_wave; // [n_expert] wave each touched expert belongs to, 0xff = untouched
    std::vector<int32_t> plan_pool;   // resident slots the masked-out pairs of this wave park on
    std::vector<int32_t> pool_used;   // scratch: pool slots already used in the current token row

    // Pair partitioning (LLAMA_MOE_STREAM_PARTITION=1). The default design runs the expert GEMMs
    // once per wave over EVERY (token, expert) pair and masks the other waves' pairs to zero, so GPU
    // cost scales with wave count - measured at ~8.15 s per wave atop a ~26.9 s fixed cost for a
    // 3798-token prefill, i.e. ~33 s discarded at the default 5 waves.
    //
    // Partitioning by TOKEN is impossible here: a token needs n_expert_used experts and a wave holds
    // only plan_capacity of n_expert, so ~no token's picks fit in one wave. Pairs, however, already
    // belong to exactly one wave via expert_wave[], so the GEMM can run over a dense pair list with
    // the expert dimension collapsed to 1, then scatter back. plan_pair_chunk is fixed at
    // ceil(n_tokens*n_ids / n_waves) because ggml shapes are static; short waves pad.
    uint32_t plan_pair_chunk = 0;                // static per-wave bound, set at graph build
    uint32_t plan_waves_want = 0;                // wave count the graph built, 0 = not partitioned
    std::vector<uint32_t> wave_first, wave_count; // [n_waves] this wave's slice of uniq
    std::vector<std::vector<int32_t>> plan_pair;  // [n_waves][<= plan_pair_chunk] flat idx t*n_ids+k
    std::vector<int32_t>              pair_count; // [n_expert] pairs per expert, for wave balancing

    std::vector<std::unique_ptr<llama_moe_stream_wave>> wave_ud; // stable per-wave op userdata

    // stable userdata for wave w (grows lazily); called at graph build time only
    llama_moe_stream_wave * wave_userdata(int32_t wave, uint32_t capacity);

    // whether the exps tensors passed to build_moe_ffn are this layer's cache tensors
    // (e.g. grovemoe evaluates a second, unstreamed expert group on the same layer index)
    bool matches(const ggml_tensor * gate, const ggml_tensor * up,
                 const ggml_tensor * down, const ggml_tensor * gate_up) const;

    // Device-resident residency state for GPU slot resolution (LLAMA_MOE_STREAM_GPU_SLOT), one I32
    // buffer per streamed layer. At mode 3 the GPU owns every field here and the CPU only services
    // the requests the kernel emits; see llama_moe_slot_state_layout for the element layout.
    ggml_tensor * state      = nullptr;
    int32_t *     state_host = nullptr; // resolved lazily, null until the buffer is real
    bool          state_init = false;   // the buffer starts as garbage - do not read counters yet

    llama_moe_slot_state_layout layout() const { return { (int32_t) n_expert, (int32_t) n_slots }; }

    void publish_state_locked(llama_moe_stream & mgr);

    // mode 3: load whatever the resolve kernel asked for, then let the GPU continue. Runs on
    // Metal's listener queue, NOT on the graph thread.
    void service_requests(llama_moe_stream & mgr);

    // one-layer-ahead prefetch (LLAMA_MOE_STREAM_LOOKAHEAD=K). Set by the model at load time so
    // build_moe_ffn can reach the NEXT layer's router without a signature change.
    ggml_tensor * la_gate_inp = nullptr;   // next layer's ffn_gate_inp
    ggml_tensor * la_bias_src = nullptr;   // next layer's exp_probs_b, may be null
    struct llama_moe_stream_lookahead * la = nullptr;
};

// A layer whose expert choice is a token-id lookup instead of a router matmul (deepseek4 sets
// hash_layer_count = 3). Routing there is known before the graph runs, so the loads can be issued at
// the top of the pass rather than at the layer.
//
// Worth its own path because the miss load is wildly uneven: measured at cache 40, layers 0-2 are 3
// of 43 layers but 36% of steady-state decode misses, and their share GROWS as the cache warms
// (14% -> 36% over 200 tokens). They hash the token id, so they gain no locality - 237 distinct
// experts touched over 200 tokens against 113 slots - while the learned-router layers settle.
struct llama_moe_stream_hash_router {
    llama_moe_stream_layer * sl  = nullptr;
    ggml_tensor *            map = nullptr; // tid2eid {n_expert_used, n_vocab}, I32
    std::vector<int32_t>     rows;          // host copy of map, filled on first use
};

// one queued expert load
struct llama_moe_stream_work {
    llama_moe_stream_layer * sl = nullptr;

    int32_t  expert = -1;
    int32_t  slot   = -1; // -1: a read onto the reading room's belt, not into a desk slot
    int32_t  widx   = -1; // which weight slab of the expert; one work item per slab
    uint64_t gen    = 0;  // stale unless it matches slot_gen[slot] (belt: the part's seq, still filling)

    size_t   ring_offs = 0; // belt only: where on the belt this slab goes

    // the lent belt (llama-moe-room.h): the writing remap's trips and the lookahead's guesses
    uint64_t lent = 0; // a restore: copy this slab back from the lent belt's copy with this seq, no read
    uint64_t save = 0; // the slot's old book is being kept (that copy's seq): its slab is copied out first
};

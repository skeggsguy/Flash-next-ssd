// Desk books first (study #39, LLAMA_MOE_STREAM_DESK_FIRST) in build_moe_ffn: the floor's expert GEMMs cut in two at
// a gate (ggml_moe_gate). What it is and why it is exact: llama-moe-stream-desk-first.h; the manager side (the stop
// and the gate fn) is llama-moe-stream-desk-first.cpp.
//
// The stop is one CPU op writing both id planes, I32 [k, n_tok, 2]: plane 0 the desk's, plane 1 the arrivals'. Then,
// as the reading room runs its groups (llama-graph-moe-room.cpp): the desk's group over plane 0, its down link the
// chain's root; the gate, after the desk's down link and the shared expert (built first by the model, so the GPU runs
// it while the gate waits); the arrivals' group over plane 1, reading the floor's input through the gate and writing
// its down link into the desk's (ggml_mul_mat_id_into). Every pick is on exactly one plane, so each output row is
// written once, by the kernel a run without streaming uses. The down scale and bias come once, after the chain.

#include "llama-graph.h"

#include "llama-adapter.h"
#include "llama-cparams.h"
#include "llama-moe-stream.h"
#include "llama-moe-stream-desk-first.h"

#include <algorithm>

bool llm_graph_context::moe_stream_desk_first_on(int il, int64_t n_tokens) const {
    // a floor with no rows (a read-in ubatch with no output rows, on its last floor) computes nothing: the CPU skips
    // its empty stop, but Metal would still call its gate fn, and the gate would find no stop before it
    if (!cparams.moe_desk_first || n_tokens < 1 || n_tokens > LLAMA_MOE_DESK_FIRST_MAX_TOKENS || mstream == nullptr) {
        return false;
    }
    const llama_moe_stream_layer * sl = mstream->layer(il);
    // one wave (the floor's books all fit on its desk at once), no LoRA on the books (the room's GEMMs carry none),
    // and not the arch whose down GEMM needs F32 precision on its input (build_moe_expert_down)
    return sl != nullptr && (loras == nullptr || loras->empty()) && arch != LLM_ARCH_MISTRAL4 &&
            std::min<int64_t>(sl->n_expert, n_tokens*n_expert_used) <= (int64_t) sl->n_slots;
}

// the stop: src[0] the router's ids, src[1] the lookahead's next-floor scores (null: no lookahead)
ggml_tensor * llm_graph_context::build_moe_desk_first_ids(llama_moe_stream_layer * msl, ggml_tensor * ids_cont,
        ggml_tensor * la_logits) const {
    ggml_tensor * args[2] = { ids_cont, la_logits };
    ggml_tensor * ids2 = ggml_custom_4d(ctx0, GGML_TYPE_I32, ids_cont->ne[0], ids_cont->ne[1], 2, 1, args,
            la_logits ? 2 : 1, la_logits ? llama_moe_stream_desk_first_remap_la : llama_moe_stream_desk_first_remap, 1,
            la_logits ? (void *) msl->la : (void *) msl);
    cb(ids2, "ffn_moe_topk_desk_first", msl->il);
    return ids2;
}

ggml_tensor * llm_graph_context::build_moe_desk_first_experts(llama_moe_stream_layer * msl, const llm_moe_gemms & g,
        ggml_tensor * cur, ggml_tensor * ids2, ggml_tensor * selected_experts) const {
    const int il = g.il;
    GGML_ASSERT(ids2->ne[2] == 2 && cur->ne[1] == 1);

    // the wave planner reads these at run time; this graph has no waves (as the room's)
    msl->plan_pair_chunk = 0;
    msl->plan_waves_want = 0;

    ggml_tensor * ids_desk = ggml_view_2d(ctx0, ids2, ids2->ne[0], ids2->ne[1], ids2->nb[1], 0);
    ggml_tensor * ids_arr  = ggml_view_2d(ctx0, ids2, ids2->ne[0], ids2->ne[1], ids2->nb[1], ids2->nb[2]);
    cb(ids_desk, "ffn_moe_desk_ids", il);
    cb(ids_arr,  "ffn_moe_arriving_ids", il);

    // the desk's books, sent at once
    ggml_tensor * act  = build_moe_expert_act(g, cur, ids_desk, selected_experts, true);
    ggml_tensor * down = ggml_mul_mat_id_into(ctx0, g.down_exps, act, ids_desk, nullptr);
    cb(down, "ffn_moe_desk_down", il);

    // the gate: after the desk's down link and the shared expert, before the arrivals' books
    ggml_tensor * deps[2] = { down, moe_stream_shexp };
    ggml_tensor * gated = ggml_moe_gate(ctx0, cur, deps, moe_stream_shexp ? 2 : 1, msl);
    cb(gated, "ffn_moe_gate_desk_first", il);

    // the arrivals' books, written into the desk's down link
    act  = build_moe_expert_act(g, gated, ids_arr, selected_experts, true);
    down = ggml_mul_mat_id_into(ctx0, g.down_exps, act, ids_arr, down);
    cb(down, "ffn_moe_arriving_down", il);

    ggml_tensor * experts = down; // [n_embd, n_expert_used, n_tok], every row written once
    if (g.down_exps_s) {
        experts = build_moe_expert_scale(experts, g.down_exps_s, selected_experts, cur->ne[2]);
        cb(experts, "ffn_moe_down_scaled", il);
    }
    if (g.down_exps_b) {
        experts = ggml_add_id(ctx0, experts, g.down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }
    return experts;
}

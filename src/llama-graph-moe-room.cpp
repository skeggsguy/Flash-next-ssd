// The fork's reading room (--moe-stream-room) in build_moe_ffn: a floor's expert GEMMs for a read-in
// long enough to read nearly every book. The manager side is llama-moe-room.h.
//
// The floor runs one GEMM group per source of books - the desk, then each part of the belt - each over
// the whole ubatch, each reading an id plane from a CPU op: the pair's slot or record where its source
// holds the pair's book, -1 (skip) where another source does. Every pair is on exactly one plane, and
// the down GEMMs are chained through ggml_mul_mat_id_into into the desk's down link (the chain's root),
// so each output row is written once, by the kernel a run without streaming uses, and the words do not
// depend on which books happen to be on the desk. Then build_moe_ffn's own weighting and combine run.
//
// Per group: its ids op, then gate_up (or up and gate) with into = NULL and its own per-expert scale and
// bias, the activation, and its down link. The down scale and bias are applied once, after the chain:
// the same arithmetic as mul_mat_id -> scale -> bias on the whole tensor. A part op's ordering input is
// a fresh one-element view of the previous group's down link: a tensor is copied in as a split input
// only once per backend, so a shared view would order only the first op, and one element costs 4 bytes.

#include "llama-graph.h"

#include "llama-adapter.h"
#include "llama-moe-room.h"
#include "llama-moe-stream.h"

#include <string>

ggml_tensor * llm_graph_context::build_moe_room_experts(llama_moe_stream_layer * msl, const llm_moe_gemms & g,
        ggml_tensor * cur, ggml_tensor * selected_experts) const {
    llama_moe_room       * room = msl->mgr->room.get();
    llama_moe_room_floor * F    = room->floor(msl->il);
    GGML_ASSERT(F != nullptr && F->views.size() == msl->weights.size());
    const int il = g.il;

    // the wave planner reads these at run time; this graph has no waves (build_moe_stream_masked's rule)
    msl->plan_pair_chunk = 0;
    msl->plan_waves_want = 0;

    // A LoRA adapter on the book weights would need its own GEMMs per group; the streamed books never had
    // them either (the adapter is keyed by the model's tensor names, the desk has its own), so say so.
    for (const auto & lora : *loras) {
        for (const auto & w : msl->weights) {
            std::string name = w.cache->name;
            name = name.substr(0, name.rfind(".stream_cache"));
            if (lora.first->ab_map.count(name) > 0) {
                GGML_ABORT("reading room: the LoRA adapter changes %s, a streamed book weight; "
                           "run with --moe-stream-room 0 and without streaming", name.c_str());
            }
        }
    }

    // the belt groups read each weight's view of the belt where the desk group reads its slots
    llm_moe_gemms belt = g;
    for (ggml_tensor ** t : { &belt.up_exps, &belt.gate_exps, &belt.down_exps, &belt.gate_up_exps }) {
        for (size_t wi = 0; *t != nullptr && wi < msl->weights.size(); wi++) {
            if (msl->weights[wi].cache == *t) {
                *t = F->views[wi];
                break;
            }
        }
        GGML_ASSERT(*t == nullptr || (*t)->view_src == nullptr);
    }

    ggml_tensor * ids_cont = ggml_is_contiguous(selected_experts) ? selected_experts : ggml_cont(ctx0, selected_experts);

    ggml_tensor * down = nullptr; // the chain: the desk's down link, then each part's written into it
    for (int32_t group = -1; group < room->lay.parts; group++) {
        ggml_tensor * args[2] = { ids_cont, nullptr };
        int n_args = 1;
        if (group >= 0) {
            args[1] = ggml_view_1d(ctx0, down, 1, 0);
            n_args  = 2;
        }
        ggml_tensor * ids = ggml_custom_4d(ctx0, GGML_TYPE_I32, ids_cont->ne[0], ids_cont->ne[1], 1, 1, args, n_args,
                group < 0 ? llama_moe_room_desk_ids : llama_moe_room_part_ids, 1, room->op_userdata(msl, group));
        cb(ids, group < 0 ? "ffn_moe_room_desk_ids" : "ffn_moe_room_part_ids", il);

        const llm_moe_gemms & src = group < 0 ? g : belt;
        ggml_tensor * act = build_moe_expert_act(src, cur, ids, selected_experts, true);
        down = ggml_mul_mat_id_into(ctx0, src.down_exps, act, ids, down);
        cb(down, "ffn_moe_room_down", il);
    }

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

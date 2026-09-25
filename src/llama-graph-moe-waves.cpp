// The fork's expert streaming (--moe-stream) in build_moe_ffn, second half: the expert GEMMs, in one
// pass or - when a ubatch touches more experts than the cache holds - once per wave, either over a
// gathered list of each wave's own pairs (partition) or over every pair with the other waves' pairs
// masked to zero. The wave shape comes from llm_moe_stream_graph_shape (llama-graph-moe-stream.cpp).

#include "llama-graph.h"

#include "llama-moe-stream.h"

#include <algorithm>
#include <cstdlib>

ggml_tensor * llm_graph_context::build_moe_stream_experts(llama_moe_stream_layer * msl,
        const llm_moe_stream_shape & shape, const llm_moe_gemms & gemms, ggml_tensor * cur,
        ggml_tensor * ids_gemm, ggml_tensor * selected_experts, int64_t n_expert_used, bool weight_before_ffn) const {
    const uint32_t n_stream_waves = shape.n_waves;
    const bool     use_partition  = shape.partition;

    if (msl && n_stream_waves > 1 && use_partition) {
        return build_moe_stream_partition(msl, shape, gemms, cur, selected_experts, n_expert_used, weight_before_ffn);
    }
    if (msl && n_stream_waves > 1) {
        return build_moe_stream_masked(msl, shape, gemms, cur, selected_experts, n_expert_used);
    }
    return build_moe_expert_gemms(gemms, cur, ids_gemm, selected_experts);
}

// cur is [n_embd, 1, n_tokens], or [n_embd, n_expert_used, n_tokens] when the routing weight was
// folded in before the FFN (weight_before_ffn)
ggml_tensor * llm_graph_context::build_moe_stream_partition(llama_moe_stream_layer * msl,
        const llm_moe_stream_shape & shape, const llm_moe_gemms & gemms, ggml_tensor * cur,
        ggml_tensor * selected_experts, int64_t n_expert_used, bool weight_before_ffn) const {
    const int64_t  n_embd          = cur->ne[0];
    const int64_t  n_tokens        = cur->ne[2];
    const uint32_t n_stream_waves  = shape.n_waves;
    const uint32_t stream_wave_cap = shape.cap;
    const int      il              = gemms.il;

    auto build_expert_gemms = [&](ggml_tensor * cur, ggml_tensor * ids_gemm, ggml_tensor * sel_exp) {
        return build_moe_expert_gemms(gemms, cur, ids_gemm, sel_exp);
    };

    ggml_tensor * experts = nullptr;

    // Pair partitioning: every (token, expert) pair already belongs to exactly one wave, so each
    // wave gathers its own pairs into a dense list, runs the GEMMs once over that list with the
    // expert dimension collapsed to 1, and scatters the rows back. The masked path below instead
    // runs every wave over every pair, which is what makes its cost scale with the wave count.
    const int64_t n_pairs = (int64_t) n_expert_used*n_tokens;

    // Static bound on one wave's pair count. ggml shapes are fixed before the router has run, so
    // this cannot depend on the routing - and it must be a PROOF, not a tuned slack.
    //
    // Percentage slacks over the mean were tried and are unsound at every value, because the unit
    // being balanced is indivisible: all of an expert's pairs go to the wave that stages it. A
    // repetitive prompt can route one expert on EVERY token, and 1025 such pairs overflowed a
    // chunk of 922 on their own - no assignment could have helped.
    //
    // What made the chunk overflow was never the slack: the planner was splitting into a DIFFERENT
    // number of waves than the graph built, so each wave held n_pairs/plan_n_waves pairs against a
    // chunk sized for n_pairs/n_stream_waves. plan_waves_want (below) makes the two agree, and the
    // slack only has to cover ordinary routing imbalance again, which measures at +11-23%.
    //
    // A provable bound is available - no expert exceeds n_tokens pairs, so mean + n_tokens can
    // never be overflowed by one expert - but it costs n_waves*n_tokens of extra GEMM, which at
    // cap 27 is 2.67x the pairs against 1.5x here and measured 18.9 t/s against 73. Not worth it
    // for a bound whose failure mode was a different bug.
    int64_t slack_pct = 50;
    if (const char * s = getenv("LLAMA_MOE_STREAM_PAIR_SLACK")) {
        slack_pct = std::clamp((int64_t) atoi(s), (int64_t) 0, (int64_t) 400);
    }
    const int64_t mean  = (n_pairs + n_stream_waves - 1)/n_stream_waves;

    // A slack over the mean is not enough on its own, because the unit being balanced is
    // INDIVISIBLE: every pair of an expert goes to the wave that stages it. A single expert is
    // picked at most once per token, so it can hold up to n_tokens pairs - and a hot one really
    // does. Real usage aborted here with 2707 pairs in one wave against a 2679 chunk, on a
    // 2976-token prompt: ~91% of tokens routing to one expert, which no balancing can split.
    //
    // Taking n_tokens as a floor covers that case, and is nearly free because it only binds when
    // there are many waves: mean*(1+slack) = 9*n_tokens/n_waves at slack 50 and n_expert_used 6,
    // which already exceeds n_tokens for n_waves <= 9. At 10 waves it costs ~11% more expert GEMM;
    // below that, nothing at all.
    //
    // Not a proof - two experts each over half the tokens in one wave would still overflow - but
    // snake ordering puts the two heaviest in different waves, so the residual case is remote.
    const int64_t chunk = std::min(n_pairs, std::max(mean + mean*slack_pct/100, (int64_t) n_tokens));
    msl->plan_pair_chunk = (uint32_t) chunk;
    msl->plan_waves_want = n_stream_waves; // the planner must produce exactly this many

    // rows the gather draws from: whole tokens, or one row per pair once the routing weight has
    // already been folded in above
    ggml_tensor * cur_rows = weight_before_ffn
        ? ggml_reshape_2d(ctx0, cur, n_embd, n_pairs)
        : ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);

    ggml_tensor * ids_cont = ggml_cont(ctx0, selected_experts);

    // Scatter destination, with one row past the end. Every real row is written by the wave that
    // owns its pair, so it needs no zero-init, and a short wave pads by repeating one of its own
    // pairs - recomputing it and writing the same value to the same row. The extra row is where a
    // wave that owns no pairs at all sends its throwaway output; nothing reads it.
    //
    // It has to be produced by an OP, not by ggml_new_tensor_2d: a bare graph leaf is not reused
    // across layers by the graph allocator, and at 352 MiB per layer that measured as a Metal
    // compute buffer of 9267 MiB against 1139 MiB for the masked path - enough to put
    // --moe-stream-cache 44 and 46 out of reach. Repeating one row is the cheapest op that yields
    // a full-size node; the values are irrelevant since every row read is written by some wave.
    experts = ggml_repeat_4d(ctx0,
            ggml_view_2d(ctx0, cur, n_embd, 1, cur->nb[1], 0), n_embd, n_pairs + 1, 1, 1);

    for (uint32_t w = 0; w < n_stream_waves; w++) {
        ggml_tensor * args[2] = { ids_cont, nullptr };
        int n_args = 1;
        if (w > 0) {
            // ordering token: forces this wave's staging op to run only after the previous wave's
            //   GEMMs consumed their slots; a 1-element view keeps the cross-backend copy tiny
            args[1] = ggml_view_1d(ctx0, experts, 1, 0);
            n_args  = 2;
        }
        ggml_tensor * pairs_w = ggml_custom_4d(ctx0, GGML_TYPE_I32,
                chunk, LLAMA_MOE_PAIR_ROWS, 1, 1,
                args, n_args, llama_moe_stream_wave_pairs, 1, msl->wave_userdata(w, stream_wave_cap));
        cb(pairs_w, "ffn_moe_wave_pairs", il);

        auto row = [&](llama_moe_pair_row r) {
            return ggml_view_1d(ctx0, pairs_w, chunk, (size_t) r*chunk*ggml_element_size(pairs_w));
        };
        ggml_tensor * r_pair = row(LLAMA_MOE_PAIR_PAIR);

        ggml_tensor * cur_w = ggml_get_rows(ctx0, cur_rows,
                weight_before_ffn ? r_pair : row(LLAMA_MOE_PAIR_TOK));    // [n_embd, chunk]
        cur_w = ggml_reshape_3d(ctx0, cur_w, n_embd, 1, chunk);

        // one expert per row, so the GEMM ids and the bias/scale ids are both [1, chunk]
        ggml_tensor * ids_w = ggml_reshape_2d(ctx0, row(LLAMA_MOE_PAIR_SLOT), 1, chunk);
        ggml_tensor * sel_w = ggml_reshape_2d(ctx0, row(LLAMA_MOE_PAIR_EXP),  1, chunk);

        ggml_tensor * e_w = build_expert_gemms(cur_w, ids_w, sel_w);      // [n_embd, 1, chunk]

        experts = ggml_set_rows(ctx0, experts, ggml_reshape_2d(ctx0, e_w, n_embd, chunk), r_pair);
        cb(experts, "ffn_moe_wave_scatter", il);
        ggml_build_forward_expand(gf, experts);
    }

    experts = ggml_view_2d(ctx0, experts, n_embd, n_pairs, experts->nb[1], 0); // drop the scratch row
    experts = ggml_reshape_3d(ctx0, experts, n_embd, n_expert_used, n_tokens);

    return experts;
}

ggml_tensor * llm_graph_context::build_moe_stream_masked(llama_moe_stream_layer * msl,
        const llm_moe_stream_shape & shape, const llm_moe_gemms & gemms, ggml_tensor * cur,
        ggml_tensor * selected_experts, int64_t n_expert_used) const {
    const int64_t  n_tokens        = cur->ne[2];
    const uint32_t n_stream_waves  = shape.n_waves;
    const uint32_t stream_wave_cap = shape.cap;
    const int      il              = gemms.il;

    auto build_expert_gemms = [&](ggml_tensor * cur, ggml_tensor * ids_gemm, ggml_tensor * sel_exp) {
        return build_moe_expert_gemms(gemms, cur, ids_gemm, sel_exp);
    };

    ggml_tensor * experts = nullptr;

    // plan_pair_chunk doubles as the signal that THIS graph is partitioned: the runtime planner
    // keys off it rather than off the env var, so a ubatch that falls back to masking cannot be
    // checked against a chunk left over from a previous, larger ubatch.
    msl->plan_pair_chunk = 0;
    msl->plan_waves_want = 0;

    ggml_tensor * ids_cont = ggml_cont(ctx0, selected_experts);

    for (uint32_t w = 0; w < n_stream_waves; w++) {
        ggml_tensor * args[2] = { ids_cont, nullptr };
        int n_args = 1;
        if (experts != nullptr) {
            // ordering token: forces this wave's ids op to run after the previous wave's GEMMs
            //   consumed their slots; a 1-element view keeps the cross-backend copy tiny
            args[1] = ggml_view_1d(ctx0, experts, 1, 0);
            n_args  = 2;
        }
        ggml_tensor * ids_w = ggml_custom_4d(ctx0, GGML_TYPE_I32,
                ids_cont->ne[0], ids_cont->ne[1], 1, 1,
                args, n_args, llama_moe_stream_wave_ids, 1, msl->wave_userdata(w, stream_wave_cap));
        cb(ids_w, "ffn_moe_wave_ids", il);

        ggml_tensor * e_w = build_expert_gemms(cur, ids_w, selected_experts);

        ggml_tensor * margs[2] = { ids_cont, ids_w };
        ggml_tensor * mask_w = ggml_custom_4d(ctx0, GGML_TYPE_F32,
                1, n_expert_used, n_tokens, 1,
                margs, 2, llama_moe_stream_wave_mask, 1, msl->wave_userdata(w, stream_wave_cap));
        cb(mask_w, "ffn_moe_wave_mask", il);

        e_w = ggml_mul(ctx0, e_w, mask_w); // zero the pairs that belong to other waves

        experts = experts == nullptr ? e_w : ggml_add(ctx0, experts, e_w);
        ggml_build_forward_expand(gf, experts);
    }

    return experts;
}

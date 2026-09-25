// Mirrors upstream build_moe_ffn's expert pipeline (llama-graph.cpp); keep in sync on rebase. Split in
// two at the down GEMM (act, down) so the reading room can chain its down links (llama-graph-moe-room.cpp).
//
// The fork runs these GEMMs from three places - once as upstream does, once per wave on the masked
// path, once per wave over a gathered pair list on the partition path - so upstream's lambda lives
// here as a member, taking the weights it used to capture. The body is upstream's, unchanged.

#include "llama-graph.h"

#include <cmath>

// the expert GEMM pipeline: run once normally, or once per wave under multi-pass prefill;
//   biases and per-expert scales are always indexed by the original selected_experts
// sel_exp is a parameter rather than a capture because the partition path feeds a PAIR SUBSET,
// whose biases and per-expert scales must be indexed by that subset's original expert ids;
// every shape follows from cur and ids_gemm.
ggml_tensor * llm_graph_context::build_moe_expert_gemms(const llm_moe_gemms & g,
        ggml_tensor * cur, ggml_tensor * ids_gemm, ggml_tensor * sel_exp) const {
    return build_moe_expert_down(g, build_moe_expert_act(g, cur, ids_gemm, sel_exp, false), ids_gemm, sel_exp);
}

// The fork's one change to upstream's pipeline: under the reading room (room = true) each GEMM group
// computes only the pairs its source holds (ids of the others are -1) through ggml_mul_mat_id_into, with
// a per-expert scale applied exactly as build_lora_mm_id applies it (keep the two in sync). LoRA adapters
// on the book weights are refused before this runs (build_moe_room_experts).
ggml_tensor * llm_graph_context::build_moe_mm_id(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * ids,
        ggml_tensor * w_s, ggml_tensor * sel_exp, bool room) const {
    if (!room) {
        return build_lora_mm_id(w, cur, ids, w_s, sel_exp);
    }
    ggml_tensor * res = ggml_mul_mat_id_into(ctx0, w, cur, ids, nullptr);
    return w_s ? build_moe_expert_scale(res, w_s, sel_exp, cur->ne[2]) : res;
}

// build_lora_mm_id's per-expert scale: w_s covers all experts, so it is indexed by the original ids
ggml_tensor * llm_graph_context::build_moe_expert_scale(ggml_tensor * res, ggml_tensor * w_s,
        ggml_tensor * sel_exp, int64_t n_tokens) const {
    const int64_t n_expert = w_s->ne[0];
    ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
    s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
    s = ggml_get_rows(ctx0, s, sel_exp);
    return ggml_mul(ctx0, res, s);
}

// up/gate through the activation, upstream's lambda up to its down GEMM
ggml_tensor * llm_graph_context::build_moe_expert_act(const llm_moe_gemms & g,
        ggml_tensor * cur, ggml_tensor * ids_gemm, ggml_tensor * sel_exp, bool room) const {
    ggml_tensor * up_exps        = g.up_exps;
    ggml_tensor * up_exps_b      = g.up_exps_b;
    ggml_tensor * gate_exps      = g.gate_exps;
    ggml_tensor * gate_exps_b    = g.gate_exps_b;
    ggml_tensor * gate_up_exps   = g.gate_up_exps;
    ggml_tensor * gate_up_exps_b = g.gate_up_exps_b;
    ggml_tensor * up_exps_s      = g.up_exps_s;
    ggml_tensor * gate_exps_s    = g.gate_exps_s;
    const llm_ffn_op_type type_op = g.type_op;
    const int             il      = g.il;

    ggml_tensor * up = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_moe_mm_id(gate_up_exps, cur, ids_gemm, up_exps_s, sel_exp, room); // [n_ff*2, n_expert_used, n_tok]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (up_exps_s) {
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, sel_exp);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_moe_mm_id(up_exps, cur, ids_gemm, up_exps_s, sel_exp, room); // [n_ff, n_expert_used, n_tok]
        cb(up, "ffn_moe_up", il);

        if (up_exps_s) {
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, sel_exp);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps) {
            cur = build_moe_mm_id(gate_exps, cur, ids_gemm, gate_exps_s, sel_exp, room); // [n_ff, n_expert_used, n_tok]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_s) {
            cb(cur, "ffn_moe_gate_scaled", il);
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, sel_exp);
            cb(cur, "ffn_moe_gate_biased", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        if (arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0) || arch == LLM_ARCH_HY_V4) {
                            cur = ggml_swiglu_clamp(ctx0, cur, up, limit);
                        } else {
                            up = ggml_clamp(ctx0, up, -limit, limit);
                            cb(up, "ffn_moe_up_clamped", il);
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_moe_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_moe_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, up);
                        }
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_SITU:
            {
                // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
                GGML_ASSERT(has_gate);
                const float beta = hparams.situ_beta;
                const float lb   = hparams.situ_linear_beta;

                ggml_tensor * act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, cur));
                if (lb > 0.0f) {
                    up = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/lb)), lb);
                }
                cur = ggml_mul(ctx0, act, up);
                cb(cur, "ffn_moe_situ", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    return cur;
}

// the down GEMM and its scale and bias, the rest of upstream's lambda
ggml_tensor * llm_graph_context::build_moe_expert_down(const llm_moe_gemms & g,
        ggml_tensor * cur, ggml_tensor * ids_gemm, ggml_tensor * sel_exp) const {
    ggml_tensor * down_exps      = g.down_exps;
    ggml_tensor * down_exps_b    = g.down_exps_b;
    ggml_tensor * down_exps_s    = g.down_exps_s;
    const int     il             = g.il;

    ggml_tensor * experts = build_lora_mm_id(down_exps, cur, ids_gemm, down_exps_s, sel_exp); // [n_embd, n_expert_used, n_tok]
    if (arch == LLM_ARCH_MISTRAL4) {
        // src1 can exceed F16 range
        ggml_prec_set_src(experts, GGML_PREC_F32, 1);
    }
    cb(experts, "ffn_moe_down", il);

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, sel_exp);
        cb(experts, "ffn_moe_down_biased", il);
    }

    return experts;
}

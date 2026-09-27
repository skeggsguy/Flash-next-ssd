#include "qwen4exp-qsa-union.h"

#include "../llama-qsa-picks.h"

#include "ggml.h"

#include <cstring>

llama_qsa_union_mode llama_qsa_union_parse(const char * value, bool * bad) {
    if (bad != nullptr) {
        *bad = false;
    }
    if (value == nullptr) {
        return LLAMA_QSA_UNION_DEFAULT;
    }
    if (strcmp(value, "0") == 0) {
        return LLAMA_QSA_UNION_OFF;
    }
    if (strcmp(value, "bias") == 0) {
        return LLAMA_QSA_UNION_BIAS;
    }
    if (strcmp(value, "1") == 0) {
        return LLAMA_QSA_UNION_ATTN;
    }
    if (bad != nullptr) {
        *bad = true;
    }
    return LLAMA_QSA_UNION_OFF;
}

const char * llama_qsa_union_fallback(const llama_qsa_union_gate & g) {
    if (g.mode != LLAMA_QSA_UNION_ATTN) {
        return nullptr;
    }
    if (!g.flash_attn) {
        return "this context has no flash attention";
    }
    if (!g.kv_f16) {
        return "the context memory is not F16";
    }
    if (!g.same_head) {
        return "K and V heads differ in size";
    }
    return nullptr;
}

std::string llama_qsa_union_describe(llama_qsa_union_mode mode, const char * fallback) {
    if (mode == LLAMA_QSA_UNION_ATTN && fallback != nullptr) {
        return std::string("bias (LLAMA_QSA_UNION=1 asked for union attention, but ") + fallback +
               "), reading in builds the picker's block bias on the GPU from small tables and keeps the mask input "
               "(exact; LLAMA_QSA_UNION=0 uploads the bias from the CPU)";
    }
    switch (mode) {
        case LLAMA_QSA_UNION_BIAS:
            return "bias, reading in builds the picker's block bias on the GPU from small tables "
                   "(exact; LLAMA_QSA_UNION=0 uploads it from the CPU, =1 also hands attention the picks)";
        case LLAMA_QSA_UNION_ATTN:
            return "on, reading in hands attention each token's picks directly (union attention, no mask "
                   "input) and builds the block bias on the GPU (LLAMA_QSA_UNION=0 is the masked path)";
        case LLAMA_QSA_UNION_OFF:
        default:
            return "off, reading in uploads the picker's block bias and the attention mask from the CPU "
                   "(LLAMA_QSA_UNION=bias builds the bias on the GPU, =1 also hands attention the picks)";
    }
}

llama_qsa_union_mode llama_qsa_union_route(const llama_qsa_union_gate & g) {
    if (g.mode == LLAMA_QSA_UNION_OFF) {
        return LLAMA_QSA_UNION_OFF;
    }

    const bool bias = g.block_topk && g.causal && g.one_seq && !g.gather && g.n_stream == 1 &&
                      g.n_tokens >= LLAMA_QSA_UNION_MIN_ROWS && g.n_kv > 0 && g.n_kv <= LLAMA_QSA_UNION_MAX_KV;
    if (!bias) {
        return LLAMA_QSA_UNION_OFF;
    }

    if (g.mode == LLAMA_QSA_UNION_ATTN && g.flash_attn && g.kv_f16 && g.same_head) {
        return LLAMA_QSA_UNION_ATTN;
    }

    return LLAMA_QSA_UNION_BIAS;
}

// rows [t0, t0 + n) of the row index table, as [1, n]: one value a row, broadcast along the blocks or cells
static ggml_tensor * llama_qsa_union_rows(ggml_context * ctx, const llama_qsa_union_tables & tab, int64_t t0, int64_t n) {
    ggml_tensor * q = ggml_view_1d(ctx, tab.q_idx, n, t0*ggml_element_size(tab.q_idx));
    return ggml_reshape_2d(ctx, q, 1, n);
}

ggml_tensor * llama_qsa_union_bias(ggml_context * ctx, const llama_qsa_union_tables & tab, int64_t t0, int64_t n) {
    GGML_ASSERT(tab.on() && t0 >= 0 && t0 + n <= tab.q_idx->ne[0]);

    const int64_t n_blocks = tab.blk_lo->ne[0];

    ggml_tensor * q = llama_qsa_union_rows(ctx, tab, t0, n);

    // [q >= lo] = step(q - (lo - 0.5)), [q < hi] = step((hi - 0.5) - q) (llama-qsa-picks.h)
    ggml_tensor * vis  = ggml_step(ctx, ggml_add(ctx, ggml_repeat_4d(ctx, tab.blk_lo, n_blocks, n, 1, 1), q));
    ggml_tensor * tail = ggml_step(ctx, ggml_sub(ctx, ggml_repeat_4d(ctx, tab.blk_hi, n_blocks, n, 1, 1), q));

    // log(0) = -inf where the block is not visible, then +1e9 on the token's tail (and the spare)
    return ggml_add(ctx, ggml_log(ctx, vis), ggml_scale(ctx, tail, 1e9f));
}

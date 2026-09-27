#include "llama-mtp-record.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"

#include <cstdlib>

bool llama_mtp_record_only_parse(const char * value) {
    return value == nullptr ? LLAMA_MTP_RECORD_ONLY_DEFAULT : std::atoi(value) > 0;
}

bool llama_mtp_record_only_supported(llm_arch arch) {
    return arch == LLM_ARCH_QWEN4EXP;
}

bool llama_mtp_record_only_init(const char * value, bool is_mtp_ctx, llm_arch arch) {
    if (!is_mtp_ctx) {
        return false;
    }
    const bool asked = llama_mtp_record_only_parse(value);
    const bool on    = asked && llama_mtp_record_only_supported(arch);
    if (on) {
        LLAMA_LOG_WARN("%s: apprentice: reading in records K/V only, no attention, books or head "
                "(LLAMA_MTP_RECORD_ONLY=0 runs the whole floor)\n", __func__);
    } else if (asked) {
        LLAMA_LOG_WARN("%s: apprentice: LLAMA_MTP_RECORD_ONLY is set, but %s's apprentice has no record-only "
                "graph; reading in runs the whole floor\n", __func__, llm_arch_name(arch));
    } else {
        LLAMA_LOG_WARN("%s: apprentice: reading in runs the whole floor, though only its K/V is kept "
                "(LLAMA_MTP_RECORD_ONLY=1 records K/V only)\n", __func__);
    }
    return on;
}

void llama_mtp_record_kv(const llm_graph_context & g, llm_graph_input_attn_kv * inp,
        ggml_tensor * k_cur, ggml_tensor * v_cur, int il) {
    ggml_context * ctx0 = g.ctx0;

    // build_attn's store, step for step, without its q
    if (inp->self_k_rot) {
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }
    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    ggml_build_forward_expand(g.gf, v_cur);
    ggml_build_forward_expand(g.gf, k_cur);

    ggml_build_forward_expand(g.gf, inp->mctx->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(g.gf, inp->mctx->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));
}

#include "qwen4exp-qsa-union.h"
#include "qwen4exp-qsa-slice.h"

#include "models.h"
#include "../llama-qsa-picks.h"
#include "llama-kv-cache.h"

#include "ggml.h"

#include <algorithm>
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

ggml_tensor * llama_qsa_union_select(ggml_context * ctx, const llama_qsa_union_tables & tab, ggml_tensor * cells,
        int64_t t0, int64_t n) {
    GGML_ASSERT(tab.on() && cells->type == GGML_TYPE_I32 && ggml_is_contiguous(cells) && cells->ne[1] == n);

    const int64_t width = cells->ne[0];
    const int64_t n_kv  = tab.cell_pos->ne[0];

    ggml_tensor * q = llama_qsa_union_rows(ctx, tab, t0, n);

    // each picked cell's index (an empty cell's is past every token), and whether it is after the row's
    ggml_tensor * pos = ggml_get_rows(ctx, ggml_reshape_2d(ctx, tab.cell_pos, 1, n_kv), ggml_reshape_1d(ctx, cells, width*n));
    pos = ggml_reshape_2d(ctx, pos, width, n);

    ggml_tensor * fut = ggml_step(ctx, ggml_sub(ctx, pos, q));

    // cell - fut*(cell + 1): the cell itself, or -1, which union_build drops (cells are exact in F32 below 2^24)
    ggml_tensor * cells_f = ggml_cast(ctx, cells, GGML_TYPE_F32);
    ggml_tensor * sel     = ggml_sub(ctx, cells_f, ggml_mul(ctx, fut, ggml_scale_bias(ctx, cells_f, 1.0f, 1.0f)));

    return ggml_cast(ctx, sel, GGML_TYPE_I32);
}

// build_attn_qsa_sliced with union attention in place of the mask: the same cache writes, the same picks
// (build_qsa_slice_picks), and for each slice the picked cells at or before each row's token handed to
// ggml_flash_attn_union, whose output goes into the slice's rows of q_cur as the sliced path's does. A
// batch no longer than a slice is one slice.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa_union(
        llm_graph_input_attn_kv *     inp,
        ggml_tensor *                 q_cur,
        ggml_tensor *                 k_cur,
        ggml_tensor *                 v_cur,
        const llama_qsa_slice_parts & parts,
        float                         kq_scale,
        int                           il) {
    // the route asks for an F16 cache, which is never rotated
    GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr);
    GGML_ASSERT(parts.union_attn && parts.tables.on() && parts.tables.cell_pos != nullptr);

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));

    auto spans = llama_qsa_slice_plan(n_tokens, parts.rows);
    if (spans.empty()) {
        spans.push_back({ 0, n_tokens });
    }

    // [D, n_kv, n_head_kv]: the cache's rows as flash attention reads them
    ggml_tensor * k = ggml_permute(ctx0, mctx_cur->get_k(ctx0, il), 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx0, mctx_cur->get_v(ctx0, il), 0, 2, 1, 3);
    GGML_ASSERT(k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && k->ne[3] == 1);

    const int64_t n_kv = k->ne[1];

    GGML_ASSERT(ggml_is_contiguous(q_cur) && q_cur->ne[2] == n_tokens);

    // union attention reads its mask for the dense prefix only, and there is none (n_dense 0), but the op
    // wants an F16 mask of one column a row: one zero leaf as long as the longest slice
    int64_t n_max = 0;
    for (const auto & sp : spans) {
        n_max = std::max(n_max, sp.n);
    }
    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, 1, n_max), 0.0f);

    const auto name = [&](ggml_tensor * t, const char * base, size_t i) {
        cb(t, base, il);
        ggml_format_name(t, "%s-%d-s%zu", base, il, i);
    };

    ggml_tensor * out = ggml_reshape_2d(ctx0, q_cur, q_cur->ne[0]*q_cur->ne[1], n_tokens);
    ggml_tensor * tbl = ggml_reshape_3d(ctx0, parts.blk_cells, parts.r, parts.n_blocks, 1);

    for (size_t i = 0; i < spans.size(); ++i) {
        const int64_t t0 = spans[i].t0;
        const int64_t n  = spans[i].n;

        ggml_tensor * cells = build_qsa_slice_picks(parts, tbl, t0, n, i, il);

        // the visible set: the picked cells at or before the row's token, the rest -1 (dropped)
        ggml_tensor * sel = llama_qsa_union_select(ctx0, parts.tables, cells, t0, n);
        name(sel, "qsa_sel", i);

        ggml_tensor * uids = ggml_union_build(ctx0, sel, (int) n_kv, 8);

        ggml_tensor * q = ggml_view_3d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], n,
                q_cur->nb[1], q_cur->nb[2], t0*q_cur->nb[2]);
        q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [D, n, n_head]

        ggml_tensor * mask = ggml_view_4d(ctx0, zeros, 1, n, 1, 1, zeros->nb[1], zeros->nb[2], zeros->nb[3], 0);

        ggml_tensor * cur = ggml_flash_attn_union(ctx0, q, k, v, mask, uids, 0, kq_scale); // [D, n_head, n]
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], n);
        name(cur, "kqv_out", i);

        if (spans.size() == 1) {
            out = cur;
            break;
        }

        out = ggml_set_inplace(ctx0, out, cur, out->nb[1], out->nb[2], out->nb[3], t0*out->nb[1]);
        ggml_build_forward_expand(gf, out);
    }

    cb(out, "kqv_out", il);

    return out;
}

#include "qwen4exp-qsa-slice.h"

#include "models.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

uint32_t llama_qsa_slice_parse(const char * value) {
    if (value == nullptr) {
        return LLAMA_QSA_SLICE_DEFAULT;
    }

    const long rows = std::atol(value);
    if (rows <= 0) {
        return 0;
    }

    // capped well above any batch (no batch is longer than a slice this wide, so it never slices),
    // so that the cast below cannot wrap a huge value under the floor
    const long rounded = std::min<long>(rows, 1L << 30)/32*32;

    return (uint32_t) (rounded < (long) LLAMA_QSA_SLICE_MIN ? LLAMA_QSA_SLICE_MIN : rounded);
}

std::vector<llama_qsa_slice_span> llama_qsa_slice_plan(int64_t n_tokens, uint32_t rows) {
    std::vector<llama_qsa_slice_span> spans;

    if (rows == 0 || n_tokens <= (int64_t) rows) {
        return spans;
    }

    const int64_t k        = (n_tokens + rows - 1)/rows; // slices
    const int64_t n_chunks = (n_tokens + 31)/32;         // 32-row chunks, the last one maybe short
    const int64_t base     = n_chunks/k;
    const int64_t n_extra  = n_chunks%k;                 // slices that take one chunk more: the last ones

    int64_t t0 = 0;
    for (int64_t i = 0; i < k; ++i) {
        const int64_t chunks = base + (i >= k - n_extra ? 1 : 0);
        const int64_t n      = std::min<int64_t>(32*chunks, n_tokens - t0);

        spans.push_back({ t0, n });
        t0 += n;
    }

    return spans;
}

// Each slice repeats, for its rows, build_qsa_top_k's block path (from the score on) and
// build_attn_qsa's masked path, op for op and in the same shapes but for the row count: the bytes of
// every row are the whole batch's (test-qsa-keep's slice runs compare them). A change to either of
// those two paths must be made here too; the slice tests fail if they drift apart.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa_sliced(
        llm_graph_input_attn_kv *     inp,
        ggml_tensor *                 q_cur,
        ggml_tensor *                 k_cur,
        ggml_tensor *                 v_cur,
        const llama_qsa_slice_parts & parts,
        float                         kq_scale,
        int                           il) {
    // as build_attn_qsa: rotate q/k/v for a quantized cache (once, for the whole batch), store k/v
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));

    const auto spans = llama_qsa_slice_plan(n_tokens, parts.rows);
    GGML_ASSERT(spans.size() > 1);

    ggml_tensor * kq_mask = inp->get_kq_mask(); // [n_kv, n_tokens, 1, 1]
    ggml_tensor * k       = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v       = mctx_cur->get_v(ctx0, il);

    const int64_t idx_dim = parts.q->ne[0];
    const int64_t n_idx_h = parts.q->ne[1];
    const int64_t r       = parts.r;
    const int64_t width   = r*parts.n_blk_sel; // cells a row may see

    GGML_ASSERT(kq_mask->ne[1] == n_tokens && kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1);
    GGML_ASSERT(parts.q->ne[2] == n_tokens && parts.bias->ne[1] == n_tokens);
    GGML_ASSERT(ggml_is_contiguous(q_cur) && ggml_is_contiguous(parts.q));

    // the one source of zeros every slice's picks are written from, as long as the longest slice
    int64_t n_max = 0;
    for (const auto & s : spans) {
        n_max = std::max(n_max, s.n);
    }
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, width, n_max, 1);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // the per-slice tensors carry the layer's names with -s<slice>, so an observer can join them
    const auto name = [&](ggml_tensor * t, const char * base, size_t i) {
        cb(t, base, il);
        ggml_format_name(t, "%s-%d-s%zu", base, il, i);
    };

    // each slice's attention output goes into q_cur's own rows once that slice has read them: the rows
    // of a query and of its output are the same size ([n_embd_head, n_head] a token), and no other
    // slice reads them (the delta-net precedent, delta-net-base.cpp)
    ggml_tensor * out = ggml_reshape_2d(ctx0, q_cur, q_cur->ne[0]*q_cur->ne[1], n_tokens);

    ggml_tensor * tbl = ggml_reshape_3d(ctx0, parts.blk_cells, r, parts.n_blocks, 1);

    for (size_t i = 0; i < spans.size(); ++i) {
        const int64_t t0 = spans[i].t0;
        const int64_t n  = spans[i].n;

        // the picker (build_qsa_top_k from the score on), for these rows
        ggml_tensor * q = ggml_view_3d(ctx0, parts.q, idx_dim, n_idx_h, n,
                parts.q->nb[1], parts.q->nb[2], t0*parts.q->nb[2]);

        ggml_tensor * score = ggml_mul_mat(ctx0, parts.pooled, ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n, 1));
        score = ggml_reshape_4d(ctx0, score, parts.n_blocks, n_idx_h, n, 1);
        score = ggml_relu(ctx0, score);

        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx0, score, parts.n_blocks, n, 1,
                    score->nb[2], score->nb[3], h*score->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
        }
        score = summed;
        name(score, "indexer_score", i);

        ggml_tensor * bias = ggml_view_3d(ctx0, parts.bias, parts.n_blocks, n, 1,
                parts.bias->nb[1], parts.bias->nb[2], t0*parts.bias->nb[1]);
        score = ggml_add(ctx0, score, bias);

        ggml_tensor * blk_top = ggml_top_k(ctx0, score, parts.n_blk_sel);
        name(blk_top, "indexer_top_blk", i);

        ggml_tensor * sel   = ggml_reshape_3d(ctx0, blk_top, parts.n_blk_sel*n, 1, 1);
        ggml_tensor * cells = ggml_get_rows(ctx0, tbl, sel);       // [r, n_blk_sel*n, 1]
        cells = ggml_reshape_4d(ctx0, cells, width, n, 1, 1);
        name(cells, "indexer_top_k", i);

        // the mask (build_attn_qsa's masked path), for these rows
        ggml_tensor * mask = ggml_view_4d(ctx0, kq_mask, kq_mask->ne[0], n, 1, 1,
                kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], t0*kq_mask->nb[1]);

        ggml_tensor * mask_all = ggml_fill(ctx0, mask, -INFINITY);
        mask_all = ggml_view_4d(ctx0, mask_all, 1, mask_all->ne[0], mask_all->ne[1], mask_all->ne[3],
                mask_all->nb[0], mask_all->nb[1], mask_all->nb[2], 0);

        ggml_tensor * top_k_3d = ggml_view_4d(ctx0, cells, cells->ne[0], cells->ne[1], cells->ne[3], 1,
                cells->nb[1], cells->nb[2], cells->ne[3]*cells->nb[3], 0);

        ggml_tensor * zeros_n = ggml_view_4d(ctx0, zeros, 1, width, n, 1, zeros->nb[1], zeros->nb[2], zeros->nb[3], 0);

        ggml_tensor * mask_top_k = ggml_set_rows(ctx0, mask_all, zeros_n, top_k_3d);
        mask_top_k = ggml_view_4d(ctx0, mask_top_k, mask_top_k->ne[1], mask_top_k->ne[2], 1, mask_top_k->ne[3],
                mask_top_k->nb[2], mask_top_k->nb[3], mask_top_k->nb[3], 0);
        mask_top_k = ggml_add(ctx0, mask_top_k, mask);

        // attention for these rows
        ggml_tensor * q_rows = ggml_view_3d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], n,
                q_cur->nb[1], q_cur->nb[2], t0*q_cur->nb[2]);

        ggml_tensor * cur = build_attn_mha(q_rows, k, v, nullptr, mask_top_k, nullptr, nullptr,
                parts.sparse_fa ? width : 0, kq_scale, il);
        name(cur, "kqv_out", i);

        GGML_ASSERT(cur->ne[0] == out->ne[0] && cur->ne[1] == n);

        out = ggml_set_inplace(ctx0, out, cur, out->nb[1], out->nb[2], out->nb[3], t0*out->nb[1]);
        ggml_build_forward_expand(gf, out);
    }

    cb(out, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, inp->self_v_rot);
    }

    return out;
}

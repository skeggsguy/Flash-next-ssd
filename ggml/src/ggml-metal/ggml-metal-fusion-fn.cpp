// Fix 2 (GGML_METAL_FUSION_FN): the merged chains' patterns and the switch. See ggml-metal-fusion-fn.h.
//
// The patterns are ordinary entries of the Metal fusion table, so the graph optimizer packs them
// before its reorder (a pattern must be contiguous in the graph's own order, before the reorder) and
// the op encoders find them with ggml_metal_fusion_next. raw_ops must list the graph's nodes exactly,
// views included: ggml_metal_fusion_add_alloc_deps matches it against the graph to keep every input of
// a merged chain alive until the chain's last node, since the merged kernel reads them all at once.

#include "ggml-metal-fusion-fn.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

bool ggml_metal_fusion_fn_enabled(void) {
    static const bool enabled = [] {
        const char * v = getenv("GGML_METAL_FUSION_FN");
        return v != nullptr && strcmp(v, "1") == 0;
    }();
    return enabled;
}

// ---- checks ----------------------------------------------------------------

// A merged kernel reads the chain's external inputs while it writes the chain's last node. The alloc
// deps keep them apart when the chain was contiguous before the reorder; one that became contiguous only
// after it has none, so the encoder refuses a chain whose output overlaps an input. Exactly the same
// bytes are allowed where every thread reads its element before writing it (elementwise in place).
static bool ggml_metal_fusion_fn_apart(const ggml_tensor * dst, const ggml_tensor * src, bool same_ok) {
    ggml_backend_buffer_t bd = dst->view_src ? dst->view_src->buffer : dst->buffer;
    ggml_backend_buffer_t bs = src->view_src ? src->view_src->buffer : src->buffer;

    const ggml_metal_buffer_id d = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bd->context, dst);
    const ggml_metal_buffer_id s = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bs->context, src);

    if (d.metal != s.metal) {
        return true;
    }
    if (same_ok && d.offs == s.offs && ggml_nbytes(dst) == ggml_nbytes(src)) {
        return true;
    }
    return d.offs + ggml_nbytes(dst) <= s.offs || s.offs + ggml_nbytes(src) <= d.offs;
}

// P4: hc mix's SCALE(1/hc) + SILU (build_hc_mix). The merged kernel is kernel_unary's SILU with the
// SCALE step in front, the same expression and vector width as the SCALE kernel. The generic checks
// (unsafe = false) already hold it to a chain of one shape whose SCALE feeds only the SILU.
static bool ggml_metal_fusion_fn_check_scale_unary(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_cgraph            * gf,
        const int                    * node_idxs,
              int                      idx,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * scale = nodes[0];
    const ggml_tensor * un    = nodes[1];
    const ggml_tensor * src   = scale->src[0];

    if (un->src[0] != scale || ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        return false;
    }

    // F32 only: a half chain would also have to round through the SCALE's half store
    if (src->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32) {
        return false;
    }

    // the merged kernel reads with the SCALE's layout and writes with the SILU's; flat is the simple case
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(scale) || !ggml_is_contiguous(un)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL && !ggml_metal_fusion_fn_apart(un, src, true)) {
        return false;
    }

    return true;
}

// P1: hc combine's scatter weights, SCALE(1/hc) + SIGMOID + SCALE(2), folded into the DSV4_HC_POST
// that reads them (build_hc_combine). The merged kernel computes each token's 4 weights as one float4,
// the width the unary kernels use on a row of 4, then HC_POST's own arithmetic. Not an elision chain
// (HC_POST reads the weights as src[2], and x and residual beside them), so this check is the sole
// validator (unsafe = true).
static bool ggml_metal_fusion_fn_check_hc_post_w(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_cgraph            * gf,
        const int                    * node_idxs,
              int                      idx,
              ggml_metal_fusion_mode   mode) {
    const ggml_tensor * s0     = nodes[0];
    const ggml_tensor * g      = nodes[1];
    const ggml_tensor * s1     = nodes[2];
    const ggml_tensor * post   = nodes[3];
    const ggml_tensor * inject = s0->src[0];

    if (g->src[0] != s0 || ggml_get_unary_op(g) != GGML_UNARY_OP_SIGMOID || s1->src[0] != g) {
        return false;
    }

    // the weights are HC_POST's `post` only; with a comb matrix (src[3]) HC_POST is another kernel
    if (post->src[2] != s1 || post->src[3] != nullptr || post->src[0] == s1 || post->src[1] == s1) {
        return false;
    }

    const ggml_tensor * f32s[] = { inject, s0, g, s1, post, post->src[0], post->src[1] };
    for (const ggml_tensor * t : f32s) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }

    // one float4 of weights per token, as the unary kernels read a row of hc = 4
    if (inject->ne[0] != 4 || post->src[1]->ne[1] != 4 || !ggml_is_contiguous(inject) ||
        !ggml_are_same_shape(inject, s0) || !ggml_are_same_shape(inject, g) || !ggml_are_same_shape(inject, s1)) {
        return false;
    }

    // four consecutive graph nodes, whose intermediates nothing else reads
    if (node_idxs[idx + fusion->n_ops - 1] - node_idxs[idx] != fusion->n_ops - 1) {
        return false;
    }
    const int outputs[1] = { node_idxs[idx + fusion->n_ops - 1] };
    if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, fusion->n_ops, fusion->ops, outputs, 1)) {
        return false;
    }

    // HC_POST's own inputs keep today's rule; inject is read by the merged kernel only
    if (mode == GGML_METAL_FUSION_FULL && !ggml_metal_fusion_fn_apart(post, inject, false)) {
        return false;
    }

    return true;
}

// ---- patterns --------------------------------------------------------------

static const ggml_op ops_fn_scale_unary[] = { GGML_OP_SCALE, GGML_OP_UNARY };
static const ggml_op ops_fn_hc_post_w[]   = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

static const std::vector<ggml_metal_fusion> ggml_metal_fusion_fn_patterns = {
    { GGML_METAL_FUSION_FN_SCALE_UNARY, ops_fn_scale_unary, 2, ops_fn_scale_unary, 2, false, ggml_metal_fusion_fn_check_scale_unary },
    { GGML_METAL_FUSION_FN_HC_POST_W,   ops_fn_hc_post_w,   4, ops_fn_hc_post_w,   4, true,  ggml_metal_fusion_fn_check_hc_post_w },
};

const ggml_metal_fusion * ggml_metal_fusion_fn_table(const ggml_metal_fusion * base, int n_base, int * n) {
    if (!ggml_metal_fusion_fn_enabled()) {
        *n = n_base;
        return base;
    }

    // built once, then only read: the fusion counters index entries by their address in this table
    static const std::vector<ggml_metal_fusion> table = [base, n_base] {
        std::vector<ggml_metal_fusion> t(base, base + n_base);
        t.insert(t.end(), ggml_metal_fusion_fn_patterns.begin(), ggml_metal_fusion_fn_patterns.end());
        return t;
    }();

    *n = (int) table.size();
    return table.data();
}

bool ggml_metal_fusion_fn_is_fn(ggml_metal_fusion_id id) {
    for (const ggml_metal_fusion & f : ggml_metal_fusion_fn_patterns) {
        if (f.id == id) {
            return true;
        }
    }
    return false;
}

// ---- P8: the command-buffer split ------------------------------------------

int ggml_metal_fusion_fn_split(const ggml_cgraph * gf, int n_split) {
    if (n_split <= 0 || n_split >= gf->n_nodes) {
        return n_split;
    }

    // the non-empty nodes around the split, as the encoder sees them (views and reshapes filtered)
    int idxs[2*GGML_METAL_FUSION_MAX];
    int n_idxs = 0;
    const int i0 = std::max(0, n_split - GGML_METAL_FUSION_MAX + 1);
    const int i1 = std::min(gf->n_nodes, n_split + GGML_METAL_FUSION_MAX);
    for (int i = i0; i < i1; ++i) {
        if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
            idxs[n_idxs++] = i;
        }
    }

    // A group reaching past the split starts at most GGML_METAL_FUSION_MAX - 1 nodes before it. Single
    // patterns are looked up (not the optimizer's back-to-back packing), and the earliest start that
    // reaches past the split is taken: a later one could be the tail of that same group, and splitting
    // earlier than needed only moves nodes to the next buffer. Any split is correct, since every merge
    // is exact; this only keeps a group merged.
    for (int i = 0; i < n_idxs && idxs[i] < n_split; ++i) {
        int len = 1;
        const ggml_metal_fusion * f = ggml_metal_fusion_next(gf, idxs, n_idxs, i, GGML_METAL_FUSION_STRUCTURAL, &len);
        if (f && idxs[i + len - 1] >= n_split) {
            return idxs[i];
        }
    }

    return n_split;
}

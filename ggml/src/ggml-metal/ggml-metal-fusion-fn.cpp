// Fix 2 (GGML_METAL_FUSION_FN): the merged chains' patterns and the switch. See ggml-metal-fusion-fn.h.
//
// The patterns are ordinary entries of the Metal fusion table, so the graph optimizer packs them
// before its reorder (a pattern must be contiguous in the graph's own order, before the reorder) and
// the op encoders find them with ggml_metal_fusion_next. raw_ops must list the graph's nodes exactly,
// views included: ggml_metal_fusion_add_alloc_deps matches it against the graph to keep every input of
// a merged chain alive until the chain's last node, since the merged kernel reads them all at once.

#include "ggml-metal-fusion-fn.h"

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

// ---- patterns --------------------------------------------------------------

static const std::vector<ggml_metal_fusion> ggml_metal_fusion_fn_patterns = {
    // (the patterns land with their kernels: P4, P1, P3)
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

// Fix 2: fewer GPU jobs per written token (GGML_METAL_FUSION_FN).
//
// A written token of Qwen3.8-Flash-Next runs ~3,500 GPU jobs, many of them tiny elementwise or
// routing steps whose start-up costs more than their work. The switch merges a few of those chains
// into one job each. Every merge is exact: the merged kernel computes each step with the same
// expression, vector width and reduction order as the step's own kernel, so the words written are
// the same with the switch on or off. The patterns join the Metal fusion table (ggml-metal-fusion.h)
// only when the switch is on, so with it off the table, the graph and every kernel are today's.
//
//   GGML_METAL_FUSION_FN=1   merge (read once, the first time the fusion table is consulted)
//   GGML_METAL_FUSION_FN=0   today's jobs (the default until the F2-fusion rung passes)

#pragma once

#include "ggml-metal-fusion.h"
#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

// whether GGML_METAL_FUSION_FN is on; read once per process
bool ggml_metal_fusion_fn_enabled(void);

// the fusion table: `base` alone while the switch is off, `base` followed by the FN patterns when it
// is on. The returned table is stable for the life of the process (fusion counters index into it).
const struct ggml_metal_fusion * ggml_metal_fusion_fn_table(const struct ggml_metal_fusion * base, int n_base, int * n);

// true for the ids of the FN patterns (those ggml_metal_op_fn_encode handles)
bool ggml_metal_fusion_fn_is_fn(enum ggml_metal_fusion_id id);

// P8: where the second command buffer should start so a fusion group is not cut in two. A group cut
// by the split runs unmerged (still exact, but more jobs); returns the start of the earliest group in
// the window that reaches past n_split, else n_split.
int ggml_metal_fusion_fn_split(const struct ggml_cgraph * gf, int n_split);

// encode one merged FN chain: nodes[0..n) are the chain's non-empty nodes in graph order.
// Returns n (the number of nodes consumed).
int ggml_metal_op_fn_encode(
        ggml_metal_library_t              lib,
        ggml_metal_encoder_t              enc,
        const struct ggml_metal_fusion  * fusion,
        const struct ggml_tensor * const * nodes,
        int                               n);

#ifdef __cplusplus
}
#endif

// The small-batch mat-vec (mul_mv_ext) up to 16 columns (GGML_METAL_MV_EXT_MAX). WRITING-PLAN.md step 3b.
//
// With the apprentice at depth 8 the library checks 9 words at once. Upstream's small-batch kernels take
// 2-8 columns (4-8 for the K-quants), so a 9-column check batch fell to the large-batch kernel (mul_mm):
// the staff's Q8_0 matrices cost ~2.3x as much a cycle and writing at depth 8 ran a third slower than a
// prototype with the limit at 16 (the study's results/engine/write-analysis/summary.md, 2026-09-28).
//
// The kernel already handles any column count: each threadgroup walks r1ptg columns and masks the ones
// past the end, and each output's sum depends only on nxpsg (the threads along a row), never on r1ptg.
// So 9-16 columns need no new kernel, only an r1ptg among the instantiated 2..5, and 2-8 columns are
// untouched (the same r1ptg, the same kernels, the same bytes). 9-16 columns move from mul_mm's sums to
// mul_mv_ext's, a rounding-level change in the words (the depth-change class), which is why the limit
// stays a switch:
//
//   GGML_METAL_MV_EXT_MAX unset      16: up to 16 columns (10 on matrices of 4,096 rows or more)
//   GGML_METAL_MV_EXT_MAX=8 (or 0)   today's graph: 9+ columns go to mul_mm
//   GGML_METAL_MV_EXT_MAX=9..15      that many columns at most
//
// Read once at device init (ggml_metal_device_props.mv_ext_max); the startup line says which.

#pragma once

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// today's limit and the widest this routine takes
#define GGML_METAL_MV_EXT_MAX_OLD 8
#define GGML_METAL_MV_EXT_MAX_NEW 16

// the switch's value: unset or empty -> 16; "0" -> 8 (off); a number is clamped to 8..16; anything else
// (not a whole number) -> 16 and *bad is set, so the caller can warn
int ggml_metal_mv_ext_max_parse(const char * s, bool * bad);

// the most columns mul_mv_ext takes for a matrix of ne01 rows, given the switch's value (8..16): the switch's
// value up to 4,095 rows, at most 10 from 4,096 rows (where 11+ columns lose to mul_mm; the source has the
// numbers). The same for every type the kernel takes; the caller still checks the type and the fewest
// columns (2, or 4 for the K-quants).
#define GGML_METAL_MV_EXT_ROWS_BIG 4096
#define GGML_METAL_MV_EXT_MAX_BIG  10
int ggml_metal_mv_ext_limit(int64_t ne01, int mv_ext_max);

// columns per threadgroup for 9-16 columns (2-8 keep upstream's table in ggml_metal_op_mul_mat); 0 for
// any other count. Always one of the instantiated 2..5.
int ggml_metal_mv_ext_r1ptg_wide(int ne11);

#ifdef __cplusplus
}
#endif

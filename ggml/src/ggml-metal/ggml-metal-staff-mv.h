// #23: the staff's Q8_0 mat-vec for a check of 2-16 words, on the matrix units (GGML_METAL_STAFF_MV).
// The study's STAFF-MATRIX-PLAN.md step 2.
//
// When the apprentice's guesses are checked, the library runs every staff matrix (always-loaded Q8_0 weights)
// against 2-16 columns at once. Today's small-batch kernel (mul_mv_ext) is held back by its multiplies past
// 4 columns (6 words cost ~1.6x one word on 10240 x 2560), and past 10 columns big matrices fall to mul_mm.
// This kernel (kernels/mul_mv_staff.metal) reads each weight once for every column and hands the multiplies
// to matmul2d: the int8 codes straight from device memory, the columns as half, each block's scale applied to
// the F32 partial. The numbers: the study's results/engine/staff-matrix/summary.md.
//
// It takes Q8_0 x F32 on matrices of 2,048 rows or more (fewer rows can't keep the GPU busy with one
// threadgroup a 32 rows), from 4 columns where mul_mv_ext takes the product (k a multiple of 128; at 2-3
// columns mul_mv_ext is as fast or faster) and from 2 where it doesn't (hc up, k 320, goes to the one-word
// kernel column by column today), up to 16. Its sums are not today's (half columns, another order): a
// rounding-level change, as MV16 and U-union were.
//
//   GGML_METAL_STAFF_MV=1              on (when the device has the Metal 4 tensor API, M5 on)
//   GGML_METAL_STAFF_MV unset or "0"   today's route (mul_mv_ext, mul_mm past its limit): the fork's default from
//                                      2026-10-15 (the study's C-staff rung kept it off; src/llama-flashnext.h)
//
// Read once at device init (ggml_metal_device_props.staff_mv). The start-up line says which: at INFO from the
// device init, and again at WARN from each context's init (ggml_metal_init), since a server installs its log
// callback after the device init and shows the library's INFO only at -lv 4.

#pragma once

#include "ggml.h"
#include "ggml-metal-device.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// the most columns, the fewest rows, and the fewest columns where mul_mv_ext would take the product
#define GGML_METAL_STAFF_MV_MAX      16
#define GGML_METAL_STAFF_MV_ROWS     2048
#define GGML_METAL_STAFF_MV_MIN_EXT  4

// the switch's value: "1" -> 1 (on); unset, empty or "0" -> 0 (today's route); anything else -> 0 and *bad
// is set, so the caller can warn
int ggml_metal_staff_mv_parse(const char * s, bool * bad);

// the most columns the staff mat-vec takes: 0 when the switch is off or the device has no tensor API (the
// kernel is matmul2d's), else GGML_METAL_STAFF_MV_MAX
int ggml_metal_staff_mv_limit(int staff_mv, bool has_tensor);

// the route's start-up line, without a "name: " prefix or newline: "staff mat-vec (Q8_0)  = on, up to 16 columns
// on 2048+ rows (GGML_METAL_STAFF_MV)", "... = off: it needs the tensor API (...)" or "... = off, today's route
// (GGML_METAL_STAFF_MV=0)" (the study's gates and rungs grep for these); written into buf, cut to size; returns buf
const char * ggml_metal_staff_mv_route_line(char * buf, size_t size, int staff_mv, bool has_tensor);

// the fewest columns for rows of ne00 values: GGML_METAL_STAFF_MV_MIN_EXT where mul_mv_ext takes the product
// (ne00 a multiple of 128), else 2
int ggml_metal_staff_mv_min_cols(int64_t ne00);

// whether the staff mat-vec takes this MUL_MAT: src0 Q8_0 with GGML_METAL_STAFF_MV_ROWS rows or more, src1 F32
// with contiguous rows, min_cols..limit columns
bool ggml_metal_staff_mv_takes(const struct ggml_tensor * op, int limit);

// encode the MUL_MAT op (ggml_metal_staff_mv_takes must have said yes); returns the nodes used (1)
int ggml_metal_staff_mv_encode(ggml_metal_library_t lib, ggml_metal_encoder_t enc, const struct ggml_tensor * op);

#ifdef __cplusplus
}
#endif

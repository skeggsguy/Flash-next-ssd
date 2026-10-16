// Desk books first (study #39): the Metal side of GGML_OP_MOE_GATE (ggml.h).
//
// A floor's GPU stretch is sent as a main command buffer, then the rest (cb[0..n_cb)), queued in that order up
// front. With a gate fn set (ggml_backend_metal_set_moe_gate) and a gate in the graph, the main buffer ends right
// after the graph's first gate: it carries the desk books and the shared expert and is sent at once. The rest is
// encoded, the fn is called for every gate in the graph (in node order, on the thread computing the graph: the
// floor's wait for its trips), and only then is the rest sent, so the arrivals' GEMMs read books that have landed.
// The command-buffer boundary is what makes the CPU's stores visible to them. Both encoding paths do this: the
// plain one and the commit of a graph encoded ahead (GGML_METAL_ENCODE_AHEAD).
//
// With no fn set, nothing changes: no scan, the usual split. With n_cb == 0 (one buffer: fusion stats) the fn runs
// before that one buffer is sent: exact, with no overlap.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cgraph;
struct ggml_tensor;
struct ggml_backend_buffer;

typedef void (*ggml_metal_moe_gate_t)(void * user_data, void * gate_userdata);

// the index of gf's first gate node, -1 if none. Aborts if a gate's view is not in a Metal buffer (its source was
// computed on another backend), before anything reads it.
int ggml_metal_moe_gate_first(const struct ggml_cgraph * gf);

// aborts unless the gate's view is in a Metal buffer; also run where a gate is encoded, so a gate with no fn set
// (no scan) is refused too, before the kernels after it in its command buffer are encoded
void ggml_metal_moe_gate_check(const struct ggml_tensor * gate);

// calls fn for each gate in gf from node idx_first on; cut: the command buffers were split at idx_first (counts)
void ggml_metal_moe_gate_run(const struct ggml_cgraph * gf, int idx_first, bool cut, ggml_metal_moe_gate_t fn, void * user_data);

// for tests: gate fn calls so far, and of those, calls made with the main buffer already sent (the overlap)
void ggml_metal_moe_gate_counts(int64_t * calls, int64_t * cut);

// ggml-metal.cpp: whether buffer is one of Metal's (the gate's view must be)
bool ggml_metal_moe_gate_buffer_ok(struct ggml_backend_buffer * buffer);

#ifdef __cplusplus
}
#endif

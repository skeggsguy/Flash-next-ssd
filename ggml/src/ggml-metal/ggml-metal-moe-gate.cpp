// Desk books first (study #39): the gate's scan and fn calls. See ggml-metal-moe-gate.h.

#include "ggml-metal-moe-gate.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <atomic>

static std::atomic<int64_t> g_calls{0};
static std::atomic<int64_t> g_cut{0};

void ggml_metal_moe_gate_check(const struct ggml_tensor * gate) {
    // the arrivals' kernels read through the gate's view; Metal casts any buffer to its own, so a view of a CPU op's
    // output would be read from CPU memory (a bus error or garbage), not refused
    ggml_backend_buffer * buffer = gate->view_src ? gate->view_src->buffer : gate->buffer;
    if (buffer == nullptr || !ggml_metal_moe_gate_buffer_ok(buffer)) {
        GGML_ABORT("moe gate '%s': its view is not in a Metal buffer (buffer %s); gate a tensor computed on Metal",
                gate->name, buffer ? ggml_backend_buffer_name(buffer) : "none");
    }
}

int ggml_metal_moe_gate_first(const struct ggml_cgraph * gf) {
    int first = -1;
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * node = gf->nodes[i];
        if (node->op != GGML_OP_MOE_GATE) {
            continue;
        }
        ggml_metal_moe_gate_check(node);
        if (first < 0) {
            first = i;
        }
    }
    return first;
}

void ggml_metal_moe_gate_run(const struct ggml_cgraph * gf, int idx_first, bool cut, ggml_metal_moe_gate_t fn, void * user_data) {
    for (int i = idx_first; i < gf->n_nodes; ++i) {
        const ggml_tensor * node = gf->nodes[i];
        if (node->op != GGML_OP_MOE_GATE) {
            continue;
        }
        fn(user_data, ggml_moe_gate_get_userdata(node));
        g_calls.fetch_add(1, std::memory_order_relaxed);
        if (cut) {
            g_cut.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void ggml_metal_moe_gate_counts(int64_t * calls, int64_t * cut) {
    *calls = g_calls.load(std::memory_order_relaxed);
    *cut   = g_cut.load(std::memory_order_relaxed);
}

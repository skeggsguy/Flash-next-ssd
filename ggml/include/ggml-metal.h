// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

// CPU servicer for MoE expert streaming. When the graph carries GGML_OP_MOE_SLOT_RESOLVE ops that
// own residency (no reference source), the backend stalls the GPU on a shared event after each one
// and calls this on Metal's listener queue so the slabs the kernel asked for can be read in. Set it
// before the first graph runs; passing NULL disables the ownership mode.
//
// state_host points at the op's state buffer, which must live in shared storage.
typedef void (*ggml_backend_metal_moe_servicer_t)(void * user_data, void * state_host, int32_t layer);

GGML_BACKEND_API void ggml_backend_metal_set_moe_servicer(ggml_backend_t backend, ggml_backend_metal_moe_servicer_t fn, void * user_data);

// Desk books first (study #39): the fn a GGML_OP_MOE_GATE node (ggml_moe_gate, ggml.h) calls. When a graph holds a
// gate, its first command buffer ends right after the gate and is sent; the rest is encoded; then fn(user_data,
// the gate's userdata) runs for every gate in the graph, in node order, on the thread computing the graph, and
// only after it returns is the rest sent. So the GPU runs everything before the gate (the desk's books, the shared
// expert) while fn waits, and the kernels after it see what fn's thread wrote to shared buffers before returning.
// NULL (the default) turns it off: gates are no-ops and the split is the usual one. Set it before the first graph.
typedef void (*ggml_backend_metal_moe_gate_t)(void * user_data, void * gate_userdata);

GGML_BACKEND_API void ggml_backend_metal_set_moe_gate(ggml_backend_t backend, ggml_backend_metal_moe_gate_t fn, void * user_data);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

#ifdef __cplusplus
}
#endif

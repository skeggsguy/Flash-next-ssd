// Fix 2 (GGML_METAL_FUSION_FN): the encoders of the merged chains. See ggml-metal-fusion-fn.h.
//
// Each encoder dispatches its merged kernel with the same grid the chain's own kernels would use for
// the steps it copies (thread counts, vector widths), since a reduction's order and a kernel's
// vector width are part of what makes the merge exact.

#include "ggml-metal-fusion-fn.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-metal-impl.h"

#include <algorithm>

static ggml_metal_buffer_id ggml_metal_fn_buffer_id(const ggml_tensor * t) {
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id((ggml_metal_buffer_t) buffer->context, t);
}

int ggml_metal_op_fn_encode(
        ggml_metal_library_t        lib,
        ggml_metal_encoder_t        enc,
        const ggml_metal_fusion   * fusion,
        const ggml_tensor * const * nodes,
        int                         n) {
    GGML_UNUSED(lib);
    GGML_UNUSED(enc);
    GGML_UNUSED(nodes);
    GGML_UNUSED(ggml_metal_fn_buffer_id);

    GGML_UNUSED(n);

    switch (fusion->id) {
        default:
            break;
    }

    GGML_ABORT("%s: not an FN fusion (id %d)", __func__, (int) fusion->id);
}

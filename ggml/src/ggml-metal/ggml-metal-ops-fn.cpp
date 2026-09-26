// Fix 2 (GGML_METAL_FUSION_FN): the encoders of the merged chains. See ggml-metal-fusion-fn.h.
//
// Each encoder dispatches its merged kernel with the variant the chain's own kernels would pick for
// the steps it copies (vector width, thread counts), since a kernel's vector width and a reduction's
// order are part of what makes a merge exact.

#include "ggml-metal-fusion-fn.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-metal-impl.h"

#include <algorithm>
#include <cstdio>

static ggml_metal_buffer_id ggml_metal_fn_buffer_id(const ggml_tensor * t) {
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id((ggml_metal_buffer_t) buffer->context, t);
}

// P4: SCALE + UNARY (silu). kernel_unary for the SILU with its SCALE step switched on
// (FC_FUSION_FN + 0), in the variant ggml_metal_library_get_pipeline_unary picks for this shape:
// float4 when a row is a multiple of 4, flat under 32768 elements (the check makes all three contiguous)
static int ggml_metal_fn_encode_scale_unary(ggml_metal_library_t lib, ggml_metal_encoder_t enc, const ggml_tensor * const * nodes) {
    const ggml_tensor * scale = nodes[0];
    const ggml_tensor * op    = nodes[1];
    const ggml_tensor * src   = scale->src[0];

    const int  op_num = OP_UNARY_NUM_SILU;
    const bool is_c4  = src->ne[0] % 4 == 0;
    const bool is_cnt = ggml_nelements(op) < 32768;

    char base[256];
    char name[256];
    snprintf(base, sizeof(base), "kernel_unary_f32_f32%s", is_c4 ? "_4" : "");
    snprintf(name, sizeof(name), "%s_op=%d_cnt=%d_pre=1", base, op_num, is_cnt);

    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, op_num, FC_UNARY + 0);
        ggml_metal_cv_set_bool (cv, is_cnt, FC_UNARY + 1);
        ggml_metal_cv_set_bool (cv, true,   FC_FUSION_FN + 0);
        pipeline = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        ggml_metal_cv_free(cv);
    }

    ggml_metal_kargs_unary args = {
        /*.ne00  =*/ (int32_t) src->ne[0],
        /*.ne01  =*/ (int32_t) src->ne[1],
        /*.ne02  =*/ (int32_t) src->ne[2],
        /*.ne03  =*/ (int32_t) src->ne[3],
        /*.nb00  =*/ src->nb[0],
        /*.nb01  =*/ src->nb[1],
        /*.nb02  =*/ src->nb[2],
        /*.nb03  =*/ src->nb[3],
        /*.ne0   =*/ (int32_t) op->ne[0],
        /*.ne1   =*/ (int32_t) op->ne[1],
        /*.ne2   =*/ (int32_t) op->ne[2],
        /*.ne3   =*/ (int32_t) op->ne[3],
        /*.nb0   =*/ op->nb[0],
        /*.nb1   =*/ op->nb[1],
        /*.nb2   =*/ op->nb[2],
        /*.nb3   =*/ op->nb[3],
        /*.slope =*/ 0.0f,
        /*.scale =*/ ggml_get_op_params_f32(scale, 0),
        /*.bias  =*/ ggml_get_op_params_f32(scale, 1),
        /*.val   =*/ 0.0f,
        /*.min   =*/ 0.0f,
        /*.max   =*/ 0.0f,
    };

    if (is_c4) {
        args.ne00 /= 4;
        args.ne0  /= 4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_fn_buffer_id(src), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_fn_buffer_id(op),  2);

    // ggml_metal_op_unary's dispatch
    if (is_cnt) {
        const int n = is_c4 ? ggml_nelements(op)/4 : ggml_nelements(op);
        ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);
    } else {
        const int nth_max = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        const int nth = std::min(args.ne00, nth_max);
        const int nk0 = (args.ne00 + nth - 1)/nth;
        ggml_metal_encoder_dispatch_threadgroups(enc, nk0*args.ne01, args.ne02, args.ne03, nth, 1, 1);
    }

    return 2;
}

int ggml_metal_op_fn_encode(
        ggml_metal_library_t        lib,
        ggml_metal_encoder_t        enc,
        const ggml_metal_fusion   * fusion,
        const ggml_tensor * const * nodes,
        int                         n) {
    int n_done = 0;

    switch (fusion->id) {
        case GGML_METAL_FUSION_FN_SCALE_UNARY: n_done = ggml_metal_fn_encode_scale_unary(lib, enc, nodes); break;
        default:
            GGML_ABORT("%s: not an FN fusion (id %d)", __func__, (int) fusion->id);
    }

    GGML_ASSERT(n_done == n);
    return n_done;
}

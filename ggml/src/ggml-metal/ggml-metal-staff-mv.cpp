// #23: the staff's Q8_0 mat-vec for a check of 2-16 words. See ggml-metal-staff-mv.h.

#include "ggml-metal-staff-mv.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-metal-impl.h"

#include <cstdio>
#include <cstring>

int ggml_metal_staff_mv_parse(const char * s, bool * bad) {
    if (bad) {
        *bad = false;
    }
    if (s == nullptr || *s == '\0' || strcmp(s, "1") == 0) {
        return 1;
    }
    if (strcmp(s, "0") == 0) {
        return 0;
    }
    if (bad) {
        *bad = true;
    }
    return 1;
}

int ggml_metal_staff_mv_limit(int staff_mv, bool has_tensor) {
    return staff_mv && has_tensor ? GGML_METAL_STAFF_MV_MAX : 0;
}

int ggml_metal_staff_mv_min_cols(int64_t ne00) {
    return ne00 % 128 == 0 ? GGML_METAL_STAFF_MV_MIN_EXT : 2;
}

bool ggml_metal_staff_mv_takes(const ggml_tensor * op, int limit) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const int64_t ne11 = src1->ne[1];
    return src0->type == GGML_TYPE_Q8_0                       &&
           src1->type == GGML_TYPE_F32                        &&
           src0->ne[0] % 32 == 0                              && // whole Q8_0 blocks
           src0->ne[1] >= GGML_METAL_STAFF_MV_ROWS            &&
           src1->nb[0] == sizeof(float)                       && // a column's values side by side
           ne11 >= ggml_metal_staff_mv_min_cols(src0->ne[0])  &&
           ne11 <= limit;
}

static ggml_metal_buffer_id ggml_metal_staff_mv_buffer_id(const ggml_tensor * t) {
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id((ggml_metal_buffer_t) buffer->context, t);
}

int ggml_metal_staff_mv_encode(ggml_metal_library_t lib, ggml_metal_encoder_t enc, const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 >= 2 && ne11 <= GGML_METAL_STAFF_MV_MAX);
    GGML_ASSERT(ne12 % src0->ne[2] == 0 && ne13 % src0->ne[3] == 0);

    // the columns padded to matmul2d's 8 or 16
    const int nm = ne11 <= 8 ? 8 : 16;

    char name[128];
    snprintf(name, sizeof(name), "kernel_mul_mv_staff_%s_%s_m%d", ggml_type_name(src0->type), ggml_type_name(src1->type), nm);

    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (!pipeline.pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    ggml_metal_kargs_mul_mv_ext args = {
        /*.ne00  =*/ (int32_t) ne00,
        /*.ne01  =*/ (int32_t) ne01,
        /*.ne02  =*/ (int32_t) src0->ne[2],
        /*.nb00  =*/ src0->nb[0],
        /*.nb01  =*/ src0->nb[1],
        /*.nb02  =*/ src0->nb[2],
        /*.nb03  =*/ src0->nb[3],
        /*.ne10  =*/ (int32_t) src1->ne[0],
        /*.ne11  =*/ (int32_t) ne11,
        /*.ne12  =*/ (int32_t) ne12,
        /*.nb10  =*/ src1->nb[0],
        /*.nb11  =*/ src1->nb[1],
        /*.nb12  =*/ src1->nb[2],
        /*.nb13  =*/ src1->nb[3],
        /*.ne0   =*/ (int32_t) op->ne[0],
        /*.ne1   =*/ (int32_t) op->ne[1],
        /*.r2    =*/ (int16_t) (ne12/src0->ne[2]),
        /*.r3    =*/ (int16_t) (ne13/src0->ne[3]),
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_staff_mv_buffer_id(src0), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_staff_mv_buffer_id(src1), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_staff_mv_buffer_id(op),   3);

    // each simdgroup's staged columns (half [nm][32]) while it runs, then the k-split's partials (float [nm][32])
    const int nsg = OP_MUL_MV_STAFF_NSG;
    ggml_metal_encoder_set_threadgroup_memory_size(enc, (size_t) nsg*nm*32*sizeof(float), 0);

    // a threadgroup a 32 rows
    ggml_metal_encoder_dispatch_threadgroups(enc, (int) ((ne01 + 31)/32), 1, (int) (ne12*ne13), 32, nsg, 1);

    return 1;
}

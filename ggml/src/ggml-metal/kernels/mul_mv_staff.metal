#include "common.h"
#include "dequantize.h"

// #23 (GGML_METAL_STAFF_MV): the staff's Q8_0 mat-vec for a check of 2-16 words, on the matrix units
// (matmul2d, Metal 4 tensor ops). See ggml-metal-staff-mv.h.
//
// A threadgroup owns 32 weight rows; its SK = OP_MUL_MV_STAFF_NSG simdgroups split the rows' Q8_0 blocks
// between them (simdgroup sg_k takes blocks sg_k, sg_k + SK, ...). Each step is one block (32 k):
//
// - the block's int8 codes go straight from device memory into matmul2d (a 32 x 32 tensor view at the block,
//   rows nb01 apart; the codes start 2 bytes into the 34-byte block, an alignment matmul2d accepts);
// - the block's 32 k of every column are staged in threadgroup memory as half (the columns padded to NM, 8 or
//   16; a lane converts 8 values of one column), the next block's values already on their way in registers;
// - matmul2d multiplies codes by columns into an F32 partial, and the block's scale d (a lane's row's, shuffled
//   to each element's row) is applied to the partial in F32 before it is added to the running sum.
//
// So each weight is read once for every column, and the matrix units do the multiplies. The sum is not
// mul_mv_ext's: half columns (mul_mm's 16-bit inputs) and another order, a rounding-level change.
//
// Half's range: a column value past 65504 would be inf as half. A lane notes it (one compare a step) and, if
// any lane of the simdgroup did, the simdgroup redoes its share in F32 straight from device memory, so such a
// column costs time, never correctness.
//
// The k-split's partials are summed in order of sg_k through threadgroup memory, each written and read with
// the cooperative tensor's own store/load (its layout, and how many elements a lane holds, depend on the
// operands' address spaces).

#ifdef GGML_METAL_HAS_TENSOR

template<short NM>
kernel void kernel_mul_mv_staff_q8_0_f32(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short SK  = OP_MUL_MV_STAFF_NSG;
    constexpr short NR  = 32;     // weight rows a threadgroup
    constexpr short NCL = NM/8;   // columns a lane stages: 8 columns a pass, 4 lanes a column

    const int nb = args.ne00/QK8_0;
    const int n  = args.ne11;

    const short sg_k = sgitg;

    // the rows: r0 .. r0 + nr - 1
    const int r0 = tgpig.x*NR;
    const int nr = min((int) NR, args.ne01 - r0);

    const int i1m = tgpig.z;
    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = (uint64_t) i12*args.nb12 + (uint64_t) i13*args.nb13;

    threadgroup half * ys = (threadgroup half *) shmem + sgitg*NM*QK8_0;

    device const char * w0 = src0 + offset0 + (uint64_t) r0*args.nb01;

    // d of row tiisg (the last row, past nr): block ib's d is 17 halves on from block ib - 1's
    device const half * dp = (device const half *) (w0 + (uint64_t) min((int) tiisg, nr - 1)*args.nb01);

    // a lane's columns yc + 8*j and its 8 k of each block, yk .. yk + 7; columns past n stage zeros
    const short yc = tiisg/4;
    const short yk = (tiisg%4)*8;
    device const float * yr[NCL];
    bool yv[NCL];
    FOR_UNROLL (short j = 0; j < NCL; ++j) {
        yv[j] = yc + 8*j < n;
        yr[j] = (device const float *) (src1 + offset1 + (uint64_t) (yv[j] ? yc + 8*j : 0)*args.nb11) + yk;
    }

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NM, NR, QK8_0, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<1>> mm;

    using tw_t = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;

    auto ty  = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(ys, dextents<int32_t, 2>(QK8_0, NM), array<int, 2>({1, QK8_0}));
    auto tw0 = tw_t((device int8_t *) (w0 + 2), dextents<int32_t, 2>(QK8_0, nr), array<int, 2>({1, (int) args.nb01}));

    auto acc  = mm.template get_destination_cooperative_tensor<decltype(ty), decltype(tw0), float>();
    auto part = mm.template get_destination_cooperative_tensor<decltype(ty), decltype(tw0), float>();
    FOR_UNROLL (short i = 0; i < acc.get_capacity(); ++i) {
        acc[i] = 0.0f;
    }

    packed_float4 yq[NCL][2];
    half          dq;

    auto fetch = [&](int ib) {
        dq = dp[ib*17];
        FOR_UNROLL (short j = 0; j < NCL; ++j) {
            FOR_UNROLL (short q = 0; q < 2; ++q) {
                yq[j][q] = yv[j] ? ((device const packed_float4 *) (yr[j] + ib*QK8_0))[q] : packed_float4(0.0f);
            }
        }
    };

    bool ovf = false; // a column value of this lane's past half's range
    int ib = sg_k;
    if (ib < nb) {
        fetch(ib);
    }
    for (; ib < nb; ib += SK) {
        const float dl = (float) dq;
        float amax = 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short j = 0; j < NCL; ++j) {
            const float4 y0 = float4(yq[j][0]);
            const float4 y1 = float4(yq[j][1]);
            const float4 a4 = max(abs(y0), abs(y1));
            amax = max(amax, max(max(a4.x, a4.y), max(a4.z, a4.w)));
            threadgroup half4 * yrow = (threadgroup half4 *) (ys + (yc + 8*j)*QK8_0 + yk);
            yrow[0] = half4(y0);
            yrow[1] = half4(y1);
        }
        ovf = ovf || amax > 65504.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (ib + SK < nb) {
            fetch(ib + SK);
        }
        auto tw = tw_t((device int8_t *) (w0 + (uint64_t) ib*34 + 2), dextents<int32_t, 2>(QK8_0, nr), array<int, 2>({1, (int) args.nb01}));
        mm.run(ty, tw, part);
        FOR_UNROLL (short i = 0; i < acc.get_capacity(); ++i) {
            const auto idx = part.get_multidimensional_index(i); // [0]: the row, [1]: the column
            acc[i] += part[i]*simd_shuffle(dl, (ushort) (idx[0] & (NR - 1)));
        }
    }

    if (simd_any(ovf)) {
        // rare: a column past half's range; this simdgroup's share is redone in F32 from device memory
        FOR_UNROLL (short i = 0; i < acc.get_capacity(); ++i) {
            const auto idx = acc.get_multidimensional_index(i);
            const int row = min((int) idx[0], nr - 1);
            const int col = min((int) idx[1], n - 1);
            device const block_q8_0 * wr = (device const block_q8_0 *) (w0 + (uint64_t) row*args.nb01);
            device const float * yf = (device const float *) (src1 + offset1 + (uint64_t) col*args.nb11);
            float sum = 0.0f;
            for (int jb = sg_k; jb < nb; jb += SK) {
                float s = 0.0f;
                for (short k = 0; k < QK8_0; ++k) {
                    s += (float) wr[jb].qs[k]*yf[jb*QK8_0 + k];
                }
                sum += s*(float) wr[jb].d;
            }
            acc[i] = sum;
        }
    }

    if (SK > 1) {
        // the k-split's partials [column][row], summed in order of sg_k
        threadgroup float * red = (threadgroup float *) shmem;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_k > 0) {
            acc.store(tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(red + sg_k*NR*NM, dextents<int32_t, 2>(NR, NM), array<int, 2>({1, NR})));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_k > 0) {
            return;
        }
        for (short s = 1; s < SK; ++s) {
            part.load(tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(red + s*NR*NM, dextents<int32_t, 2>(NR, NM), array<int, 2>({1, NR})));
            FOR_UNROLL (short i = 0; i < acc.get_capacity(); ++i) {
                acc[i] += part[i];
            }
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t) i1m*args.ne0*args.ne1 + r0;
    acc.store(tensor<device float, dextents<int32_t, 2>, tensor_inline>(dst_f32, dextents<int32_t, 2>(nr, n), array<int, 2>({1, args.ne0})));
}

typedef decltype(kernel_mul_mv_staff_q8_0_f32<8>) mul_mv_staff_t;

template [[host_name("kernel_mul_mv_staff_q8_0_f32_m8")]]  kernel mul_mv_staff_t kernel_mul_mv_staff_q8_0_f32<8>;
template [[host_name("kernel_mul_mv_staff_q8_0_f32_m16")]] kernel mul_mv_staff_t kernel_mul_mv_staff_q8_0_f32<16>;

#endif

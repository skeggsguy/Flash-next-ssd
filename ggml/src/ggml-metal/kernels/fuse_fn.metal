#include "common.h"

// Fix 2 (GGML_METAL_FUSION_FN): merged chains that need a kernel of their own.
//
// Every step keeps its own kernel's expression, vector width and reduction order, so the merged job
// writes the bytes the chain's jobs would. Fast math, as for every Metal kernel here, is no obstacle
// as long as no step's arithmetic is regrouped with its neighbour's.

// P3: the router's weights (build_moe_ffn with norm_w). One token per threadgroup:
//   w = GET_ROWS(probs, ids)                    each book's probability
//   c = CLAMP(SUM_ROWS(w), min, max)            the sum, kept off zero
//   dst = DIV(w, c)
// The sum is kernel_sum_rows_impl's, line for line: the host dispatches the thread count SUM_ROWS
// would (T0 = float4 when n_used is a multiple of 4, as SUM_ROWS picks), then the same two simd_sum
// stages through threadgroup memory and sum() of the result. CLAMP is kernel_unary's clamp(), DIV
// kernel_bin's a / b on each weight.

static inline float fn_router_w(constant ggml_metal_kargs_fn_router_w & args,
        device const char * probs, device const char * ids, int it, int i) {
    const int32_t book = *(device const int32_t *) (ids + i*args.nb_i0 + it*args.nb_i1);
    return *(device const float *) (probs + book*args.nb_p1 + it*args.nb_p2);
}

// SUM_ROWS' element i0 of the row: one weight, or four in its float4 kernel
static inline void fn_router_w_load(thread float & v, constant ggml_metal_kargs_fn_router_w & args,
        device const char * probs, device const char * ids, int it, int i0) {
    v = fn_router_w(args, probs, ids, it, i0);
}

static inline void fn_router_w_load(thread float4 & v, constant ggml_metal_kargs_fn_router_w & args,
        device const char * probs, device const char * ids, int it, int i0) {
    v = float4(fn_router_w(args, probs, ids, it, 4*i0 + 0), fn_router_w(args, probs, ids, it, 4*i0 + 1),
               fn_router_w(args, probs, ids, it, 4*i0 + 2), fn_router_w(args, probs, ids, it, 4*i0 + 3));
}

template <typename T0>
kernel void kernel_fn_router_w_impl(
        constant ggml_metal_kargs_fn_router_w & args,
        device const char * probs,
        device const char * ids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.x;

    // SUM_ROWS
    threadgroup T0 * shmem_t = (threadgroup T0 *) shmem;

    if (sgitg == 0) {
        shmem_t[tiisg] = 0.0f;
    }

    T0 sumf = T0(0.0f);

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        T0 v;
        fn_router_w_load(v, args, probs, ids, it, i0);
        sumf += v;
    }

    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_t[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_t[tiisg];
    sumf = simd_sum(sumf);

    // CLAMP of the float SUM_ROWS stores
    const float c = clamp(sum(sumf), args.min, args.max);

    // DIV
    for (int i = tpitg.x; i < args.n_used; i += ntg.x) {
        *(device float *) (dst + i*args.nb_d0 + it*args.nb_d1) = fn_router_w(args, probs, ids, it, i) / c;
    }
}

typedef decltype(kernel_fn_router_w_impl<float>) kernel_fn_router_w_t;

template [[host_name("kernel_fn_router_w_f32")]]   kernel kernel_fn_router_w_t kernel_fn_router_w_impl<float>;
template [[host_name("kernel_fn_router_w_f32_4")]] kernel kernel_fn_router_w_t kernel_fn_router_w_impl<float4>;

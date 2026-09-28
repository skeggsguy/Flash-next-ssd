// WRITING-PLAN.md step 3b: Metal's small-batch mat-vec (mul_mv_ext) up to 16 columns (GGML_METAL_MV_EXT_MAX).
//
// A depth-8 apprentice has the library check 9 words at once; the small-batch kernel stopped at 8 columns, so
// that batch fell to the large-batch kernel (see ggml-metal-mv-ext.h). Two parts:
//
// - the switch and the table, on the CPU: the value's parsing ("8" and "0" are today's graph), the limit
//   (the switch's value, at most 10 on matrices of 4,096 rows or more, where 11+ columns lose to mul_mm),
//   and the columns per threadgroup for 9-16 (always an instantiated 2..5, pinned);
// - the route, on MTL0: the kernel's sums depend only on the threads along a row, not on how many columns
//   a threadgroup walks, so a 9-16-column product through mul_mv_ext writes, column for column, the bytes
//   of the same columns run as two batches of 4-8 (which mul_mv_ext takes today). With the limit at 16
//   that must hold byte for byte; at "8" the 9-16 columns go to mul_mm, whose sums differ, and it must not
//   (the switch really restores today's route). The same on 4,099 rows, where 11+ columns must stay on
//   mul_mm. Rows not a multiple of a threadgroup's and every column count 9-16 exercise the kernel's tails.
//
// ctest runs it twice: GGML_METAL_MV_EXT_MAX unset (16) and =8.

#include "testing.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Metal-internal (ggml-metal-mv-ext.h), linked from ggml-metal
extern "C" int ggml_metal_mv_ext_max_parse(const char * s, bool * bad);
extern "C" int ggml_metal_mv_ext_limit(int64_t ne01, int mv_ext_max);
extern "C" int ggml_metal_mv_ext_r1ptg_wide(int ne11);

namespace {

// rows: neither a multiple of a threadgroup's 8; the second past the 4,096 where the limit drops to 10
constexpr int64_t M_SMALL = 37;
constexpr int64_t M_BIG   = 4099;

std::vector<float> seeded(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(n);
    for (float & v : f) {
        v = u(rng);
    }
    return f;
}

// src0 (k x m of type t) from a seeded fill, quantized or converted on the CPU
void set_src0(ggml_tensor * a, uint32_t seed) {
    const int64_t k = a->ne[0];
    const int64_t M = a->ne[1];
    const std::vector<float> f = seeded((size_t) (k*M), seed);
    if (a->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(a, f.data(), 0, ggml_nbytes(a));
        return;
    }
    std::vector<uint8_t> q(ggml_nbytes(a));
    ggml_quantize_chunk(a->type, f.data(), q.data(), 0, M, k, nullptr);
    ggml_backend_tensor_set(a, q.data(), 0, q.size());
}

// out = src0 x columns [c0, c0 + n) of the seeded k x n_all src1, as bytes
std::vector<uint8_t> mul_mat_bytes(ggml_backend_t backend, ggml_type t, int64_t M, int64_t k, int64_t n_all, int64_t c0, int64_t n) {
    ggml_init_params p = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, t, k, M);
    ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    set_src0(a, 1234u + (uint32_t) k + 1000u*(uint32_t) t);
    const std::vector<float> all = seeded((size_t) (k*n_all), 99u + (uint32_t) (k + n_all));
    ggml_backend_tensor_set(b, all.data() + c0*k, 0, ggml_nbytes(b));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<uint8_t> bytes(ggml_nbytes(out));
    ggml_backend_tensor_get(out, bytes.data(), 0, bytes.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return bytes;
}

bool any_nonzero(const std::vector<uint8_t> & b) {
    for (uint8_t c : b) {
        if (c != 0) {
            return true;
        }
    }
    return false;
}

void test_switch(testing & t) {
    bool bad = true;
    t.assert_equal("unset", 16, ggml_metal_mv_ext_max_parse(nullptr, &bad));
    t.assert_true("unset is not bad", !bad);
    t.assert_equal("empty", 16, ggml_metal_mv_ext_max_parse("", &bad));
    t.assert_equal("\"8\" is today", 8, ggml_metal_mv_ext_max_parse("8", &bad));
    t.assert_equal("\"0\" is off (today)", 8, ggml_metal_mv_ext_max_parse("0", &bad));
    t.assert_true("\"0\" is not bad", !bad);
    t.assert_equal("\"16\"", 16, ggml_metal_mv_ext_max_parse("16", &bad));
    t.assert_equal("\"12\"", 12, ggml_metal_mv_ext_max_parse("12", &bad));
    t.assert_equal("nothing narrower than today", 8, ggml_metal_mv_ext_max_parse("3", &bad));
    t.assert_equal("nothing wider than 16", 16, ggml_metal_mv_ext_max_parse("99", &bad));
    t.assert_equal("negative is off", 8, ggml_metal_mv_ext_max_parse("-1", &bad));
    t.assert_equal("a word falls back to 16", 16, ggml_metal_mv_ext_max_parse("on", &bad));
    t.assert_true("a word is bad", bad);
    t.assert_equal("trailing junk falls back to 16", 16, ggml_metal_mv_ext_max_parse("12x", &bad));
    t.assert_true("trailing junk is bad", bad);
}

void test_table(testing & t) {
    for (int mx = 8; mx <= 16; mx++) {
        const std::string at = " at " + std::to_string(mx);
        // up to 4,095 rows the switch's value
        for (int64_t rows : { 1, 37, 512, 2560, 4095 }) {
            t.assert_equal(std::to_string(rows) + " rows" + at, mx, ggml_metal_mv_ext_limit(rows, mx));
        }
        // from 4,096 rows at most 10 (the staff's 6,144-248,320-row matrices), and never under today's 8
        for (int64_t rows : { 4096, 6144, 248320 }) {
            t.assert_equal(std::to_string(rows) + " rows" + at, std::min(mx, 10), ggml_metal_mv_ext_limit(rows, mx));
        }
    }
    // columns per threadgroup: pinned, instantiated (2..5), and 0 outside 9-16
    const int pin[17] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, /*9*/ 5, 5, 4, 4, 5, 5, 5, 4 };
    for (int n = -1; n <= 20; n++) {
        const int r = ggml_metal_mv_ext_r1ptg_wide(n);
        if (n < 9 || n > 16) {
            t.assert_equal("r1ptg outside 9-16 at " + std::to_string(n), 0, r);
            continue;
        }
        t.assert_true("r1ptg instantiated at " + std::to_string(n), r >= 2 && r <= 5);
        t.assert_equal("r1ptg pinned at " + std::to_string(n), pin[n], r);
        // the fewest rounds of threadgroups any instantiated width needs (upstream's rule for 2-8)
        int fewest = n;
        for (int c = 2; c <= 5; c++) {
            fewest = std::min(fewest, (n + c - 1)/c);
        }
        t.assert_equal("r1ptg the fewest rounds at " + std::to_string(n), fewest, (n + r - 1)/r);
    }
}

// the route's expectation, written out here rather than asked of ggml_metal_mv_ext_limit, so a change to the
// helper that the device follows too still shows as red: ctest sets GGML_METAL_MV_EXT_MAX to 16 or 8
int expected_limit(int64_t rows, int mv_ext_max) {
    return rows >= 4096 ? std::min(mv_ext_max, 10) : mv_ext_max;
}

void test_route(testing & t, ggml_backend_t backend, int mv_ext_max) {
    struct shape { int64_t m, k; };
    const shape shapes[] = { { M_SMALL, 768 }, { M_SMALL, 2560 }, { M_BIG, 768 } }; // k a whole number of K-quant blocks
    const ggml_type types[] = { GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0,
                                GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    for (ggml_type ty : types) {
        for (const shape & sh : shapes) {
            const int64_t k = sh.k;
            const int lim = expected_limit(sh.m, mv_ext_max);
            for (int64_t n = 9; n <= 16; n++) {
                const std::string name = std::string(ggml_type_name(ty)) + " m=" + std::to_string(sh.m) + " k=" + std::to_string(k) + " n=" + std::to_string(n);
                const int64_t n1 = (n + 1)/2; // two batches of 5-8 (mul_mv_ext today, K-quants included)
                const std::vector<uint8_t> whole = mul_mat_bytes(backend, ty, sh.m, k, n, 0, n);
                std::vector<uint8_t> split = mul_mat_bytes(backend, ty, sh.m, k, n, 0, n1);
                const std::vector<uint8_t> rest = mul_mat_bytes(backend, ty, sh.m, k, n, n1, n - n1);
                split.insert(split.end(), rest.begin(), rest.end());
                t.assert_true(name + ": nonzero", any_nonzero(whole));
                if (n <= lim) {
                    t.assert_true(name + ": mul_mv_ext, the bytes of two small batches", whole == split);
                } else {
                    t.assert_true(name + ": mul_mm (past the limit " + std::to_string(lim) + "), other sums", whole != split);
                }
            }
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    testing t(std::cout);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    const char * env = getenv("GGML_METAL_MV_EXT_MAX");
    // unset and "16" are 16, "8" is 8 (ctest's two settings), read here, not by the parser under test
    const int mv_ext_max = env && strcmp(env, "8") == 0 ? 8 : env == nullptr || strcmp(env, "16") == 0 ? 16 :
                           ggml_metal_mv_ext_max_parse(env, nullptr);
    printf("GGML_METAL_MV_EXT_MAX=%s: up to %d columns\n", env ? env : "(unset)", mv_ext_max);

    t.test("the switch", test_switch);
    t.test("the table", test_table);

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (dev == nullptr) {
        printf("no MTL0 device: the route skipped\n");
        return t.summary();
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    t.test("the route", [&](testing & t) { test_route(t, backend, mv_ext_max); });
    ggml_backend_free(backend);
    return t.summary();
}

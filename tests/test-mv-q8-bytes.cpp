// Fix 2a: the Q8_0 mat-vec's trimmed simdgroups must write the bytes the untrimmed kernel wrote.
//
// ggml_metal_mv_q8_0_nsg (ggml-metal-device.cpp) runs only the simdgroups a short row can use (hc up has
// k = 320, 10 blocks: 2 simdgroups, not 4). The trim is exact by construction: every thread keeps the
// same blocks, and the simdgroups dropped only ever added +0.0 to the same 32-lane reduction. But it is
// not behind a switch, and test-backend-ops compares with a tolerance, so a change to the trim that only
// regrouped the sum (nsg = 1 at k = 320 gives each thread two blocks) would pass every tolerance gate
// unseen, and the study's byte gates run the 64-book test models, which are F32.
//
// This test prints one hash per case: Q8_0 MUL_MAT and MUL_MAT_ID at short and long k, n = 1/3/8, on
// deterministic inputs (quantized on the CPU from a seeded fill). The gate is the same binary run against
// two builds' libraries, and the two listings diffed:
//
//   build/bin/test-mv-q8-bytes --hashes                                            > new.txt
//   DYLD_LIBRARY_PATH=$PWD/build/bin.67eb49ef3 build/bin/test-mv-q8-bytes --hashes > old.txt
//   diff old.txt new.txt
//
// (DYLD_LIBRARY_PATH wins over the executable's rpath, so the second run computes with the old build's
// ggml and Metal libraries; DYLD_PRINT_LIBRARIES=1 shows which loaded.) Under ctest, with no old build to
// hand, it holds the outputs to being nonzero and the same bytes on every repeat.

#include "testing.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N_RUNS = 3;

// the rows of the Q8_0 matrix: the kernel's block walk does not depend on it beyond the dispatch
constexpr int64_t M = 64;
constexpr int64_t N_EXPERT = 16;
constexpr int64_t N_USED = 4;

struct mv_case {
    bool    id; // MUL_MAT_ID rather than MUL_MAT
    int64_t k;
    int64_t n;  // tokens

    std::string name() const {
        return std::string(id ? "mul_mat_id" : "mul_mat") + " q8_0 k=" + std::to_string(k) + " n=" + std::to_string(n);
    }
};

std::vector<mv_case> all_cases() {
    std::vector<mv_case> cases;
    // 1, 3, 10 (hc up), 20, 25, 32 and 80 blocks: nsg 1, 1, 2, 3, 4, 4, 4
    for (int64_t k : { 32, 96, 320, 640, 800, 1024, 2560 }) {
        for (int64_t n : { 1, 3, 8 }) {
            cases.push_back({ false, k, n });
            cases.push_back({ true,  k, n });
        }
    }
    return cases;
}

uint64_t fnv1a(const std::vector<uint8_t> & b) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t c : b) {
        h = (h ^ c) * 1099511628211ull;
    }
    return h;
}

// quantized on the CPU from a seeded fill, so both builds see the same bytes
void fill_q8_0(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t k     = t->ne[0];
    const int64_t nrows = ggml_nrows(t);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(k*nrows);
    for (float & v : f) {
        v = u(rng);
    }
    std::vector<uint8_t> q(ggml_nbytes(t));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, nrows, k, nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

void fill_f32(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(ggml_nelements(t));
    for (float & v : f) {
        v = u(rng);
    }
    ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
}

// distinct books a token
void fill_ids(ggml_tensor * t, std::mt19937 & rng) {
    std::vector<int32_t> ids(ggml_nelements(t));
    std::vector<int32_t> books(N_EXPERT);
    for (int64_t r = 0; r < t->ne[1]; r++) {
        for (int64_t e = 0; e < N_EXPERT; e++) {
            books[e] = (int32_t) e;
        }
        std::shuffle(books.begin(), books.end(), rng);
        for (int64_t i = 0; i < t->ne[0]; i++) {
            ids[r*t->ne[0] + i] = books[i];
        }
    }
    ggml_backend_tensor_set(t, ids.data(), 0, ggml_nbytes(t));
}

std::vector<uint8_t> run(ggml_backend_t backend, const mv_case & c) {
    ggml_init_params p = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a;
    ggml_tensor * b;
    ggml_tensor * ids = nullptr;
    ggml_tensor * out;
    if (c.id) {
        a   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, c.k, M, N_EXPERT);
        b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  c.k, N_USED, c.n);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,  N_USED, c.n);
        out = ggml_mul_mat_id(ctx, a, b, ids);
    } else {
        a   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, c.k, M);
        b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  c.k, c.n);
        out = ggml_mul_mat(ctx, a, b);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    std::mt19937 rng(1234u + (uint32_t) (2*c.k + c.n + (c.id ? 100000 : 0)));
    fill_q8_0(a, rng);
    fill_f32(b, rng);
    if (ids) {
        fill_ids(ids, rng);
    }

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

} // namespace

int main(int argc, char ** argv) {
    const bool hashes = argc > 1 && strcmp(argv[1], "--hashes") == 0;

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (dev == nullptr) {
        printf("no MTL0 device: skipped\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    if (hashes) {
        for (const mv_case & c : all_cases()) {
            printf("%-32s %016llx\n", c.name().c_str(), (unsigned long long) fnv1a(run(backend, c)));
        }
        ggml_backend_free(backend);
        return 0;
    }

    testing t;
    t.test("q8_0 mat-vec: nonzero, the same bytes every run", [&](testing & t) {
        for (const mv_case & c : all_cases()) {
            const std::vector<uint8_t> first = run(backend, c);
            t.assert_true(c.name() + ": nonzero", any_nonzero(first));
            for (int r = 1; r < N_RUNS; r++) {
                const std::vector<uint8_t> again = run(backend, c);
                t.assert_true(c.name() + ": run " + std::to_string(r) + " same bytes", again == first);
            }
        }
    });

    ggml_backend_free(backend);
    return t.summary();
}

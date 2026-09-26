// Fix 2 (GGML_METAL_FUSION_FN): merged op chains on Metal must write the same bytes as the unmerged ops.
//
// A written token of Qwen3.8-Flash-Next runs ~3,500 GPU jobs; fix 2 merges a few chains of tiny ones
// into one job each (ggml-metal-fusion-fn.{h,cpp}, ggml-metal-ops-fn.cpp). The words written must not
// change, so every merge is exact by construction, and this test holds it to that: each chain is built
// at the model's shapes, run on MTL0 through the scheduler (the path llama.cpp takes, with the graph
// optimizer's packing and reorder), once with fusion on and once with it off, and the outputs are
// compared with memcmp. The fusion counters must show the merge fired (with the switch on), and the
// negative cases - a chain that must not merge - must leave it unfired.
//
// ctest runs it twice: GGML_METAL_FUSION_FN=1 (everything above) and =0 (no FN pattern may exist in the
// fusion table, and the outputs still agree).

#include "testing.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

// Metal-internal (ggml-metal-fusion-fn.h), linked from ggml-metal for the split test
extern "C" int ggml_metal_fusion_fn_split(const struct ggml_cgraph * gf, int n_split);
extern "C" int ggml_metal_fusion_max(const struct ggml_cgraph * gf, int idx);

namespace {

// the fusion debugging API the Metal backend exports through get_proc_address (see test-fusion.cpp)
typedef void * fusion_t;
struct fusion_api {
    fusion_t h = nullptr;
    void (*stats_init)(fusion_t)                                  = nullptr;
    void (*stats_reset)(fusion_t)                                 = nullptr;
    int  (*stats_get)(fusion_t, const char **, uint64_t *, int)   = nullptr;
    void (*set_enabled)(fusion_t, bool)                           = nullptr;

    std::map<std::string, uint64_t> counts() const {
        const int n = stats_get(h, nullptr, nullptr, 0);
        std::vector<const char *> labels(n);
        std::vector<uint64_t>     values(n);
        stats_get(h, labels.data(), values.data(), n);
        std::map<std::string, uint64_t> res;
        for (int i = 0; i < n; i++) {
            res[labels[i]] += values[i];
        }
        return res;
    }
};

struct env {
    ggml_backend_t       mtl   = nullptr;
    ggml_backend_t       cpu   = nullptr;
    ggml_backend_sched_t sched = nullptr;
    fusion_api           api;
    bool                 fn    = false; // GGML_METAL_FUSION_FN as this process read it
};

// inputs in their own Metal buffer, the chain's ops in a graph the scheduler allocates
struct graph_case {
    std::string name;
    std::function<void(ggml_context * ctx_in, ggml_context * ctx_g, ggml_cgraph * gf,
                       std::vector<ggml_tensor *> & outs)> build;
    std::function<void(ggml_tensor * in, std::mt19937 & rng)> fill;
};

struct run_result {
    std::vector<uint8_t>            bytes;   // every output, in order
    std::map<std::string, uint64_t> counts;  // fusion counters of this run
    bool                            nonzero = false;
};

run_result run(env & e, const graph_case & gc, bool fusion, uint32_t seed) {
    ggml_init_params p_in = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_init_params p_g  = { ggml_tensor_overhead()*256 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx_in = ggml_init(p_in);
    ggml_context * ctx_g  = ggml_init(p_g);
    ggml_cgraph  * gf     = ggml_new_graph(ctx_g);

    std::vector<ggml_tensor *> outs;
    gc.build(ctx_in, ctx_g, gf, outs);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_in, e.mtl);
    std::mt19937 rng(seed);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_in); t; t = ggml_get_next_tensor(ctx_in, t)) {
        gc.fill(t, rng);
    }

    e.api.set_enabled(e.api.h, fusion);
    e.api.stats_reset(e.api.h);
    ggml_backend_sched_reset(e.sched);
    GGML_ASSERT(ggml_backend_sched_graph_compute(e.sched, gf) == GGML_STATUS_SUCCESS);
    e.api.set_enabled(e.api.h, true);

    run_result r;
    r.counts = e.api.counts();
    for (ggml_tensor * t : outs) {
        std::vector<uint8_t> b(ggml_nbytes(t));
        ggml_backend_tensor_get(t, b.data(), 0, b.size());
        const float * f = (const float *) b.data();
        for (size_t i = 0; i < b.size()/sizeof(float); i++) {
            r.nonzero = r.nonzero || f[i] != 0.0f;
        }
        r.bytes.insert(r.bytes.end(), b.begin(), b.end());
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx_g);
    ggml_free(ctx_in);
    return r;
}

// fused vs unfused, byte for byte; the named counter fired `expect` times (0: must not merge)
void check_case(testing & t, env & e, const graph_case & gc, const std::string & label, uint64_t expect) {
    for (uint32_t seed : { 1u, 2u, 3u }) {
        // merged first: an output the merged kernel failed to write then holds the previous seed's
        // values, not the ones the unmerged run is about to write
        const run_result on  = run(e, gc, true,  seed);
        const run_result off = run(e, gc, false, seed);

        t.assert_true(gc.name + ": outputs nonzero", on.nonzero);
        t.assert_true(gc.name + ": same bytes merged and unmerged",
                      on.bytes.size() == off.bytes.size() && memcmp(on.bytes.data(), off.bytes.data(), on.bytes.size()) == 0);

        const auto it = on.counts.find(label);
        const uint64_t fired = it == on.counts.end() ? 0 : it->second;
        t.assert_equal(gc.name + ": " + label + " fired", e.fn ? expect : 0, fired);
        t.assert_true(gc.name + ": unmerged run fired nothing", off.counts.count(label) == 0 || off.counts.at(label) == 0);
    }
}

// random values with the edge cases mixed in: signed zeros, +-80 (exp(80) is still finite), 20.0,
// subnormals
void fill_f32(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    static const float edges[] = { 0.0f, -0.0f, 80.0f, -80.0f, 20.0f, 1e-40f, -1e-40f, 1.0f };
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(ggml_nelements(t));
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = i % 7 == 3 ? edges[(i/7) % 8] : u(rng);
    }
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

// a second reader of an intermediate, built after the chain: the chain must then not merge, since the
// merged kernel never writes its intermediates (an output flag alone is no test here: the allocator
// may still let the next op overwrite a flagged tensor in place)
ggml_tensor * also_read(ggml_context * ctx_g, ggml_cgraph * gf, ggml_tensor * t) {
    ggml_tensor * r = ggml_scale(ctx_g, t, 3.0f);
    ggml_set_output(r);
    ggml_build_forward_expand(gf, r);
    return r;
}

// ---- P8: the command-buffer split ------------------------------------------

// rms_norm + mul + add groups back to back: a split inside a group moves to its start
void test_split(testing & t) {
    ggml_init_params p = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 2);
    for (int i = 0; i < 4; i++) {
        x = ggml_add(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w), w);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, x);

    // the optimizer packs the groups back to back; the encoder merges them one pattern at a time
    t.assert_equal("four groups of three", 12, ggml_graph_n_nodes(gf));
    t.assert_equal("the last group spans three nodes", 3, ggml_metal_fusion_max(gf, 9));
    for (int s = 1; s < 12; s++) {
        t.assert_equal("split " + std::to_string(s), s - s % 3, ggml_metal_fusion_fn_split(gf, s));
    }
    t.assert_equal("split 0 stays", 0, ggml_metal_fusion_fn_split(gf, 0));
    t.assert_equal("split at the end stays", 12, ggml_metal_fusion_fn_split(gf, 12));
    ggml_free(ctx);
}

// ---- P4: hc mix's SCALE(1/4) + SILU ----------------------------------------

const char * LABEL_P4 = "SCALE+UNARY";

// lo [ne0, n] (the down GEMM's output) -> scale(1/4) -> silu; neg 1: the SCALE is also read by another
// op, neg 2: a SIGMOID where the model has SILU. The model's 1/4 is a power of two, which makes the product
// exact; a scale of 1/3 with a bias is what shows the SCALE step rounds exactly where its kernel does.
graph_case case_scale_silu(int64_t ne0, int64_t nt, int neg, float scale = 0.25f, float bias = 0.0f) {
    graph_case gc;
    gc.name = "scale_silu ne0=" + std::to_string(ne0) + " n=" + std::to_string(nt) + " neg=" + std::to_string(neg) +
              " scale=" + std::to_string(scale) + " bias=" + std::to_string(bias);
    gc.build = [=](ggml_context * ctx_in, ggml_context * ctx_g, ggml_cgraph * gf, std::vector<ggml_tensor *> & outs) {
        ggml_tensor * lo  = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, ne0, nt);
        ggml_tensor * s   = ggml_scale_bias(ctx_g, lo, scale, bias);
        ggml_tensor * out = neg == 2 ? ggml_sigmoid(ctx_g, s) : ggml_silu(ctx_g, s);
        ggml_set_output(out);
        outs.push_back(out);
        ggml_build_forward_expand(gf, out);
        if (neg == 1) {
            outs.push_back(also_read(ctx_g, gf, s));
        }
    };
    gc.fill = [](ggml_tensor * t, std::mt19937 & rng) { fill_f32(t, rng, -30.0f, 30.0f); };
    return gc;
}

void test_scale_silu(testing & t, env & e) {
    for (int64_t nt : { 1, 4, 6, 512 }) {
        check_case(t, e, case_scale_silu(320, nt, 0), LABEL_P4, 1);
    }
    check_case(t, e, case_scale_silu(322, 3, 0), LABEL_P4, 1); // not a multiple of 4: the scalar kernels
    check_case(t, e, case_scale_silu(320, 6, 0, 1.0f/3.0f, 0.5f), LABEL_P4, 1);
    check_case(t, e, case_scale_silu(322, 3, 0, 1.0f/3.0f, 0.5f), LABEL_P4, 1);
    check_case(t, e, case_scale_silu(320, 6, 1), LABEL_P4, 0);
    check_case(t, e, case_scale_silu(320, 6, 2), LABEL_P4, 0);
}

} // namespace

int main(int argc, char ** argv) {
    testing t;
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    const char * fn = getenv("GGML_METAL_FUSION_FN");
    env e;
    e.fn = fn != nullptr && strcmp(fn, "1") == 0;

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (dev == nullptr) {
        printf("no MTL0 device: skipped\n");
        return 0;
    }
    e.mtl = ggml_backend_dev_init(dev, nullptr);
    e.cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[2] = { e.mtl, e.cpu };
    e.sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto proc = [&](const char * name) { return ggml_backend_reg_get_proc_address(reg, name); };
    e.api.h           = ((fusion_t (*)(ggml_backend_dev_t)) proc("ggml_backend_fusion_get"))(dev);
    e.api.stats_init  = (void (*)(fusion_t)) proc("ggml_backend_fusion_stats_init");
    e.api.stats_reset = (void (*)(fusion_t)) proc("ggml_backend_fusion_stats_reset");
    e.api.stats_get   = (int  (*)(fusion_t, const char **, uint64_t *, int)) proc("ggml_backend_fusion_stats_get");
    e.api.set_enabled = (void (*)(fusion_t, bool)) proc("ggml_backend_fusion_set_enabled");
    e.api.stats_init(e.api.h);

    printf("GGML_METAL_FUSION_FN=%s\n", e.fn ? "1" : "0");

    t.test("split", test_split);
    t.test("P4 scale+silu", [&](testing & t) { test_scale_silu(t, e); });

    ggml_backend_sched_free(e.sched);
    ggml_backend_free(e.cpu);
    ggml_backend_free(e.mtl);
    return t.summary();
}

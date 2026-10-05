// GGML_METAL_ENCODE_AHEAD (study patch, C2: cheaper floor stops), on MTL0 with no model. The scheduler hands
// the Metal backend the graph after the next CPU split while the GPU still runs this one; the backend encodes it
// then and the graph's own compute only commits it. That is exact only if:
//   - encoding captures no tensor value: inputs written after the graph was encoded ahead are the ones it reads;
//   - the encoding is used only by the graph it was made for: another graph computed first drops it (and gives
//     its own result), and so does the same graph object once rebuilt (a new uid);
//   - the switch really is off at "0".
// Each case checks the result bytes against the same graph computed with nothing encoded ahead, and the
// backend's counts of graphs encoded ahead and committed, so the path is known to have run (or not).
//
// ctest runs it twice: GGML_METAL_ENCODE_AHEAD unset (on) and =0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "../ggml/src/ggml-impl.h" // ggml_cgraph::uid, which the scheduler sets per split

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    printf("  %-72s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

typedef void (*encode_ahead_t)(ggml_backend_t, ggml_cgraph *);
typedef void (*ahead_counts_t)(int64_t *, int64_t *);

ahead_counts_t g_counts = nullptr;

struct counts {
    int64_t encoded = 0, used = 0;
};

counts now() {
    counts c;
    g_counts(&c.encoded, &c.used);
    return c;
}

std::vector<float> seeded(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(n);
    for (float & v : f) {
        v = u(rng);
    }
    return f;
}

std::vector<float> get(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

} // namespace

int main() {
    const char * env = getenv("GGML_METAL_ENCODE_AHEAD");
    const bool on = env == nullptr || strcmp(env, "0") != 0;
    printf("GGML_METAL_ENCODE_AHEAD=%s: %s\n", env ? env : "(unset)", on ? "on" : "off");

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (dev == nullptr) {
        printf("no MTL0 device: skipped\n");
        return 0;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto encode_ahead = (encode_ahead_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_encode_ahead");
    g_counts = (ahead_counts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_encode_ahead_counts");
    check(encode_ahead != nullptr && g_counts != nullptr, "the backend exports encode ahead and its counts");
    if (encode_ahead == nullptr || g_counts == nullptr) {
        printf("FAILED\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    // two graphs over the same inputs, each a few hundred nodes long so the backend splits it across command
    // buffers as it does a model's: x <- x*a + b, 128 times, for "mul" and the same with sub for "sub"
    const int64_t n = 4096;
    const int     steps = 128;
    ggml_init_params ip = { ggml_tensor_overhead()*(4*steps + 16) + ggml_graph_overhead()*2, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * x1 = a;
    ggml_tensor * x2 = a;
    for (int i = 0; i < steps; i++) {
        x1 = ggml_add(ctx, ggml_mul(ctx, x1, b), a);
        x2 = ggml_sub(ctx, ggml_mul(ctx, x2, b), a);
    }
    ggml_cgraph * g1 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g1, x1);
    ggml_cgraph * g2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g2, x2);
    g1->uid = 101;
    g2->uid = 202;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    const std::vector<float> a0 = seeded(n, 1), a1 = seeded(n, 2), bv = seeded(n, 3);
    ggml_backend_tensor_set(b, bv.data(), 0, ggml_nbytes(b));

    // the plain results, nothing encoded ahead
    auto plain = [&](ggml_cgraph * g, ggml_tensor * out, const std::vector<float> & av) {
        ggml_backend_tensor_set(a, av.data(), 0, ggml_nbytes(a));
        ggml_backend_graph_compute(backend, g);
        return get(out);
    };
    const counts c0 = now();
    const std::vector<float> r1_a0 = plain(g1, x1, a0), r1_a1 = plain(g1, x1, a1);
    const std::vector<float> r2_a0 = plain(g2, x2, a0), r2_a1 = plain(g2, x2, a1);
    check(!same(r2_a0, r2_a1) && !same(r1_a0, r1_a1) && !same(r1_a1, r2_a1), "the cases' results differ from each other");
    check(now().encoded == c0.encoded && now().used == c0.used, "a plain compute encodes nothing ahead");

    // 1. encoded ahead, then its input rewritten: the graph reads the new values (the graph's intermediates still
    //    hold the other input's, from the plain run just before, so a node run out of order would show)
    {
        const counts c = now();
        encode_ahead(backend, g2);
        ggml_backend_tensor_set(a, a0.data(), 0, ggml_nbytes(a));
        ggml_backend_graph_compute(backend, g2);
        const counts d = now();
        check(same(get(x2), r2_a0), "encoded ahead, inputs written after: the new inputs' result");
        check(on ? d.encoded == c.encoded + 1 && d.used == c.used + 1 : d.encoded == c.encoded && d.used == c.used,
                on ? "encoded ahead and committed" : "off: nothing encoded ahead");
    }
    // 2. encoded ahead for g2, but g1 computed first: g1's own result, the encoding dropped; then g2 plain.
    //    Each output still holds the other input's result (x1 r1_a1 from the plain runs, x2 r2_a0 from case 1),
    //    so a graph that did not run at all would show, not only one that ran the wrong encoding.
    {
        ggml_backend_tensor_set(a, a0.data(), 0, ggml_nbytes(a));
        const counts c = now();
        encode_ahead(backend, g2);
        ggml_backend_graph_compute(backend, g1);
        const std::vector<float> got1 = get(x1);
        ggml_backend_tensor_set(a, a1.data(), 0, ggml_nbytes(a));
        ggml_backend_graph_compute(backend, g2);
        const std::vector<float> got2 = get(x2);
        const counts d = now();
        check(same(got1, r1_a0), "another graph first: its own result");
        check(same(got2, r2_a1), "then the graph that was encoded for: its plain result");
        check(d.used == c.used, "another graph first: the encoding was dropped, not committed");
    }
    // 3. encoded ahead, then the same graph object rebuilt (a new uid): dropped
    {
        ggml_backend_tensor_set(a, a0.data(), 0, ggml_nbytes(a));
        const counts c = now();
        encode_ahead(backend, g2);
        g2->uid = 303;
        ggml_backend_graph_compute(backend, g2);
        const counts d = now();
        check(same(get(x2), r2_a0), "a rebuilt graph: its plain result");
        check(d.used == c.used, "a rebuilt graph: the encoding was dropped, not committed");
    }
    // 4. many rounds, encoded ahead then computed, alternating graphs and inputs
    {
        bool all = true;
        const counts c = now();
        for (int r = 0; r < 20; r++) {
            const bool first = r % 2 == 1;
            const bool new_a = r % 3 == 0;
            plain(first ? g1 : g2, first ? x1 : x2, new_a ? a0 : a1); // the intermediates hold the other input's
            ggml_backend_tensor_set(a, (new_a ? a1 : a0).data(), 0, ggml_nbytes(a));
            encode_ahead(backend, first ? g1 : g2);
            ggml_backend_graph_compute(backend, first ? g1 : g2);
            const std::vector<float> & want = first ? (new_a ? r1_a1 : r1_a0) : (new_a ? r2_a1 : r2_a0);
            all = all && same(get(first ? x1 : x2), want);
        }
        const counts d = now();
        check(all, "20 rounds encoded ahead: every result its plain one");
        check(on ? d.used - c.used == 20 && d.encoded - c.encoded == 20 : d.encoded == c.encoded,
                on ? "20 rounds: 20 encoded ahead, 20 committed" : "off: nothing encoded ahead");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    printf("%s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}

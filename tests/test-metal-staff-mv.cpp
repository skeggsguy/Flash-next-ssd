// #23 (STAFF-MATRIX-PLAN.md step 2): the staff's Q8_0 mat-vec on the matrix units for a check of 2-16 words
// (GGML_METAL_STAFF_MV, ggml-metal-staff-mv.h).
//
// The kernel's sums are not today's (half columns, matmul2d's order), so the check is against a reference:
// each output within a half-rounding bound of the exact sum (in double, from the CPU's dequantized weights).
// Four parts:
//
// - the switch and the rule, on the CPU: "1" on, unset/"0" today's route, off without the tensor API; the
//   fewest columns (4 where mul_mv_ext takes the product, k a multiple of 128, else 2), the most (16), the
//   fewest rows (2,048), Q8_0 x F32 with a column's values side by side;
// - the route, from the pipelines the library compiles (its debug log): on, the staff kernel at 8 and 16
//   columns' padding, and none for 3 columns of a mul_mv_ext shape; at "0" never;
// - the sums, on MTL0: every column count on rows off 32's multiple, k off the simdgroups' multiple (10 and
//   81 blocks) and a broadcast batch; a column with values past half's range (65504) must still come out
//   within F32's bound (the kernel redoes such a share in F32), and the columns beside it within half's;
// - the bytes: a product repeated gives the same bytes, and a column's bytes do not depend on the columns
//   beside it at the same padding (5 columns are the first 5 of 8, 9 the first 9 of 16).
// Beside them, the route's start-up line: its three texts, and one line a context from ggml_metal_init at WARN
// (a server installs its log callback after the device init, and shows the library's INFO only at -lv 4).
//
// ctest runs it with the switch at "1", at "0" and unset (today's sums, which the bound holds too; unset is today's
// route since 2026-10-15, the fork's proven defaults).

#include "testing.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Metal-internal (ggml-metal-staff-mv.h), linked from ggml-metal
extern "C" int  ggml_metal_staff_mv_parse(const char * s, bool * bad);
extern "C" int  ggml_metal_staff_mv_limit(int staff_mv, bool has_tensor);
extern "C" int  ggml_metal_staff_mv_min_cols(int64_t ne00);
extern "C" bool ggml_metal_staff_mv_takes(const struct ggml_tensor * op, int limit);
extern "C" const char * ggml_metal_staff_mv_route_line(char * buf, size_t size, int staff_mv, bool has_tensor);

namespace {

const std::string STAFF_BASE = "kernel_mul_mv_staff_q8_0_f32_m";

// the pipelines the Metal library compiled, from its debug log
std::set<std::string> g_compiled;

// the route's start-up lines as logged, with their level
const char * ROUTE_KEY = "staff mat-vec (Q8_0)  = ";
std::vector<std::pair<ggml_log_level, std::string>> g_route;

void log_cb(enum ggml_log_level level, const char * text, void * user) {
    (void) user;
    if (strstr(text, ROUTE_KEY) != nullptr) {
        g_route.emplace_back(level, text);
    }
    const char * key = "compiling pipeline: base = '";
    const char * p   = strstr(text, key);
    if (p != nullptr) {
        p += strlen(key);
        const char * e = strchr(p, '\'');
        if (e != nullptr) {
            g_compiled.insert(std::string(p, e - p));
        }
        return;
    }
    if (level != GGML_LOG_LEVEL_DEBUG) {
        fputs(text, stderr);
    }
}

bool compiled(const std::string & prefix) {
    for (const std::string & s : g_compiled) {
        if (s.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
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

constexpr int64_t ALL = 16; // the columns the seeded src1 holds a batch; a product takes the first n

struct product {
    int64_t M   = 2048;
    int64_t k   = 2560;
    int64_t n   = 4;
    int64_t b0  = 1;    // src0 matrices
    int64_t b1  = 1;    // src1 batches (a multiple of b0: broadcast)
    float   big = 0.0f; // if nonzero: column 1's values in block 1 are +-this (past half's range at 1e5)
};

struct result {
    std::vector<float>  out;   // [batch][column][row]
    std::vector<double> exact; // the same, in double from the dequantized weights
    std::vector<double> bound; // sum |w y| an output
};

result run(ggml_backend_t backend, const product & p) {
    ggml_init_params ip = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, p.k, p.M, p.b0);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, p.k, p.n, p.b1);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    const std::vector<float> fa = seeded((size_t) (p.k*p.M*p.b0), 4321u + (uint32_t) (p.k + p.M));
    std::vector<uint8_t> q(ggml_nbytes(a));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, fa.data(), q.data(), 0, p.M*p.b0, p.k, nullptr);
    ggml_backend_tensor_set(a, q.data(), 0, q.size());
    std::vector<float> w((size_t) (p.k*p.M*p.b0));
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q.data(), w.data(), (int64_t) w.size());

    const std::vector<float> all = seeded((size_t) (p.k*ALL*p.b1), 77u + (uint32_t) p.k);
    std::vector<float> y;
    for (int64_t ib = 0; ib < p.b1; ib++) {
        y.insert(y.end(), all.begin() + ib*ALL*p.k, all.begin() + (ib*ALL + p.n)*p.k);
        if (p.big != 0.0f) {
            for (int64_t i = 32; i < 64; i++) {
                y[(ib*p.n + 1)*p.k + i] = (i % 2 ? 1.0f : -0.5f)*p.big;
            }
        }
    }
    ggml_backend_tensor_set(b, y.data(), 0, y.size()*sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    result r;
    r.out.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, r.out.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    for (int64_t ib = 0; ib < p.b1; ib++) {
        const float * wm = w.data() + (ib/(p.b1/p.b0))*p.M*p.k;
        for (int64_t c = 0; c < p.n; c++) {
            const float * yc = y.data() + (ib*p.n + c)*p.k;
            for (int64_t row = 0; row < p.M; row++) {
                double s = 0.0, m = 0.0;
                for (int64_t i = 0; i < p.k; i++) {
                    s += (double) wm[row*p.k + i]*yc[i];
                    m += std::fabs((double) wm[row*p.k + i]*yc[i]);
                }
                r.exact.push_back(s);
                r.bound.push_back(m);
            }
        }
    }
    return r;
}

std::string name(const product & p) {
    return "m=" + std::to_string(p.M) + " k=" + std::to_string(p.k) + " n=" + std::to_string(p.n) +
           (p.b1 > 1 ? " batches=" + std::to_string(p.b0) + "/" + std::to_string(p.b1) : "") +
           (p.big != 0.0f ? " big=" + std::to_string((int) p.big) : "");
}

// the outputs of columns [c0, c1) of every batch within rel * sum |w y| of the exact sum, and finite; the
// first miss is reported in why
bool within(const result & r, const product & p, int64_t c0, int64_t c1, double rel, std::string & why) {
    for (int64_t ib = 0; ib < p.b1; ib++) {
        for (int64_t c = c0; c < c1; c++) {
            for (int64_t row = 0; row < p.M; row++) {
                const size_t i = (size_t) ((ib*p.n + c)*p.M + row);
                const double err = std::fabs((double) r.out[i] - r.exact[i]);
                if (!std::isfinite(r.out[i]) || err > rel*r.bound[i] + 1e-6) {
                    why = name(p) + ": batch " + std::to_string(ib) + " column " + std::to_string(c) + " row " +
                          std::to_string(row) + " is " + std::to_string(r.out[i]) + ", exact " + std::to_string(r.exact[i]);
                    return false;
                }
            }
        }
    }
    return true;
}

// columns [0, n) of each batch of a result of p
std::vector<float> first(const result & r, const product & p, int64_t n) {
    std::vector<float> res;
    for (int64_t ib = 0; ib < p.b1; ib++) {
        res.insert(res.end(), r.out.begin() + ib*p.n*p.M, r.out.begin() + (ib*p.n + n)*p.M);
    }
    return res;
}

void test_switch(testing & t) {
    bool bad = true;
    // off unless "1" since 2026-10-15 (the study's C-staff rung kept it off: the fork's proven defaults)
    t.assert_equal("unset is today's route", 0, ggml_metal_staff_mv_parse(nullptr, &bad));
    t.assert_true("unset is not bad", !bad);
    t.assert_equal("empty is today's route", 0, ggml_metal_staff_mv_parse("", &bad));
    t.assert_true("empty is not bad", !bad);
    t.assert_equal("\"1\" is on", 1, ggml_metal_staff_mv_parse("1", &bad));
    t.assert_true("\"1\" is not bad", !bad);
    t.assert_equal("\"0\" is today's route", 0, ggml_metal_staff_mv_parse("0", &bad));
    t.assert_true("\"0\" is not bad", !bad);
    t.assert_equal("a word is today's route", 0, ggml_metal_staff_mv_parse("on", &bad));
    t.assert_true("a word is bad", bad);
    t.assert_equal("\"2\" is today's route", 0, ggml_metal_staff_mv_parse("2", &bad));
    t.assert_true("\"2\" is bad", bad);
}

void test_rule(testing & t) {
    t.assert_equal("on with the tensor API: 16", 16, ggml_metal_staff_mv_limit(1, true));
    t.assert_equal("on without the tensor API: none", 0, ggml_metal_staff_mv_limit(1, false));
    t.assert_equal("off: none", 0, ggml_metal_staff_mv_limit(0, true));
    t.assert_equal("k 2560 (mul_mv_ext's): from 4 columns", 4, ggml_metal_staff_mv_min_cols(2560));
    t.assert_equal("k 640 (mul_mv_ext's): from 4 columns", 4, ggml_metal_staff_mv_min_cols(640));
    t.assert_equal("k 320 (not mul_mv_ext's): from 2 columns", 2, ggml_metal_staff_mv_min_cols(320));
    t.assert_equal("k 2592 (not mul_mv_ext's): from 2 columns", 2, ggml_metal_staff_mv_min_cols(2592));

    ggml_init_params ip = { ggml_tensor_overhead()*96, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    auto takes = [&](ggml_type ta, ggml_type tb, int64_t m, int64_t k, int64_t n, int limit) {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, ta, k, m);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, tb, k, n);
        return ggml_metal_staff_mv_takes(ggml_mul_mat(ctx, a, b), limit);
    };
    const ggml_type Q8 = GGML_TYPE_Q8_0, F32 = GGML_TYPE_F32;
    t.assert_true("4 columns", takes(Q8, F32, 2048, 2560, 4, 16));
    t.assert_true("16 columns", takes(Q8, F32, 248320, 2560, 16, 16));
    t.assert_true("not 3 columns of a mul_mv_ext shape", !takes(Q8, F32, 10240, 2560, 3, 16));
    t.assert_true("2 columns of k 320", takes(Q8, F32, 10240, 320, 2, 16));
    t.assert_true("not 1 column", !takes(Q8, F32, 10240, 320, 1, 16));
    t.assert_true("not 17 columns", !takes(Q8, F32, 10240, 2560, 17, 16));
    t.assert_true("not 2,047 rows", !takes(Q8, F32, 2047, 2560, 8, 16));
    t.assert_true("not when off", !takes(Q8, F32, 10240, 2560, 8, 0));
    t.assert_true("not q4_0", !takes(GGML_TYPE_Q4_0, F32, 10240, 2560, 8, 16));
    t.assert_true("not f32 weights", !takes(F32, F32, 10240, 2560, 8, 16));
    t.assert_true("not f16 columns", !takes(Q8, GGML_TYPE_F16, 10240, 2560, 8, 16));
    // a column's values a row apart (a transposed view): not side by side
    ggml_tensor * a  = ggml_new_tensor_2d(ctx, Q8, 2560, 2048);
    ggml_tensor * bt = ggml_transpose(ctx, ggml_new_tensor_2d(ctx, F32, 8, 2560));
    t.assert_true("not strided columns", !ggml_metal_staff_mv_takes(ggml_mul_mat(ctx, a, bt), 16));
    ggml_free(ctx);
}

void test_line(testing & t) {
    char buf[160];
    auto line = [&](int staff_mv, bool has_tensor) {
        return std::string(ggml_metal_staff_mv_route_line(buf, sizeof(buf), staff_mv, has_tensor));
    };
    t.assert_equal("on", std::string("staff mat-vec (Q8_0)  = on, up to 16 columns on 2048+ rows (GGML_METAL_STAFF_MV)"), line(1, true));
    t.assert_equal("on without the tensor API", std::string("staff mat-vec (Q8_0)  = off: it needs the tensor API (GGML_METAL_STAFF_MV)"), line(1, false));
    t.assert_equal("\"0\"", std::string("staff mat-vec (Q8_0)  = off, today's route (GGML_METAL_STAFF_MV=0)"), line(0, true));
    t.assert_equal("\"0\" without the tensor API", line(0, true), line(0, false));
    char small[12];
    t.assert_equal("a short buffer: cut, ended", std::string("staff mat-"), std::string(ggml_metal_staff_mv_route_line(small, 11, 1, true)));
}

// the context lines (ggml_metal_init's) so far
std::vector<std::pair<ggml_log_level, std::string>> context_lines() {
    std::vector<std::pair<ggml_log_level, std::string>> res;
    for (const auto & l : g_route) {
        if (l.second.rfind("ggml_metal_init: ", 0) == 0) {
            res.push_back(l);
        }
    }
    return res;
}

void test_startup(testing & t, ggml_backend_dev_t dev, bool on) {
    char buf[160];
    const std::string want = std::string("ggml_metal_init: ") + ggml_metal_staff_mv_route_line(buf, sizeof(buf), on, true) + "\n";
    auto lines = context_lines();
    t.assert_equal("one route line from the first context", (size_t) 1, lines.size());
    ggml_backend_t second = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_free(second);
    lines = context_lines();
    t.assert_equal("one more from a second context", (size_t) 2, lines.size());
    for (const auto & l : lines) {
        t.assert_equal("the line says the route asked for", want, l.second);
        t.assert_true("at WARN, so a server log shows it at its default verbosity", l.first == GGML_LOG_LEVEL_WARN);
    }
}

void test_route(testing & t, ggml_backend_t backend, bool on) {
    run(backend, { 2048, 2560, 3 });
    t.assert_true("3 columns of a mul_mv_ext shape: no staff kernel", !compiled(STAFF_BASE));
    run(backend, { 2048, 2560, 5 });
    t.assert_true(std::string("5 columns: the staff kernel at 8 ") + (on ? "compiled" : "never compiled"), compiled(STAFF_BASE + "8") == on);
    run(backend, { 2048, 2560, 12 });
    t.assert_true(std::string("12 columns: the staff kernel at 16 ") + (on ? "compiled" : "never compiled"), compiled(STAFF_BASE + "16") == on);
}

void check(testing & t, const std::string & what, const result & r, const product & p, int64_t c0, int64_t c1, double rel) {
    std::string why;
    const bool ok = within(r, p, c0, c1, rel, why);
    t.assert_true(name(p) + ": " + what, ok);
    if (!ok) {
        t.assert_true(why, false);
    }
}

void test_sums(testing & t, ggml_backend_t backend, bool on) {
    std::vector<product> ps;
    for (int64_t n = 2; n <= 16; n++) {
        ps.push_back({ 2051, 320, n });       // rows off 32's multiple; 10 blocks, off the simdgroups' 4
    }
    for (int64_t n : { 4, 5, 8, 9, 16 }) {
        ps.push_back({ 2048, 2592, n });      // 81 blocks
        ps.push_back({ 2048, 640, n, 2, 4 }); // a broadcast batch
    }
    // half rounds each input to 2^-11 of itself and the sums are F32: 1e-3 of sum |w y| is twice that bound
    for (const product & p : ps) {
        check(t, "within half's bound", run(backend, p), p, 0, p.n, 1e-3);
    }
    // a column past half's range: its share is redone in F32, so it is within F32's bound (at "0", 12 columns
    // of k 320 are mul_mm's, whose 16-bit inputs hold it only to half's)
    for (product p : { product{ 2048, 2560, 5 }, product{ 2051, 320, 12 }, product{ 2048, 640, 9, 2, 4 } }) {
        p.big = 1e5f;
        const result r = run(backend, p);
        if (on) {
            check(t, "the big column within F32's bound", r, p, 1, 2, 1e-5);
        }
        check(t, "every column within half's bound", r, p, 0, p.n, 1e-3);
    }
}

void test_bytes(testing & t, ggml_backend_t backend) {
    const product p8 = { 2051, 2560, 8 };
    const result a = run(backend, p8);
    t.assert_true("a product repeated: the same bytes", a.out == run(backend, p8).out);
    const product p5 = { 2051, 2560, 5 };
    t.assert_true("5 columns: the first 5 of 8", run(backend, p5).out == first(a, p8, 5));
    const product p16 = { 2051, 320, 16, 1, 2 }, p9 = { 2051, 320, 9, 1, 2 };
    t.assert_true("9 columns: the first 9 of 16", run(backend, p9).out == first(run(backend, p16), p16, 9));
}

} // namespace

int main(int argc, char ** argv) {
    testing t(std::cout);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    const char * env = getenv("GGML_METAL_STAFF_MV");
    // read here, not by the parser under test: ctest sets it "1", "0" or unset (today's route since 2026-10-15)
    const bool on = env != nullptr && strcmp(env, "1") == 0;
    printf("GGML_METAL_STAFF_MV=%s: %s\n", env ? env : "(unset)", on ? "on" : "today's route");

    t.test("the switch", test_switch);
    t.test("the rule", test_rule);
    t.test("the route line", test_line);

    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (dev == nullptr) {
        printf("no MTL0 device: the sums skipped\n");
        return t.summary();
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    t.test("the start-up lines", [&](testing & t) { test_startup(t, dev, on); });
    // the route first: a pipeline is compiled (and logged) only the first time it is asked for
    t.test("the route", [&](testing & t) { test_route(t, backend, on); });
    t.test("the sums", [&](testing & t) { test_sums(t, backend, on); });
    t.test("the bytes", [&](testing & t) { test_bytes(t, backend); });
    ggml_backend_free(backend);
    return t.summary();
}

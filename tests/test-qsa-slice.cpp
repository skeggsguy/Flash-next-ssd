// LLAMA_QSA_SLICE's parse and the slice plan (src/models/qwen4exp-qsa-slice.h): worked by hand, then
// the plan's promises checked over every batch length a 4,096-token ubatch can have. The graph side
// (every row's bytes as the whole batch's) is test-qsa-keep's slice runs (QSA_KEEP_SLICE).

#include "../src/models/qwen4exp-qsa-slice.h"

#include <cstdio>
#include <initializer_list>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static void test_parse() {
    CHECK(llama_qsa_slice_parse(nullptr) == LLAMA_QSA_SLICE_DEFAULT);
    CHECK(llama_qsa_slice_parse("0")     == 0);
    CHECK(llama_qsa_slice_parse("")      == 0);   // not a number: off, like the other QSA switches
    CHECK(llama_qsa_slice_parse("off")   == 0);
    CHECK(llama_qsa_slice_parse("-512")  == 0);
    CHECK(llama_qsa_slice_parse("512")   == 512);
    CHECK(llama_qsa_slice_parse("256")   == 256);
    CHECK(llama_qsa_slice_parse("500")   == 480); // down to a multiple of 32
    CHECK(llama_qsa_slice_parse("128")   == 128);
    CHECK(llama_qsa_slice_parse("127")   == 128); // at least 128
    CHECK(llama_qsa_slice_parse("64")    == 128);
    CHECK(llama_qsa_slice_parse("1")     == 128);
    CHECK(llama_qsa_slice_parse("4096")  == 4096);
    // absurdly large: a multiple of 32 and at least 128 still, never wrapped under the floor by the cast
    CHECK(llama_qsa_slice_parse("4294967328") % 32 == 0 && llama_qsa_slice_parse("4294967328") >= LLAMA_QSA_SLICE_MIN);
    CHECK(llama_qsa_slice_parse("99999999999999999999") % 32 == 0 && llama_qsa_slice_parse("99999999999999999999") >= LLAMA_QSA_SLICE_MIN);
}

static void test_plan_by_hand() {
    // off, or a batch that fits in one slice: no slices (today's graph)
    CHECK(llama_qsa_slice_plan(4096, 0).empty());
    CHECK(llama_qsa_slice_plan(512, 512).empty());
    CHECK(llama_qsa_slice_plan(1, 512).empty());

    // the run's batch: 8 slices of 512
    {
        const auto p = llama_qsa_slice_plan(4096, 512);
        CHECK(p.size() == 8);
        for (size_t i = 0; i < p.size(); ++i) {
            CHECK(p[i].t0 == (int64_t) (512*i) && p[i].n == 512);
        }
    }

    // one row over: two slices, 256 + 257 (17 chunks of 32, 9 to the last slice, which keeps the odd row)
    {
        const auto p = llama_qsa_slice_plan(513, 512);
        CHECK(p.size() == 2);
        CHECK(p[0].t0 == 0   && p[0].n == 256);
        CHECK(p[1].t0 == 256 && p[1].n == 257);
    }

    // test-qsa-keep's shapes: 512 rows at 128 are 4 slices of 128; 178 rows (6 chunks, 3 a slice) are 96 + 82
    {
        const auto p = llama_qsa_slice_plan(512, 128);
        CHECK(p.size() == 4);
        for (size_t i = 0; i < p.size(); ++i) {
            CHECK(p[i].t0 == (int64_t) (128*i) && p[i].n == 128);
        }
        const auto q = llama_qsa_slice_plan(178, 128);
        CHECK(q.size() == 2);
        CHECK(q[0].t0 == 0 && q[0].n == 96 && q[1].t0 == 96 && q[1].n == 82);
    }

    // where "ceil(n/k) rows rounded up to 32, the rest last" would leave a 10-row tail: 3,850 at 256
    {
        const auto p = llama_qsa_slice_plan(3850, 256);
        CHECK(p.size() == 16);
        CHECK(p.back().n >= LLAMA_QSA_SLICE_MIN_ROWS);
        CHECK(p.back().t0 + p.back().n == 3850);
    }

    // the smallest slice, one row over: 64 + 65
    {
        const auto p = llama_qsa_slice_plan(129, 128);
        CHECK(p.size() == 2);
        CHECK(p[0].n == 64 && p[1].t0 == 64 && p[1].n == 65);
    }
}

// the promises, over every batch length up to a 4,096 ubatch and every slice from 128 to 1,024
static void test_plan_promises() {
    for (uint32_t rows = LLAMA_QSA_SLICE_MIN; rows <= 1024; rows += 32) {
        for (int64_t n = 1; n <= 4096; ++n) {
            const auto p = llama_qsa_slice_plan(n, rows);

            if (n <= (int64_t) rows) {
                if (!p.empty()) { fprintf(stderr, "rows %u n %lld: sliced a batch that fits\n", rows, (long long) n); n_fail++; }
                continue;
            }

            const int64_t k = (n + rows - 1)/rows;
            bool ok = (int64_t) p.size() == k;

            int64_t t0 = 0;
            for (size_t i = 0; ok && i < p.size(); ++i) {
                ok = p[i].t0 == t0                            // back to back from row 0
                  && p[i].n <= (int64_t) rows                 // never longer than the reserve sized
                  && p[i].n >= LLAMA_QSA_SLICE_MIN_ROWS       // never attention's few-row paths
                  && (i + 1 == p.size() || p[i].n % 32 == 0)  // every slice starts on a multiple of 32
                  && (i == 0 || p[i].n >= p[i - 1].n - 32 || i + 1 == p.size()); // even
                t0 += p[i].n;
            }
            ok = ok && t0 == n;

            if (!ok) {
                fprintf(stderr, "rows %u n %lld: bad plan:", rows, (long long) n);
                for (const auto & s : p) {
                    fprintf(stderr, " [%lld,+%lld)", (long long) s.t0, (long long) s.n);
                }
                fprintf(stderr, "\n");
                n_fail++;
            }
        }
    }
}

int main() {
    test_parse();
    test_plan_by_hand();
    test_plan_promises();

    if (n_fail == 0) {
        printf("test-qsa-slice: all checks passed\n");
    }
    return n_fail == 0 ? 0 : 1;
}

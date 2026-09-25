// Metal's radix top-k must give the same picks, in the same order, every run.
//
// qwen4exp's sparse attention picks the blocks each token reads with ggml_top_k (513 of the context's
// blocks when reading in, 2304 cells when writing past ~9K of context). Once a row is longer than 2048,
// Metal answers with the radix-select kernel (ggml_metal_op_top_k_radix), which used to write its picks
// through an atomic counter: the order of the picks, and which of the values tied at the threshold filled
// the last places, changed from run to run. The indexer's scores are relu'd sums, so many blocks tie at
// 0 (and future blocks at -inf); a ~10K-token prompt at temperature 0 gave five different answers in
// five runs (T4 rerun, 2026-09-26). Reading in only unmasks the picked cells, so there the tie fill
// matters; the writing path gathers K/V in pick order and attends over them in that order, so there
// the order matters too.
//
// The kernel now promises a stated result: every value above the k-th largest, then the lowest-index
// values equal to it, all in ascending index order. That is what this test holds it to, row by row,
// against a reference built by a plain stable sort, on data made to tie at the threshold. It also runs
// every case several times, with the output reset in between, and requires the same bytes each time.
//
// Every case here is on the radix path (ggml_metal_op_top_k: ncols > 2048 and k > 64, or more than
// 4 rows of at least 8192). The bitonic path answers in descending value order, so a case that fell
// back to it would fail the order check rather than pass unseen.
//
// Values are never -0.0 or NaN: the kernel orders floats by their bits (+0 above -0), a plain float
// comparison does not, and the reference below uses the latter.

#include "testing.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N_RUNS = 8;

enum fill_kind {
    FILL_GROUPS,    // floor(i/group), shuffled: every value repeats `group` times
    FILL_INDEXER,   // the block indexer's scores: relu zeros, repeated positives, -inf past the query
    FILL_CELLS,     // one score per block of 4 cells, repeated scores: the writing path's cell top-k
    FILL_EQUAL,     // one value everywhere: the picks must be 0 .. k-1
};

struct topk_case {
    std::string name;
    int64_t     ncols;
    int64_t     nrows;
    int         k;
    fill_kind   fill;
    int         group; // FILL_GROUPS only
};

std::vector<float> make_input(const topk_case & tc, std::mt19937 & rng) {
    std::vector<float> x(tc.ncols * tc.nrows);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    for (int64_t r = 0; r < tc.nrows; r++) {
        float * row = x.data() + r * tc.ncols;
        switch (tc.fill) {
            case FILL_GROUPS: {
                for (int64_t i = 0; i < tc.ncols; i++) {
                    row[i] = (float) (i / tc.group);
                }
                std::shuffle(row, row + tc.ncols, rng);
            } break;
            case FILL_INDEXER: {
                // the query at row r sees the first `visible` blocks; some rows see fewer than k
                const int64_t visible = std::min<int64_t>(tc.ncols, 64 + r * tc.ncols / tc.nrows);
                for (int64_t i = 0; i < tc.ncols; i++) {
                    if (i >= visible) {
                        row[i] = -INFINITY;
                    } else if (u(rng) < 0.7f) {
                        row[i] = 0.0f;
                    } else {
                        row[i] = 1.0f + std::floor(u(rng) * 64.0f) / 64.0f;
                    }
                }
            } break;
            case FILL_CELLS: {
                for (int64_t b = 0; b < tc.ncols / 4; b++) {
                    const float score = std::floor(u(rng) * 256.0f);
                    std::fill(row + 4 * b, row + 4 * b + 4, score);
                }
            } break;
            case FILL_EQUAL: {
                std::fill(row, row + tc.ncols, 0.5f);
            } break;
        }
    }

    return x;
}

// the stated result: a stable sort by value, largest first, keeps equal values in index order, so its
// first k are everything above the k-th largest plus the lowest-index ties; then ascending index
std::vector<int32_t> reference_picks(const float * row, int64_t ncols, int k) {
    std::vector<int32_t> idx(ncols);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [row](int32_t a, int32_t b) { return row[a] > row[b]; });
    idx.resize(k);
    std::sort(idx.begin(), idx.end());
    return idx;
}

// the sorted value just inside the picks equals the one just outside: the tie fill decides
bool ties_at_threshold(const float * row, int64_t ncols, int k) {
    std::vector<float> v(row, row + ncols);
    std::sort(v.begin(), v.end(), std::greater<float>());
    return k < ncols && v[k - 1] == v[k];
}

void run_case(testing & t, ggml_backend_t backend, const topk_case & tc) {
    std::mt19937 rng(42 + (uint32_t) (tc.ncols * 31 + tc.nrows * 7 + tc.k));
    const std::vector<float> x = make_input(tc, rng);

    int64_t rows_with_ties = 0;
    for (int64_t r = 0; r < tc.nrows; r++) {
        rows_with_ties += ties_at_threshold(x.data() + r * tc.ncols, tc.ncols, tc.k);
    }
    t.assert_true("the data ties at the threshold (" + std::to_string(rows_with_ties) + " of " +
                  std::to_string(tc.nrows) + " rows)", rows_with_ties > 0);

    ggml_init_params params = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.ncols, tc.nrows);
    ggml_tensor * out = ggml_top_k(ctx, a, tc.k);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);
    ggml_backend_tensor_set(a, x.data(), 0, ggml_nbytes(a));

    const std::vector<int32_t> unset(ggml_nelements(out), -1);

    std::vector<int32_t> first;
    int identical_runs = 0;
    for (int run = 0; run < N_RUNS; run++) {
        // a run that wrote nothing must not pass by leaving the previous run's answer behind
        ggml_backend_tensor_set(out, unset.data(), 0, ggml_nbytes(out));
        if (!t.assert_true("graph computed", ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS)) {
            break;
        }

        std::vector<int32_t> got(ggml_nelements(out));
        ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));

        if (run == 0) {
            first = got;
            continue;
        }
        identical_runs += memcmp(first.data(), got.data(), got.size() * sizeof(int32_t)) == 0;
    }
    t.assert_equal("runs byte-identical to the first", N_RUNS - 1, identical_runs);

    int64_t bad_rows  = 0;
    int64_t first_bad = -1;
    for (int64_t r = 0; r < tc.nrows && !first.empty(); r++) {
        const std::vector<int32_t> want = reference_picks(x.data() + r * tc.ncols, tc.ncols, tc.k);
        if (memcmp(want.data(), first.data() + r * tc.k, tc.k * sizeof(int32_t)) != 0) {
            bad_rows++;
            first_bad = first_bad < 0 ? r : first_bad;
        }
    }
    t.assert_true("every row is the stated picks in ascending index order (" + std::to_string(bad_rows) +
                  " rows differ, first " + std::to_string(first_bad) + ")", !first.empty() && bad_rows == 0);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

} // namespace

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    t.verbose = verbose && std::string(verbose) == "1";

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    ggml_backend_load_all();

    // k = 2048 and k = 128 with one row and many; the indexer's and the writing path's real shapes;
    // a small k over many long rows (the radix path's second rule); and one value everywhere, with
    // more ties to place than the kernel's first tile of the row holds (4 per thread, 1024 threads)
    const std::vector<topk_case> cases = {
        { "k 2048 of 8192, 1 row, groups of 7",     8192,   1, 2048, FILL_GROUPS,  7 },
        { "k 2048 of 8192, 64 rows, groups of 7",   8192,  64, 2048, FILL_GROUPS,  7 },
        { "k 128 of 4096, 1 row, groups of 3",      4096,   1,  128, FILL_GROUPS,  3 },
        { "k 128 of 4096, 256 rows, groups of 3",   4096, 256,  128, FILL_GROUPS,  3 },
        { "k 16 of 8192, 16 rows, groups of 5",     8192,  16,   16, FILL_GROUPS,  5 },
        { "indexer: k 513 of 4096 blocks, 256 rows", 4096, 256,  513, FILL_INDEXER, 0 },
        { "writing: k 2304 of 9216 cells, 1 row",   9216,   1, 2304, FILL_CELLS,   0 },
        { "one value: k 6000 of 12288, 4 rows",    12288,   4, 6000, FILL_EQUAL,   0 },
    };

    bool found = false;
    for (size_t d = 0; d < ggml_backend_dev_count(); d++) {
        ggml_backend_dev_t dev  = ggml_backend_dev_get(d);
        const std::string  name = ggml_backend_dev_name(dev);
        if (name.rfind("MTL", 0) != 0) {
            continue;
        }
        found = true;

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend != nullptr);

        t.test(name, [&](testing & t) {
            for (const topk_case & tc : cases) {
                t.test(tc.name, [&](testing & t) { run_case(t, backend, tc); });
            }
        });

        ggml_backend_free(backend);
    }

    if (!found) {
        t.test("Metal", [](testing & t) { t.skip("no Metal device: the stated order is Metal's radix kernel's"); });
    }

    return t.summary();
}

// The phrasebook shelf's reads (src/llama-ple-shelf-io.h) on tables written to temporary files: every row
// it hands back is byte-for-byte what the table holds, turned into numbers the way ggml_get_rows does it.
//
//   rows     F32 rows at shelves of 1, 5, 100 and 2,000 slots (1 and 5 far smaller than a batch, so every
//            batch evicts and passes rows): random batches with repeats, gathered and begun/waited in
//            turn, each against the table read back by hand; the counters add up
//   iq4_nl   the real phrasebook's type (90 B rows of 160 numbers): the shelf's rows against ggml_get_rows
//            on the CPU backend (the op the mmap path runs, with streaming's offload off) and on Metal,
//            byte-identical (the tiny model fixture's table is not IQ4_NL)
//   copies   two byte-identical copies are both read; a link to the same file is one copy; a copy of
//            another size is refused
//   failure  a read that fails (the file cut short under the shelf) throws and empties the shelf, and
//            the shelf serves the right bytes again once the file is back

#include "../src/llama-ple-shelf-io.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

static const size_t OFFS = 4099; // row 0's file offset: not aligned to anything

static std::string tmp_path(const std::string & name) {
    const char * d = getenv("TMPDIR");
    std::string dir = d && *d ? d : "/tmp";
    if (dir.back() != '/') { dir += '/'; }
    return dir + "ple-shelf-" + name + "-" + std::to_string(getpid()) + ".bin";
}

static void write_file(const std::string & path, const std::vector<uint8_t> & table) {
    FILE * f = fopen(path.c_str(), "wb");
    std::vector<uint8_t> head(OFFS, 0xAB);
    fwrite(head.data(), 1, head.size(), f);
    fwrite(table.data(), 1, table.size(), f);
    fclose(f);
}

static llama_ple_shelf_config config(const std::vector<std::string> & paths, ggml_type type, int64_t relems,
                                     int64_t n_rows, int64_t slots) {
    llama_ple_shelf_config c;
    c.paths = paths; c.offs = OFFS; c.type = type; c.row_elems = relems; c.n_rows = n_rows;
    c.mib = 1; c.n_readers = 8; c.rows_per_token = 4; c.slots = slots;
    return c;
}

// the table's rows by hand, as F32
static std::vector<float> reference(const std::vector<float> & values, int64_t relems, const std::vector<int32_t> & rows) {
    std::vector<float> out;
    for (int32_t r : rows) {
        out.insert(out.end(), values.begin() + (size_t) r * relems, values.begin() + (size_t) (r + 1) * relems);
    }
    return out;
}

static std::vector<int32_t> random_batch(std::mt19937 & rng, int64_t n_rows, int n) {
    std::vector<int32_t> rows(n);
    for (auto & r : rows) {
        r = rng() % 3 == 0 ? (int32_t) (rng() % n_rows) : (int32_t) (rng() % 40); // a hot head and a long tail
    }
    return rows;
}

static void test_rows() {
    const int64_t relems = 8, n_rows = 3000;
    std::vector<float> values(relems * n_rows);
    std::mt19937 rng(11);
    for (auto & v : values) { v = std::uniform_real_distribution<float>(-1, 1)(rng); }
    const std::string path = tmp_path("f32");
    write_file(path, std::vector<uint8_t>((const uint8_t *) values.data(), (const uint8_t *) (values.data() + values.size())));

    for (int64_t slots : {1, 5, 100, 2000}) {
        llama_ple_shelf_io io(config({path}, GGML_TYPE_F32, relems, n_rows, slots));
        CHECK(io.capacity() == slots && io.n_copies() == 1, "capacity %lld", (long long) io.capacity());
        int64_t bad = 0, asks = 0;
        for (int b = 0; b < 400; ++b) {
            const int n = 4 * (b % 7 == 0 ? 1 + (int) (rng() % 200) : 1 + (int) (rng() % 6));
            const std::vector<int32_t> rows = random_batch(rng, n_rows, n);
            std::vector<float> got(rows.size() * relems, -99.0f);
            if (b % 2) {
                io.gather(rows.data(), (int64_t) rows.size(), got.data());
            } else {
                auto pending = io.begin(rows.data(), (int64_t) rows.size(), got.data());
                io.wait(pending);
            }
            const std::vector<float> want = reference(values, relems, rows);
            bad += memcmp(got.data(), want.data(), got.size() * sizeof(float)) != 0;
            asks += (int64_t) std::set<int32_t>(rows.begin(), rows.end()).size();
        }
        const llama_ple_shelf_stats s = io.stats();
        const int64_t hits = s.hits[0] + s.hits[1], reads = s.reads[0] + s.reads[1];
        CHECK(bad == 0, "%lld slots: %lld batches with wrong rows", (long long) slots, (long long) bad);
        CHECK(s.calls == 400 && s.begun == 200 && s.asks[0] + s.asks[1] == asks && hits + reads == asks, "%lld slots: counters", (long long) slots);
        CHECK(s.bytes == reads * (int64_t) (relems * sizeof(float)) && io.held() <= slots, "%lld slots: bytes, held", (long long) slots);
        CHECK(s.asks[0] > 0 && s.asks[1] > 0, "both kinds of batch: writing %lld, reading in %lld", (long long) s.asks[0], (long long) s.asks[1]);
        CHECK(slots > 5 ? hits > 0 : true, "%lld slots: some hits", (long long) slots);
        CHECK(slots < 1000 ? s.passed > 0 : s.passed == 0, "%lld slots: %lld passed", (long long) slots, (long long) s.passed);
    }
    remove(path.c_str());
}

static std::vector<float> get_rows(ggml_backend_t backend, const std::vector<uint8_t> & table, int64_t relems,
                                   int64_t n_rows, const std::vector<int32_t> & rows) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t   = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, relems, n_rows);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) rows.size());
    ggml_tensor * out = ggml_get_rows(ctx, t, idx);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(t, table.data(), 0, table.size());
    ggml_backend_tensor_set(idx, rows.data(), 0, rows.size() * sizeof(int32_t));
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> res(rows.size() * relems);
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static void test_iq4_nl() {
    const int64_t relems = 160, n_rows = 4096;
    CHECK(ggml_row_size(GGML_TYPE_IQ4_NL, relems) == 90, "a phrasebook row is 90 B");
    std::vector<float> src(relems * n_rows);
    std::mt19937 rng(5);
    for (auto & v : src) { v = std::normal_distribution<float>(0, 0.05f)(rng); }
    std::vector<uint8_t> table(90 * n_rows);
    ggml_quantize_chunk(GGML_TYPE_IQ4_NL, src.data(), table.data(), 0, n_rows, relems, nullptr);
    const std::string path = tmp_path("iq4nl");
    write_file(path, table);

    llama_ple_shelf_io io(config({path}, GGML_TYPE_IQ4_NL, relems, n_rows, 700));
    std::vector<ggml_backend_t> backends = { ggml_backend_cpu_init() };
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu) { backends.push_back(ggml_backend_dev_init(gpu, nullptr)); }
    int64_t bad[2] = {0, 0}, nonzero = 0;
    for (int b = 0; b < 20; ++b) {
        const std::vector<int32_t> rows = random_batch(rng, n_rows, b % 2 ? 4 : 1024);
        std::vector<float> got(rows.size() * relems);
        io.gather(rows.data(), (int64_t) rows.size(), got.data());
        for (size_t k = 0; k < backends.size(); ++k) {
            const std::vector<float> want = get_rows(backends[k], table, relems, n_rows, rows);
            bad[k] += memcmp(got.data(), want.data(), got.size() * sizeof(float)) != 0;
        }
        for (float v : got) { nonzero += v != 0.0f; }
    }
    CHECK(bad[0] == 0, "CPU get_rows: %lld batches differ", (long long) bad[0]);
    CHECK(backends.size() < 2 || bad[1] == 0, "%s get_rows: %lld batches differ", ggml_backend_name(backends.back()), (long long) bad[1]);
    CHECK(nonzero > 0, "the rows are not all zero");
    printf("test-ple-shelf-io: iq4_nl compared with %zu backend(s), last %s\n", backends.size(), ggml_backend_name(backends.back()));
    for (auto * be : backends) { ggml_backend_free(be); }
    remove(path.c_str());
}

static void test_copies_and_failure() {
    const int64_t relems = 4, n_rows = 500;
    std::vector<float> values(relems * n_rows);
    for (size_t i = 0; i < values.size(); ++i) { values[i] = (float) i * 0.5f; }
    const std::vector<uint8_t> bytes((const uint8_t *) values.data(), (const uint8_t *) (values.data() + values.size()));
    const std::string a = tmp_path("copy-a"), b = tmp_path("copy-b"), l = tmp_path("copy-link"), s = tmp_path("copy-short");
    write_file(a, bytes);
    write_file(b, bytes);
    remove(l.c_str());
    CHECK(symlink(a.c_str(), l.c_str()) == 0, "symlink");
    write_file(s, std::vector<uint8_t>(bytes.begin(), bytes.end() - 16));

    std::mt19937 rng(3);
    {
        llama_ple_shelf_io io(config({a, b}, GGML_TYPE_F32, relems, n_rows, 50));
        CHECK(io.n_copies() == 2, "two copies");
        int64_t bad = 0;
        for (int k = 0; k < 60; ++k) {
            const std::vector<int32_t> rows = random_batch(rng, n_rows, 8);
            std::vector<float> got(rows.size() * relems);
            io.gather(rows.data(), (int64_t) rows.size(), got.data());
            bad += memcmp(got.data(), reference(values, relems, rows).data(), got.size() * sizeof(float)) != 0;
        }
        const llama_ple_shelf_stats st = io.stats();
        CHECK(bad == 0 && st.reads[0] > 0 && st.reads[1] > 0, "both copies read: file %lld, alt %lld",
                (long long) st.reads[0], (long long) st.reads[1]);
    }
    {
        llama_ple_shelf_io io(config({a, l}, GGML_TYPE_F32, relems, n_rows, 50));
        CHECK(io.n_copies() == 1, "a link to the same file is one copy");
    }
    bool threw = false;
    try { llama_ple_shelf_io io(config({a, s}, GGML_TYPE_F32, relems, n_rows, 50)); } catch (const std::exception &) { threw = true; }
    CHECK(threw, "a copy of another size is refused");
    threw = false;
    try { llama_ple_shelf_io io(config({s}, GGML_TYPE_F32, relems, n_rows, 50)); } catch (const std::exception &) { threw = true; }
    CHECK(threw, "a table longer than its file is refused");

    // the failure path: rows 0-9 on the shelf, then the file cut to its head; a batch that needs a read throws
    llama_ple_shelf_io io(config({a}, GGML_TYPE_F32, relems, n_rows, 50));
    std::vector<int32_t> warm = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> got(warm.size() * relems);
    io.gather(warm.data(), (int64_t) warm.size(), got.data());
    CHECK(io.held() == 10, "held %lld", (long long) io.held());
    CHECK(truncate(a.c_str(), (off_t) OFFS) == 0, "truncate");
    std::vector<int32_t> need = {3, 400, 401};
    std::vector<float> out(need.size() * relems);
    threw = false;
    try { io.gather(need.data(), (int64_t) need.size(), out.data()); } catch (const std::exception &) { threw = true; }
    CHECK(threw && io.held() == 0, "a failed read throws (%d) and empties the shelf (held %lld)", threw, (long long) io.held());
    write_file(a, bytes);
    io.gather(need.data(), (int64_t) need.size(), out.data());
    CHECK(memcmp(out.data(), reference(values, relems, need).data(), out.size() * sizeof(float)) == 0, "right again after");
    for (const auto & p : {a, b, l, s}) { remove(p.c_str()); }
}

int main() {
    ggml_backend_load_all();
    test_rows();
    test_iq4_nl();
    test_copies_and_failure();
    if (n_fail) {
        fprintf(stderr, "test-ple-shelf-io: %d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-ple-shelf-io: all checks passed\n");
    return 0;
}

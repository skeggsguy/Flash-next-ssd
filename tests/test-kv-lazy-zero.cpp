// LANES-PLAN amendment 3, S1 item 1 (engine piece (a), lazy zeros): with LLAMA_KV_LAZY_ZERO=1 a notepad
// (llama_kv_cache: the library's, the picker index's, the apprentice's) and the QSA keep store skip the
// memset at opening when their buffer is fresh vm_allocate memory (a Metal shared buffer on macOS), and
// clear(true) zeroes only the cells each stream has ever had written (padded to the attention width),
// so the pages of a notepad nothing writes are never touched and cost no RAM.
//
// Scenarios, each on a fresh context of two seats (two streams):
//   open      lazy off: the opening memset touched every page; lazy on: (KV_LAZY_EXPECT=1) at most a few
//             percent of the pages are resident, (KV_LAZY_EXPECT=0, the CPU fallback) all of them are.
//             Either way every byte of the notepads and the keep store reads zero.
//   clear     seat 0 reads in 300 tokens, seat 1 700, and a sentinel goes into each stream's last cell (far
//             above anything written); clear(true); the lazy path leaves the sentinel (it zeroes only the
//             written cells, padded), the fallback zeroes it; every other byte reads zero. A second round
//             (40 tokens on seat 1, a sentinel at cell 512, clear) checks the clear forgot the first round's mark.
//   copy      seat 1 reads in 600 tokens, its cells are copied to seat 0's stream (seq_cp -> the stream
//             copy in the next update), clear(true): seat 0's stream reads zero too.
//   restore   seat 1 reads in 500 tokens, the whole context state is saved, cleared and loaded back (the load
//             writes cells without placing a ubatch), clear(true): everything reads zero.
//   identity  the same operations with lazy off and on leave byte-identical notepads, keep store and logits.
//
// The opening check uses mincore(2): a page nothing has touched since vm_allocate is not resident. The Metal arms
// run with GGML_METAL_NO_RESIDENCY=1, because a residency set wires every page of a buffer (and so does the
// buffer's first use on the GPU), after which mincore reports them all resident: see LANES-PLAN S1's probe.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml-backend.h"

#include "../src/llama-kv-cache.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-hybrid-idx.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#define KV_LAZY_HAVE_MINCORE 1
#endif

static int n_fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        n_fail++; \
    } \
} while (0)

struct kv_tensor {
    const ggml_tensor * t;
    bool                trans; // a V cache transposed (flash attention off): a cell is one element of each row
    std::string         what;
};

struct mem_tensors {
    std::vector<kv_tensor>             kv;   // every notepad's K and V
    std::vector<const ggml_tensor *>   keep; // the QSA keep store
};

static void add_cache(mem_tensors & res, const llama_kv_cache * c, bool v_trans, const char * what) {
    if (c == nullptr) {
        return;
    }
    for (const uint32_t il : c->get_layer_ids()) {
        const ggml_tensor * k = c->get_k_storage(il);
        const ggml_tensor * v = c->get_v_storage(il);
        auto seen = [&](const ggml_tensor * t) {
            return std::any_of(res.kv.begin(), res.kv.end(), [&](const kv_tensor & x) { return x.t == t; });
        };
        if (k && !seen(k)) {
            res.kv.push_back({ k, false, std::string(what) + " K l" + std::to_string(il) });
        }
        if (v && !seen(v)) {
            res.kv.push_back({ v, v_trans, std::string(what) + " V l" + std::to_string(il) });
        }
    }
}

static mem_tensors collect(llama_context * ctx, bool v_trans) {
    mem_tensors res;
    llama_memory_t mem = llama_get_memory(ctx);

    if (auto * hi = dynamic_cast<llama_memory_hybrid_idx *>(mem)) {
        add_cache(res, hi->get_mem_attn(), v_trans, "attn");
        add_cache(res, hi->get_mem_idx(), false, "idx"); // the indexer keeps K only, never transposed
        for (const ggml_tensor * t : hi->qsa_keep_store()) {
            res.keep.push_back(t);
        }
    } else if (auto * hy = dynamic_cast<llama_memory_hybrid *>(mem)) {
        add_cache(res, hy->get_mem_attn(), v_trans, "attn");
    } else if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) {
        add_cache(res, kv, v_trans, "kv");
    }

    return res;
}

// byte ranges of tensor t that hold cells [c0, c1) of stream s
struct byte_range { size_t offs; size_t size; };

static std::vector<byte_range> cell_ranges(const kv_tensor & x, int64_t s, int64_t c0, int64_t c1) {
    const ggml_tensor * t = x.t;
    const int64_t kv_size = t->ne[1];
    c1 = std::min(c1, kv_size);
    std::vector<byte_range> res;
    if (c1 <= c0) {
        return res;
    }
    if (!x.trans) {
        res.push_back({ (size_t) (s*t->nb[2] + c0*t->nb[1]), (size_t) ((c1 - c0)*t->nb[1]) });
    } else {
        const size_t es = ggml_type_size(t->type);
        for (int64_t j = 0; j < t->ne[0]; ++j) {
            res.push_back({ (size_t) (s*t->nb[2] + (j*kv_size + c0)*es), (size_t) ((c1 - c0)*es) });
        }
    }
    return res;
}

static bool range_is_zero(const ggml_tensor * t, size_t offs, size_t size) {
    std::vector<uint8_t> buf(size);
    ggml_backend_tensor_get(t, buf.data(), offs, size);
    return std::all_of(buf.begin(), buf.end(), [](uint8_t b) { return b == 0; });
}

static bool tensor_is_zero(const ggml_tensor * t) {
    return range_is_zero(t, 0, ggml_nbytes(t));
}

#if KV_LAZY_HAVE_MINCORE
static size_t page_size() {
    return (size_t) getpagesize();
}

// pages of [p, p + n) that are resident
static size_t resident_pages(const void * p, size_t n) {
    if (n == 0) {
        return 0;
    }
    const size_t    pg = page_size();
    const uintptr_t a  = (uintptr_t) p & ~(uintptr_t) (pg - 1);
    const uintptr_t e  = ((uintptr_t) p + n + pg - 1) & ~(uintptr_t) (pg - 1);
    std::vector<char> vec((e - a)/pg);
    if (mincore((caddr_t) a, e - a, vec.data()) != 0) {
        perror("mincore");
        abort();
    }
    size_t res = 0;
    for (char c : vec) {
        res += (c & MINCORE_INCORE) ? 1 : 0;
    }
    return res;
}

static size_t span_pages(const void * p, size_t n) {
    const size_t    pg = page_size();
    const uintptr_t a  = (uintptr_t) p & ~(uintptr_t) (pg - 1);
    const uintptr_t e  = ((uintptr_t) p + n + pg - 1) & ~(uintptr_t) (pg - 1);
    return (e - a)/pg;
}
#endif

static std::vector<llama_token> make_tokens(int n, uint32_t seed, int n_vocab) {
    std::vector<llama_token> res(n);
    uint32_t x = seed*2654435761u + 12345u;
    for (int i = 0; i < n; ++i) {
        x = x*1664525u + 1013904223u;
        res[i] = (llama_token) ((x >> 8) % (uint32_t) n_vocab);
    }
    return res;
}

static bool decode_seq(llama_context * ctx, llama_seq_id s, int p0, int n, uint32_t seed) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const auto toks = make_tokens(n, seed, n_vocab);

    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = p0 + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = s;
        b.logits[i]    = i == n - 1;
    }
    b.n_tokens = n;

    const int ret = llama_decode(ctx, b);
    llama_batch_free(b);
    llama_synchronize(ctx);

    if (ret != 0) {
        fprintf(stderr, "llama_decode failed: %d (seq %d, %d tokens at %d)\n", ret, s, n, p0);
        return false;
    }
    return true;
}

struct env_guard {
    explicit env_guard(bool lazy) {
        if (lazy) {
            setenv("LLAMA_KV_LAZY_ZERO", "1", 1);
        } else {
            unsetenv("LLAMA_KV_LAZY_ZERO");
        }
    }
    ~env_guard() { unsetenv("LLAMA_KV_LAZY_ZERO"); }
};

struct test_cfg {
    llama_model *        model;
    llama_context_params cparams;
    bool                 v_trans;
    bool                 expect_lazy; // the lazy path applies to this backend's buffers
};

static llama_context * open_ctx(const test_cfg & cfg, bool lazy) {
    env_guard env(lazy);
    llama_context * ctx = llama_init_from_model(cfg.model, cfg.cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create the context\n");
        exit(1);
    }
    return ctx;
}

static size_t total_bytes(const mem_tensors & m) {
    size_t res = 0;
    for (const auto & x : m.kv)   { res += ggml_nbytes(x.t); }
    for (const auto * t : m.keep) { res += ggml_nbytes(t); }
    return res;
}

static void test_open(const test_cfg & cfg, bool lazy) {
    const char * arm = lazy ? "lazy" : "off";
    llama_context * ctx = open_ctx(cfg, lazy);
    const mem_tensors m = collect(ctx, cfg.v_trans);

    CHECK(!m.kv.empty(), "open/%s: no notepad tensors found", arm);

#if KV_LAZY_HAVE_MINCORE
    size_t res = 0;
    size_t all = 0;
    for (const auto & x : m.kv)   { res += resident_pages(x.t->data, ggml_nbytes(x.t)); all += span_pages(x.t->data, ggml_nbytes(x.t)); }
    size_t res_keep = 0;
    size_t all_keep = 0;
    for (const auto * t : m.keep) { res_keep += resident_pages(t->data, ggml_nbytes(t)); all_keep += span_pages(t->data, ggml_nbytes(t)); }

    fprintf(stderr, "open/%s: notepads %zu of %zu pages resident, keep store %zu of %zu (%.1f MiB in all)\n",
            arm, res, all, res_keep, all_keep, total_bytes(m)/1024.0/1024.0);

    if (lazy && cfg.expect_lazy) {
        CHECK(res*20 <= all, "open/lazy: %zu of %zu notepad pages resident, expected at most 5%%", res, all);
        CHECK(res_keep*20 <= all_keep, "open/lazy: %zu of %zu keep store pages resident, expected at most 5%%", res_keep, all_keep);
    } else {
        CHECK(res*20 >= all*19, "open/%s: only %zu of %zu notepad pages resident, the opening clear should touch them all", arm, res, all);
        CHECK(res_keep*20 >= all_keep*19, "open/%s: only %zu of %zu keep store pages resident", arm, res_keep, all_keep);
    }
#endif

    for (const auto & x : m.kv) {
        CHECK(tensor_is_zero(x.t), "open/%s: %s does not read zero", arm, x.what.c_str());
    }
    for (const auto * t : m.keep) {
        CHECK(tensor_is_zero(t), "open/%s: keep store %s does not read zero", arm, t->name);
    }

    llama_free(ctx);
}

// the cells of stream s written by n tokens hold something (else the zero checks after a clear prove nothing)
static void check_written(const mem_tensors & m, int64_t s, int64_t n, const char * where) {
    bool any = false;
    for (const auto & x : m.kv) {
        for (const auto & r : cell_ranges(x, s, 0, n)) {
            any = any || !range_is_zero(x.t, r.offs, r.size);
        }
    }
    CHECK(any, "%s: the %lld written cells of stream %lld read zero before the clear", where, (long long) n, (long long) s);
}

static void check_all_zero(const mem_tensors & m, const char * where) {
    for (const auto & x : m.kv) {
        CHECK(tensor_is_zero(x.t), "%s: %s does not read zero after clear(true)", where, x.what.c_str());
    }
}

static const uint8_t SENTINEL = 0xA5;

// put the sentinel into one cell (-1: the last) of every stream of every notepad tensor
static void plant_sentinels(const mem_tensors & m, int64_t cell, uint8_t value) {
    for (const auto & x : m.kv) {
        const int64_t c = cell < 0 ? x.t->ne[1] - 1 : cell;
        for (int64_t s = 0; s < x.t->ne[2]; ++s) {
            for (const auto & r : cell_ranges(x, s, c, c + 1)) {
                std::vector<uint8_t> b(r.size, value);
                ggml_backend_tensor_set(const_cast<ggml_tensor *>(x.t), b.data(), r.offs, r.size);
            }
        }
    }
}

// every byte reads zero except the sentinels, which read `sentinel`
static void check_zero_but_sentinels(const mem_tensors & m, int64_t cell, uint8_t sentinel, const char * where) {
    for (const auto & x : m.kv) {
        std::vector<uint8_t> b(ggml_nbytes(x.t));
        ggml_backend_tensor_get(x.t, b.data(), 0, b.size());
        std::vector<uint8_t> want(b.size(), 0);
        const int64_t c = cell < 0 ? x.t->ne[1] - 1 : cell;
        for (int64_t s = 0; s < x.t->ne[2]; ++s) {
            for (const auto & r : cell_ranges(x, s, c, c + 1)) {
                std::fill(want.begin() + r.offs, want.begin() + r.offs + r.size, sentinel);
            }
        }
        size_t n_bad = 0;
        size_t first = 0;
        for (size_t i = 0; i < b.size(); ++i) {
            if (b[i] != want[i]) {
                first = n_bad == 0 ? i : first;
                n_bad++;
            }
        }
        CHECK(n_bad == 0, "%s: %s has %zu bytes not as expected after clear(true), first at %zu (0x%02x, want 0x%02x)",
                where, x.what.c_str(), n_bad, first, n_bad ? b[first] : 0, n_bad ? want[first] : 0);
    }
}

static void test_clear(const test_cfg & cfg) {
    llama_context * ctx = open_ctx(cfg, true);
    const mem_tensors m = collect(ctx, cfg.v_trans);
    llama_memory_t mem = llama_get_memory(ctx);

    if (!decode_seq(ctx, 0, 0, 300, 1) || !decode_seq(ctx, 1, 0, 700, 2)) {
        n_fail++;
        llama_free(ctx);
        return;
    }
    check_written(m, 0, 300, "clear");
    check_written(m, 1, 700, "clear");

    // nothing writes the last cell; with the lazy path the clear must not reach it
    plant_sentinels(m, -1, SENTINEL);

    llama_memory_clear(mem, true);

    check_zero_but_sentinels(m, -1, cfg.expect_lazy ? SENTINEL : 0, "clear");
    plant_sentinels(m, -1, 0);

    // again, after the high-water marks were reset: a short write, a clear
    if (!decode_seq(ctx, 1, 0, 40, 3)) {
        n_fail++;
        llama_free(ctx);
        return;
    }
    check_written(m, 1, 40, "clear/2");

    // cell 512 is past this round's padded writes but inside the first round's (700 -> 768): a clear that kept the old mark would zero it
    plant_sentinels(m, 512, SENTINEL);
    llama_memory_clear(mem, true);
    check_zero_but_sentinels(m, 512, cfg.expect_lazy ? SENTINEL : 0, "clear/2");
    plant_sentinels(m, 512, 0);

    llama_free(ctx);
}

static void test_copy(const test_cfg & cfg) {
    llama_context * ctx = open_ctx(cfg, true);
    const mem_tensors m = collect(ctx, cfg.v_trans);
    llama_memory_t mem = llama_get_memory(ctx);

    if (!decode_seq(ctx, 1, 0, 600, 4)) {
        n_fail++;
        llama_free(ctx);
        return;
    }

    // a whole-sequence copy across streams is applied as a stream copy in the next update
    llama_memory_seq_cp(mem, 1, 0, -1, -1);
    if (!decode_seq(ctx, 1, 600, 1, 5)) {
        n_fail++;
        llama_free(ctx);
        return;
    }
    check_written(m, 0, 600, "copy");

    llama_memory_clear(mem, true);
    check_all_zero(m, "copy");

    llama_free(ctx);
}

static void test_restore(const test_cfg & cfg) {
    llama_context * ctx = open_ctx(cfg, true);
    const mem_tensors m = collect(ctx, cfg.v_trans);
    llama_memory_t mem = llama_get_memory(ctx);

    if (!decode_seq(ctx, 1, 0, 500, 9)) {
        n_fail++;
        llama_free(ctx);
        return;
    }

    std::vector<uint8_t> state(llama_state_get_size(ctx));
    const size_t n_state = llama_state_get_data(ctx, state.data(), state.size());
    CHECK(n_state > 0, "restore: the state did not save");

    llama_memory_clear(mem, true);
    CHECK(llama_state_set_data(ctx, state.data(), n_state) == n_state, "restore: the state did not load");
    check_written(m, 1, 500, "restore");

    llama_memory_clear(mem, true);
    check_all_zero(m, "restore");

    llama_free(ctx);
}

struct snapshot {
    std::vector<std::vector<uint8_t>> bytes;
    std::vector<float>                logits;
};

static snapshot run_identity(const test_cfg & cfg, bool lazy) {
    llama_context * ctx = open_ctx(cfg, lazy);
    const mem_tensors m = collect(ctx, cfg.v_trans);
    llama_memory_t mem = llama_get_memory(ctx);

    snapshot res;
    bool ok = decode_seq(ctx, 0, 0, 300, 6);
    llama_memory_clear(mem, true);
    ok = ok && decode_seq(ctx, 1, 0, 200, 7);
    ok = ok && decode_seq(ctx, 0, 0, 100, 8);
    if (!ok) {
        n_fail++;
        llama_free(ctx);
        return res;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(cfg.model));
    const float * lg = llama_get_logits_ith(ctx, -1);
    res.logits.assign(lg, lg + n_vocab);

    for (const auto & x : m.kv) {
        std::vector<uint8_t> b(ggml_nbytes(x.t));
        ggml_backend_tensor_get(x.t, b.data(), 0, b.size());
        res.bytes.push_back(std::move(b));
    }
    for (const auto * t : m.keep) {
        std::vector<uint8_t> b(ggml_nbytes(t));
        ggml_backend_tensor_get(t, b.data(), 0, b.size());
        res.bytes.push_back(std::move(b));
    }

    llama_free(ctx);
    return res;
}

static void test_identity(const test_cfg & cfg) {
    const snapshot off  = run_identity(cfg, false);
    const snapshot lazy = run_identity(cfg, true);

    CHECK(off.bytes.size() == lazy.bytes.size() && !off.bytes.empty(), "identity: tensor counts differ (%zu vs %zu)",
            off.bytes.size(), lazy.bytes.size());
    for (size_t i = 0; i < std::min(off.bytes.size(), lazy.bytes.size()); ++i) {
        CHECK(off.bytes[i] == lazy.bytes[i], "identity: memory tensor %zu differs between lazy off and on", i);
    }
    CHECK(!off.logits.empty() && off.logits.size() == lazy.logits.size() &&
          memcmp(off.logits.data(), lazy.logits.data(), off.logits.size()*sizeof(float)) == 0,
          "identity: logits differ between lazy off and on");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 65536;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
        fprintf(stderr, "pass -fa on or -fa off: the test reads the V layout from it\n");
        return 1;
    }

    const char * expect = getenv("KV_LAZY_EXPECT");
    if (expect == nullptr || (strcmp(expect, "0") != 0 && strcmp(expect, "1") != 0)) {
        fprintf(stderr, "set KV_LAZY_EXPECT=1 (the lazy path applies, Metal shared buffers) or 0 (it falls back)\n");
        return 1;
    }

    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", params.model.path.c_str());
        return 1;
    }

    test_cfg cfg;
    cfg.model       = model;
    cfg.cparams     = common_context_params_to_llama(params);
    cfg.cparams.n_seq_max  = 2;
    cfg.cparams.kv_unified = false;
    cfg.cparams.n_batch    = 1024;
    cfg.cparams.n_ubatch   = 512;
    cfg.v_trans     = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cfg.expect_lazy = strcmp(expect, "1") == 0;

    fprintf(stderr, "test-kv-lazy-zero: n_ctx %u, two streams, flash attention %s, lazy path %s\n",
            cfg.cparams.n_ctx, cfg.v_trans ? "off (V transposed)" : "on", cfg.expect_lazy ? "expected" : "falls back");

    test_open(cfg, false);
    test_open(cfg, true);
    test_clear(cfg);
    test_copy(cfg);
    test_restore(cfg);
    test_identity(cfg);

    llama_model_free(model);

    if (n_fail > 0) {
        fprintf(stderr, "test-kv-lazy-zero: %d check(s) failed\n", n_fail);
        return 1;
    }
    fprintf(stderr, "test-kv-lazy-zero: OK\n");
    return 0;
}

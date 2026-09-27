// The phrasebook trace written behind LLAMA_PLE_TRACE (SHARE-PARTS-PLAN.md phase 2).
//
// The bytes are read back by hand here, not by anything in llama-ple-trace.cpp, so this file is a
// second, independent statement of the format beside ~/dev/ai/sim/ple_trace.py. Two parts:
//   writer  the writer alone: header, records, kinds, the lazy open, the skips, the flush rule
//   model   the tiny qwen4exp fixture (-m), read the default way and with direct reads
//           (--lazy-mode on-direct): the rows in the trace are the rows the n-gram hash gives for the
//           tokens decoded (worked here from the hparams, not by the engine's code), one record per
//           ubatch with the right kind, a context without the switch writes nothing, and the logits
//           are byte-identical with the trace on and off (the hook reads, it never changes a number)

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-model.h"
#include "../src/llama-ple-trace.h"

#include "ggml.h"

#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

static std::string tmp_path(const std::string & name) {
    const char * d = getenv("TMPDIR");
    std::string dir = d && *d ? d : "/tmp";
    if (dir.back() != '/') { dir += '/'; }
    return dir + "ple-trace-" + name + "-" + std::to_string(getpid()) + ".pltr";
}

// a whole file, and a cursor over it
struct bytes {
    std::vector<uint8_t> b;
    size_t at = 0;

    bool load(const std::string & path) {
        b.clear(); at = 0;
        FILE * f = fopen(path.c_str(), "rb");
        if (f == nullptr) { return false; }
        uint8_t buf[4096];
        size_t  n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) { b.insert(b.end(), buf, buf + n); }
        fclose(f);
        return true;
    }
    template <typename T> T get() { T v{}; memcpy(&v, b.data() + at, sizeof(v)); at += sizeof(v); return v; }
    bool left(size_t n) const { return at + n <= b.size(); }
};

struct header { std::string magic; uint32_t version, n_heads, n_gram; uint64_t n_rows; uint32_t row_bytes; uint64_t t0; };
struct record { uint64_t t_us; uint32_t n_tokens; uint8_t kind; std::vector<uint32_t> rows; };

// header, then every whole record; false on a partial record
static bool parse(bytes & f, header & h, std::vector<record> & recs) {
    if (!f.left(36)) { return false; }
    h.magic = std::string((const char *) f.b.data(), 4); f.at = 4;
    h.version = f.get<uint32_t>(); h.n_heads = f.get<uint32_t>(); h.n_gram = f.get<uint32_t>();
    h.n_rows = f.get<uint64_t>(); h.row_bytes = f.get<uint32_t>(); h.t0 = f.get<uint64_t>();
    while (f.at < f.b.size()) {
        if (!f.left(13)) { return false; }
        record r;
        r.t_us = f.get<uint64_t>(); r.n_tokens = f.get<uint32_t>(); r.kind = f.get<uint8_t>();
        const size_t n = (size_t) r.n_tokens*h.n_heads;
        if (!f.left(4*n)) { return false; }
        for (size_t i = 0; i < n; i++) { r.rows.push_back(f.get<uint32_t>()); }
        recs.push_back(std::move(r));
    }
    return true;
}

static void test_writer() {
    // the switch: unset, "" and "0" are off
    unsetenv("LLAMA_PLE_TRACE");
    CHECK(llama_ple_trace::from_env() == nullptr, "unset is off");
    setenv("LLAMA_PLE_TRACE", "", 1);
    CHECK(llama_ple_trace::from_env() == nullptr, "empty is off");
    setenv("LLAMA_PLE_TRACE", "0", 1);
    CHECK(llama_ple_trace::from_env() == nullptr, "0 is off");
    setenv("LLAMA_PLE_TRACE", "/x/y.pltr", 1);
    auto on = llama_ple_trace::from_env();
    CHECK(on != nullptr && on->path() == "/x/y.pltr", "a path is on");
    unsetenv("LLAMA_PLE_TRACE");

    const std::string path = tmp_path("writer");
    remove(path.c_str());
    const uint64_t before = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    {
        llama_ple_trace tr(path);
        const std::vector<int32_t> early = { 1, 2 };
        tr.record(early.data(), 2, false); // nothing described yet
        CHECK(tr.n_skipped() == 1 && tr.n_records() == 0, "a record before the table is described is skipped");
        bytes f;
        CHECK(!f.load(path), "and it creates no file (the apprentice's context never reads a row)");

        tr.describe(2, 3, 5000000000ull, 90);
        const std::vector<int32_t> one   = { 7, 4000000 };           // one token
        const std::vector<int32_t> three = { 1, 2, 3, 4, 3, 4 };     // a batch of three, a repeat kept
        const std::vector<int32_t> part  = { 1, 2, 3 };              // not a whole number of tokens
        const std::vector<int32_t> warm  = { 9, 8 };                 // the warm-up, one token
        tr.record(one.data(), 2, false);
        tr.record(three.data(), 6, false);
        tr.record(part.data(), 3, false);
        tr.record(warm.data(), 2, true);
        tr.record(three.data(), 6, true);
        CHECK(tr.n_records() == 4 && tr.n_skipped() == 2, "4 records, 2 skipped (got %lld, %lld)",
              (long long) tr.n_records(), (long long) tr.n_skipped());
    }
    const uint64_t after = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    bytes f;
    CHECK(f.load(path), "the file exists after the first real record");
    CHECK(f.b.size() == 36 + (13 + 8) + (13 + 24) + (13 + 8) + (13 + 24), "size %zu", f.b.size());
    header h; std::vector<record> recs;
    CHECK(parse(f, h, recs), "whole records only");
    CHECK(h.magic == "PLTR" && h.version == 1 && h.n_heads == 2 && h.n_gram == 3, "header fields");
    CHECK(h.n_rows == 5000000000ull && h.row_bytes == 90, "rows are u64, row bytes u32");
    CHECK(h.t0 >= before && h.t0 <= after, "t0 is the wall clock in microseconds");
    if (recs.size() == 4) {
        CHECK(recs[0].kind == 0 && recs[0].n_tokens == 1 && recs[0].rows == std::vector<uint32_t>({ 7, 4000000 }), "token");
        CHECK(recs[1].kind == 1 && recs[1].n_tokens == 3 && recs[1].rows == std::vector<uint32_t>({ 1, 2, 3, 4, 3, 4 }), "batch");
        CHECK(recs[2].kind == 2 && recs[2].n_tokens == 1, "warm-up of one token is kind 2, not 0");
        CHECK(recs[3].kind == 2 && recs[3].n_tokens == 3, "warm-up of three is kind 2, not 1");
        for (size_t i = 1; i < recs.size(); i++) { CHECK(recs[i].t_us >= recs[i - 1].t_us, "time runs forward"); }
        CHECK(recs[3].t_us < 60ull*1000000, "t_us counts from the file's opening");
    } else {
        CHECK(false, "4 records, got %zu", recs.size());
    }
    remove(path.c_str());

    // a second table stops the trace; what was written stays readable
    {
        llama_ple_trace tr(path);
        tr.describe(1, 2, 100, 18);
        const std::vector<int32_t> a = { 5 };
        tr.record(a.data(), 1, false);
        tr.describe(1, 2, 101, 18);
        tr.record(a.data(), 1, false);
        CHECK(tr.n_records() == 1, "nothing written after a second table");
    }
    CHECK(f.load(path) && f.b.size() == 36 + 13 + 4, "the first record survives the stop");
    remove(path.c_str());

    // the flush rule: nothing reaches the file before the 64th record, all 64 at it
    {
        llama_ple_trace tr(path);
        tr.describe(1, 2, 100, 18);
        const std::vector<int32_t> a = { 5 };
        for (int i = 0; i < 63; i++) { tr.record(a.data(), 1, false); }
        const bool loaded = f.load(path);
        CHECK(loaded && f.b.size() < 36 + 63*17, "63 records are still buffered (%zu on disk)", f.b.size());
        tr.record(a.data(), 1, false);
        CHECK(f.load(path) && f.b.size() == 36 + 64*17, "the 64th record flushes all (%zu on disk)", f.b.size());
    }
    remove(path.c_str());
}

// the rows the n-gram hash gives token i of a stream (the reference's rule, worked from the hparams):
// its own id, then its predecessors, a missing one (before position 0) and all before it reading as EOS
static std::vector<uint32_t> rows_of(const llama_hparams & hp, const std::vector<llama_token> & toks, size_t i) {
    std::vector<uint64_t> ctx(hp.ple_ngram_size);
    for (size_t s = 0; s < ctx.size(); s++) {
        ctx[s] = s <= i ? (uint64_t) toks[i - s] : (uint64_t) hp.ple_eos_token_id;
    }
    std::vector<uint32_t> out;
    for (uint32_t n = 2; n <= hp.ple_ngram_size; n++) {
        uint64_t mixed = ctx[0]*hp.ple_layer_multipliers[0];
        for (uint32_t j = 1; j < n; j++) { mixed ^= ctx[j]*hp.ple_layer_multipliers[j]; }
        for (uint32_t g = 0; g < hp.ple_heads_per_ngram; g++) {
            const uint32_t h = (n - 2)*hp.ple_heads_per_ngram + g;
            out.push_back((uint32_t) (mixed % hp.ple_head_vocab_sizes[h] + hp.ple_head_offsets[h]));
        }
    }
    return out;
}

// one context: a warm-up of 2 tokens, 2 tokens, a batch of 38 in ubatches of 16, then 5 single tokens.
// Returns the FNV-1a hash of every logit; `toks` is the stream after the warm-up
static uint64_t decode_all(llama_context * ctx, const std::vector<llama_token> & toks) {
    uint64_t hash = 1469598103934665603ull;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    auto mix = [&](int i) {
        const float * l = llama_get_logits_ith(ctx, i);
        for (size_t k = 0; l != nullptr && k < n_vocab*sizeof(float); k++) { hash = (hash ^ ((const uint8_t *) l)[k])*1099511628211ull; }
    };

    std::vector<llama_token> warm = { 3, 5 };
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    llama_set_warmup(ctx, true);
    CHECK(llama_decode(ctx, llama_batch_get_one(warm.data(), 2)) == 0, "warm-up decode");
    llama_set_warmup(ctx, false);
#pragma GCC diagnostic pop
    llama_memory_clear(llama_get_memory(ctx), true);

    // two tokens first, in the warm-up's exact shape (one output), so the context reuses the warm-up's
    // graph: the reused graph must not keep the warm-up's kind
    std::vector<llama_token> first(toks.begin(), toks.begin() + 2);
    CHECK(llama_decode(ctx, llama_batch_get_one(first.data(), 2)) == 0, "first decode");
    mix(-1);
    llama_batch batch = llama_batch_init(40, 0, 1);
    for (int p = 2; p < 40; p++) { common_batch_add(batch, toks[p], p, { 0 }, true); }
    CHECK(llama_decode(ctx, batch) == 0, "batch decode");
    for (int i = 0; i < 38; i++) { mix(i); }
    for (int p = 40; p < 45; p++) {
        common_batch_clear(batch);
        common_batch_add(batch, toks[p], p, { 0 }, true);
        CHECK(llama_decode(ctx, batch) == 0, "token decode");
        mix(0);
    }
    llama_batch_free(batch);
    return hash;
}

static void test_model(common_params & params, bool direct) {
    const char * name = direct ? "model-direct" : "model";
    auto mparams = common_model_params_to_llama(params);
    mparams.lazy_mode = direct ? LLAMA_LAZY_MODE_DIRECT : LLAMA_LAZY_MODE_AUTO;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) { CHECK(false, "%s: the model did not load", name); return; }
    const llama_hparams & hp = model->hparams;
    CHECK(hp.ple_n_heads > 0 && model->per_layer_tok_embd != nullptr, "%s: the fixture has a phrasebook", name);
    CHECK(direct == (model->lazy_reader(model->per_layer_tok_embd) != nullptr), "%s: the rows are read the way asked", name);

    // a token stream with no EOS (the fixture's EOS is 0), so only position 0's predecessors are missing
    std::vector<llama_token> toks;
    for (int p = 0; p < 45; p++) { toks.push_back(1 + (7*p + 3) % 90); }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = 256; cparams.n_batch = 64; cparams.n_ubatch = 16; cparams.n_seq_max = 1;

    const std::string path = tmp_path(name);
    const std::string path_off = tmp_path(std::string(name) + "-off");
    remove(path.c_str()); remove(path_off.c_str());

    uint64_t hash_on = 0, hash_off = 1;
    {
        setenv("LLAMA_PLE_TRACE", path.c_str(), 1);
        llama_context * ctx = llama_init_from_model(model, cparams);
        unsetenv("LLAMA_PLE_TRACE"); // read when the memory is made: this context keeps it
        CHECK(ctx != nullptr, "%s: context", name);
        if (ctx) { hash_on = decode_all(ctx, toks); llama_free(ctx); } // closes the trace
    }
    {
        setenv("LLAMA_PLE_TRACE", "0", 1);
        llama_context * ctx = llama_init_from_model(model, cparams);
        unsetenv("LLAMA_PLE_TRACE");
        if (ctx) { hash_off = decode_all(ctx, toks); llama_free(ctx); }
    }
    CHECK(hash_on == hash_off, "%s: the logits moved with the trace on (%016llx vs %016llx)", name,
          (unsigned long long) hash_on, (unsigned long long) hash_off);
    bytes none;
    CHECK(!none.load(path_off), "%s: the context without the switch wrote nothing", name);

    bytes f; header h; std::vector<record> recs;
    if (!f.load(path)) { CHECK(false, "%s: no trace written", name); llama_model_free(model); return; }
    CHECK(parse(f, h, recs), "%s: whole records", name);
    CHECK(h.magic == "PLTR" && h.n_heads == hp.ple_n_heads && h.n_gram == hp.ple_ngram_size, "%s: header", name);
    CHECK(h.n_rows == (uint64_t) model->per_layer_tok_embd->ne[1], "%s: n_rows is the table's", name);
    CHECK(h.row_bytes == ggml_row_size(model->per_layer_tok_embd->type, model->per_layer_tok_embd->ne[0]),
          "%s: row bytes are the table's", name);

    // warm-up (2), 2, a batch of 38 as 16 + 16 + 6, then 5 tokens
    const std::vector<std::pair<uint8_t, uint32_t>> want = { {2, 2}, {1, 2}, {1, 16}, {1, 16}, {1, 6}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1} };
    CHECK(recs.size() == want.size(), "%s: %zu records, want %zu", name, recs.size(), want.size());
    size_t pos = 0, bad = 0;
    for (size_t k = 0; k < recs.size() && k < want.size(); k++) {
        CHECK(recs[k].kind == want[k].first && recs[k].n_tokens == want[k].second, "%s: record %zu is kind %d of %u tokens",
              name, k, recs[k].kind, recs[k].n_tokens);
        if (k == 0) { continue; }
        for (uint32_t t = 0; t < recs[k].n_tokens; t++, pos++) {
            const auto expect = rows_of(hp, toks, pos);
            const std::vector<uint32_t> got(recs[k].rows.begin() + t*h.n_heads, recs[k].rows.begin() + (t + 1)*h.n_heads);
            bad += got != expect;
            for (uint32_t r : got) { CHECK(r < h.n_rows, "%s: row %u is past the table", name, r); }
        }
    }
    CHECK(pos == 45 && bad == 0, "%s: %zu of %zu tokens' rows differ from the n-gram hash", name, bad, pos);
    fprintf(stderr, "%s: %zu records, %zu tokens after the warm-up, %zu bytes, rows as the hash gives them%s\n",
            name, recs.size(), pos, f.b.size(), bad == 0 ? "" : " NOT");
    remove(path.c_str());
    llama_model_free(model);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    test_writer();
    fprintf(stderr, "writer: %s\n", n_fail == 0 ? "ok" : "FAILED");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (!params.model.path.empty()) {
        ggml_backend_load_all();
        test_model(params, false);
        test_model(params, true);
    }

    fprintf(stderr, "%s: %s\n", __func__, n_fail == 0 ? "ok" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}

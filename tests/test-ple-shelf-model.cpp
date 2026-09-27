// The phrasebook shelf (--ple-shelf, src/llama-ple-shelf-io.h) on the tiny qwen4exp fixture: the rows the
// shelf hands the graph are the rows the table holds, so the phrasebook's numbers (the "ple_embd" tensor)
// and every logit are byte-identical with the shelf and without it.
//
// The lazy mode is set explicitly (the fixture's table is far under auto's 4 GiB, so auto keeps it
// resident): the references read the rows through the mmap (-lzm on) and with the direct reader
// (on-direct); the shelf is tried with both modes (it replaces either), at 1 MiB (which holds the fixture's
// whole table after a first read) and as a 5-slot shelf far smaller than one batch, so every batch evicts
// and passes rows. Each shelf run must show that the shelf served the rows: batches counted, started ahead
// at the top of set_inputs (the phrasebook input's set_input_begin), hits where the shelf is big enough,
// passes where it is not. A stream of tokens is read in batches of 16, then written a token at a time, on
// two contexts in turn (the second reuses the shelf the first filled).

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-lazy-reader.h"
#include "../src/llama-model.h"
#include "../src/llama-ple-shelf-io.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

struct run_hash {
    uint64_t ple    = 1469598103934665603ull; // every captured ple_embd byte
    uint64_t logits = 1469598103934665603ull;
    int64_t  n_ple  = 0;                      // ple_embd tensors seen
    int64_t  nonzero = 0;                     // nonzero ple_embd numbers (the rows are not all zero)
};

static void fnv(uint64_t & h, const void * p, size_t n) {
    for (size_t k = 0; k < n; k++) { h = (h ^ ((const uint8_t *) p)[k])*1099511628211ull; }
}

static bool observe(ggml_tensor * t, bool ask, void * ud) {
    if (strcmp(t->name, "ple_embd") != 0) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    auto * h = (run_hash *) ud;
    std::vector<float> v(ggml_nelements(t));
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    fnv(h->ple, v.data(), ggml_nbytes(t));
    for (float x : v) { h->nonzero += x != 0.0f; }
    h->n_ple++;
    return true;
}

// a stream with no EOS (the fixture's EOS is 0): 64 tokens read in 16 at a time, then 8 written one by one
static void decode_all(llama_context * ctx, run_hash & h) {
    std::vector<llama_token> toks;
    for (int p = 0; p < 72; p++) { toks.push_back(1 + (11*p + 5) % 97); }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    llama_batch batch = llama_batch_init(64, 0, 1);
    for (int p = 0; p < 64; p++) { common_batch_add(batch, toks[p], p, { 0 }, true); }
    CHECK(llama_decode(ctx, batch) == 0, "read-in decode");
    for (int i = 0; i < 64; i++) { fnv(h.logits, llama_get_logits_ith(ctx, i), n_vocab*sizeof(float)); }
    for (int p = 64; p < 72; p++) {
        common_batch_clear(batch);
        common_batch_add(batch, toks[p], p, { 0 }, true);
        CHECK(llama_decode(ctx, batch) == 0, "token decode");
        fnv(h.logits, llama_get_logits_ith(ctx, 0), n_vocab*sizeof(float));
    }
    llama_batch_free(batch);
}

enum shelf_kind { SHELF_NONE, SHELF_1MIB, SHELF_TINY };

static run_hash run(common_params & params, llama_lazy_mode mode, shelf_kind shelf, const char * name) {
    run_hash h;
    auto mparams = common_model_params_to_llama(params);
    mparams.lazy_mode     = mode;
    mparams.ple_shelf_mib = shelf == SHELF_NONE ? 0 : 1;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) { CHECK(false, "%s: the model did not load", name); return h; }
    const ggml_tensor * table = model->per_layer_tok_embd;
    CHECK(model->hparams.ple_n_heads > 0 && table != nullptr, "%s: the fixture has a phrasebook", name);
    if (table == nullptr) { llama_model_free(model); return h; }

    if (shelf == SHELF_TINY) {
        // the same shelf as --ple-shelf builds, with 5 slots: far fewer than one batch's rows
        auto it = model->lazy_readers.find(table);
        CHECK(it != model->lazy_readers.end() && it->second->shelf, "%s: a shelf to shrink", name);
        if (it != model->lazy_readers.end()) {
            llama_ple_shelf_config cfg = it->second->shelf->config();
            cfg.slots = 5;
            it->second = std::make_unique<llama_lazy_reader>(llama_ple_shelf_make(cfg));
        }
    }

    const llama_lazy_reader * reader = model->lazy_reader(table);
    const bool want_reader = shelf != SHELF_NONE || mode == LLAMA_LAZY_MODE_DIRECT;
    CHECK(want_reader == (reader != nullptr), "%s: the rows are read the way asked (reader %p)", name, (const void *) reader);
    CHECK((shelf != SHELF_NONE) == (reader != nullptr && reader->shelf != nullptr), "%s: the shelf is %s", name,
            shelf != SHELF_NONE ? "on" : "off");

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = 256; cparams.n_batch = 64; cparams.n_ubatch = 16; cparams.n_seq_max = 1;
    cparams.cb_eval = observe; cparams.cb_eval_user_data = &h;
    for (int c = 0; c < 2; c++) {
        llama_context * ctx = llama_init_from_model(model, cparams);
        CHECK(ctx != nullptr, "%s: context %d", name, c);
        if (ctx) { decode_all(ctx, h); llama_free(ctx); }
    }

    if (reader && reader->shelf) {
        const llama_ple_shelf_stats s = reader->shelf->stats();
        const int64_t asks = s.asks[0] + s.asks[1], hits = s.hits[0] + s.hits[1];
        // 2 contexts x (4 batches of 16 + 8 tokens), every one started ahead
        CHECK(s.calls == 24 && s.begun == 24, "%s: the shelf served %lld batches, %lld started ahead", name,
                (long long) s.calls, (long long) s.begun);
        CHECK(asks > 0 && s.asks[0] > 0 && s.asks[1] > 0, "%s: asks writing %lld, reading in %lld", name,
                (long long) s.asks[0], (long long) s.asks[1]);
        if (shelf == SHELF_TINY) {
            CHECK(s.passed > 0 && reader->shelf->held() == 5, "%s: a 5-slot shelf passes rows (%lld) and stays full",
                    name, (long long) s.passed);
        } else {
            CHECK(hits > 0 && s.passed == 0, "%s: %lld of %lld asks on the shelf, %lld passed", name, (long long) hits,
                    (long long) asks, (long long) s.passed);
        }
        printf("%s: %lld batches, %lld asks, %lld on the shelf, %lld passed, %lld rows read\n", name, (long long) s.calls,
                (long long) asks, (long long) hits, (long long) s.passed, (long long) (s.reads[0] + s.reads[1]));
    }
    llama_model_free(model);
    return h;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();

    const run_hash mmap_ref   = run(params, LLAMA_LAZY_MODE_ON,     SHELF_NONE, "mmap");
    const run_hash direct_ref = run(params, LLAMA_LAZY_MODE_DIRECT, SHELF_NONE, "direct");
    const run_hash on_shelf   = run(params, LLAMA_LAZY_MODE_ON,     SHELF_1MIB, "mmap mode + shelf");
    const run_hash dir_shelf  = run(params, LLAMA_LAZY_MODE_DIRECT, SHELF_1MIB, "direct mode + shelf");
    const run_hash tiny       = run(params, LLAMA_LAZY_MODE_ON,     SHELF_TINY, "5-slot shelf");

    CHECK(mmap_ref.n_ple == 24 && mmap_ref.nonzero > 0, "the reference captured %lld ple_embd tensors", (long long) mmap_ref.n_ple);
    CHECK(direct_ref.ple == mmap_ref.ple && direct_ref.logits == mmap_ref.logits, "the two references agree");
    for (const auto & [name, h] : std::vector<std::pair<const char *, run_hash>>{
            { "mmap mode + shelf", on_shelf }, { "direct mode + shelf", dir_shelf }, { "5-slot shelf", tiny } }) {
        CHECK(h.n_ple == mmap_ref.n_ple && h.ple == mmap_ref.ple, "%s: ple_embd differs from the mmap read", name);
        CHECK(h.logits == mmap_ref.logits, "%s: logits differ from the mmap read", name);
    }
    printf("test-ple-shelf-model: ple %016llx logits %016llx\n", (unsigned long long) mmap_ref.ple, (unsigned long long) mmap_ref.logits);

    llama_backend_free();
    if (n_fail) {
        fprintf(stderr, "test-ple-shelf-model: %d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-ple-shelf-model: all checks passed\n");
    return 0;
}

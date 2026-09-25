// Expert streaming (--moe-stream) on the 64-expert fixtures, through each of its three graph paths:
//
//   - one wave (decode, or a ubatch whose books all fit on the desk): llama_moe_stream_remap
//   - masked waves (a ubatch that touches more books than the desk holds, too few pairs to partition)
//   - partitioned waves (the same, with >= 600 (token, expert) pairs per wave)
//
// The run reads in -b random tokens in ubatches of -ub, then writes -n more greedily, and checks the
// streamed run against the same model with every book resident (no --moe-stream). The tiny fixture
// cannot fill a real desk, so a small one (--moe-stream-cache 40s, 40 slots of 64) is what forces the
// wave paths; --expect names the path the ubatch must have taken, from the graph's own
// "n_tokens = ... partition ON/OFF" line, so a test that silently fell back cannot pass.
//
// The fixture's weights are N(0, 0.01), so the experts add less than a float's resolution to the
// residual stream and the logits would not move even if every book came from the wrong slot. Each
// floor's MoE output (ffn_moe_out) is therefore captured with the eval callback and compared too.
//
// Also the old/new gate tool for the book manager split (patch R): --no-ref skips the reference
// model so a GGML_SCHED_DEBUG=2 listing holds the streamed graphs only, and --dump FILE writes the
// streamed logits and then the captured MoE outputs as raw floats, for a byte comparison of two builds.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// which wave path the ubatch-sized graph reported, from the graph builder's own log line
static bool g_saw_partition = false;
static bool g_saw_masked    = false;

static void log_cb(ggml_log_level level, const char * text, void * /*user_data*/) {
    (void) level;
    if (strstr(text, "moe stream: n_tokens =") != nullptr) {
        g_saw_partition = g_saw_partition || strstr(text, "partition ON")  != nullptr;
        g_saw_masked    = g_saw_masked    || strstr(text, "partition OFF") != nullptr;
    }
    fputs(text, stderr);
}

struct run_result {
    std::vector<float>       logits; // every prompt position, then one row per written token
    std::vector<float>       moe;    // every ffn_moe_out the graphs computed, in compute order
    std::vector<llama_token> written;
};

static bool capture_moe_out(ggml_tensor * t, bool ask, void * user_data) {
    const bool want = strncmp(t->name, "ffn_moe_out-", strlen("ffn_moe_out-")) == 0;
    if (ask || !want) {
        return want;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    auto * moe = (std::vector<float> *) user_data;
    const size_t n0 = moe->size();
    moe->resize(n0 + ggml_nelements(t));
    ggml_backend_tensor_get(t, moe->data() + n0, 0, ggml_nbytes(t));
    return true;
}

static bool run(llama_model * model, const common_params & params, const std::vector<llama_token> & prompt,
        run_result & out) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_ctx             = (uint32_t) prompt.size() + (uint32_t) params.n_predict + 1;
    cparams.n_batch           = (uint32_t) prompt.size();
    cparams.cb_eval           = capture_moe_out;
    cparams.cb_eval_user_data = &out.moe;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create the context\n");
        return false;
    }

    llama_batch batch = llama_batch_init((int32_t) prompt.size(), 0, 1);
    for (size_t i = 0; i < prompt.size(); i++) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, true);
    }

    bool ok = llama_decode(ctx, batch) == 0;
    if (ok) {
        const float * l = llama_get_logits(ctx);
        out.logits.assign(l, l + prompt.size()*n_vocab);
    }

    // greedy writing, one token at a time: the single-wave remap path
    for (int i = 0; ok && i < params.n_predict; i++) {
        const float * row = out.logits.data() + out.logits.size() - n_vocab;
        llama_token best = 0;
        for (int v = 1; v < n_vocab; v++) {
            if (row[v] > row[best]) {
                best = v;
            }
        }
        out.written.push_back(best);

        common_batch_clear(batch);
        common_batch_add(batch, best, (llama_pos) (prompt.size() + i), { 0 }, true);
        ok = llama_decode(ctx, batch) == 0;
        if (ok) {
            const float * l = llama_get_logits_ith(ctx, -1);
            out.logits.insert(out.logits.end(), l, l + n_vocab);
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    if (!ok) {
        fprintf(stderr, "llama_decode failed\n");
    }
    return ok;
}

// normalized mean squared error = mse(a, b) / mse(a, 0), as test-llama-archs measures a backend; an
// all-zero reference is an error of 1, so a capture that caught nothing cannot pass
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double ab = 0.0, a0 = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        ab += (double) (a[i] - b[i])*(a[i] - b[i]);
        a0 += (double) a[i]*a[i];
    }
    return a0 > 0.0 ? ab/a0 : 1.0;
}

static uint64_t fnv1a64(const std::vector<float> & v, uint64_t h = 1469598103934665603ull) {
    const auto * p = (const uint8_t *) v.data();
    for (size_t i = 0; i < v.size()*sizeof(float); i++) {
        h = (h ^ p[i])*1099511628211ull;
    }
    return h;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // this test's own flags, taken out before the common parser sees the rest
    std::string dump_path;
    std::string expect;
    bool        with_ref = true;
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_path = argv[++i];
        } else if (strcmp(argv[i], "--expect") == 0 && i + 1 < argc) {
            expect = argv[++i];
        } else if (strcmp(argv[i], "--no-ref") == 0) {
            with_ref = false;
        } else {
            rest.push_back(argv[i]);
        }
    }
    if (!expect.empty() && expect != "masked" && expect != "partition") {
        fprintf(stderr, "--expect takes masked or partition, got '%s'\n", expect.c_str());
        return 1;
    }

    common_params params;
    params.n_predict = 8;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (!params.moe_stream) {
        fprintf(stderr, "needs --moe-stream (e.g. --moe-stream-cache 40s)\n");
        return 1;
    }

    llama_log_set(log_cb, nullptr);
    ggml_backend_load_all();

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }

    // a fixed stream of random ids: the fixture has no vocabulary to tokenize a text prompt with
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::mt19937 rng(1234);
    std::vector<llama_token> prompt(params.n_batch);
    for (auto & t : prompt) {
        t = (llama_token) (rng() % (uint32_t) n_vocab);
    }

    run_result streamed;
    bool ok = run(model, params, prompt, streamed);
    llama_model_free(model);

    if (ok && !expect.empty()) {
        const bool saw = expect == "partition" ? g_saw_partition : g_saw_masked;
        printf("path: expected %s, graph reported%s%s\n", expect.c_str(),
                g_saw_partition ? " partition" : "", g_saw_masked ? " masked" : "");
        ok = saw;
    }

    printf("streamed: %zu logits fnv1a64 %016llx, %zu moe outputs fnv1a64 %016llx, written:",
            streamed.logits.size(), (unsigned long long) fnv1a64(streamed.logits),
            streamed.moe.size(),    (unsigned long long) fnv1a64(streamed.moe));
    for (const llama_token t : streamed.written) {
        printf(" %d", t);
    }
    printf("\n");

    if (ok && !dump_path.empty()) {
        FILE * f = fopen(dump_path.c_str(), "wb");
        ok = f != nullptr &&
             fwrite(streamed.logits.data(), sizeof(float), streamed.logits.size(), f) == streamed.logits.size() &&
             fwrite(streamed.moe.data(),    sizeof(float), streamed.moe.size(),    f) == streamed.moe.size();
        if (f != nullptr) {
            fclose(f);
        }
    }

    if (ok && with_ref) {
        mparams.moe_stream = false;
        llama_model * ref_model = llama_model_load_from_file(params.model.path.c_str(), mparams);
        run_result ref;
        ok = ref_model != nullptr && run(ref_model, params, prompt, ref);
        if (ref_model != nullptr) {
            llama_model_free(ref_model);
        }
        ok = ok && !ref.moe.empty() && ref.logits.size() == streamed.logits.size() && ref.moe.size() == streamed.moe.size();
        if (ok) {
            // streaming changes which slot a book is read from, never which books the router
            // picked, so every floor's MoE output agrees with the all-resident run
            const double err_logits = nmse(ref.logits, streamed.logits);
            const double err_moe    = nmse(ref.moe,    streamed.moe);
            printf("reference: nmse logits %.3e, moe outputs %.3e, written %s\n", err_logits, err_moe,
                    ref.written == streamed.written ? "same" : "DIFFERENT");
            ok = err_logits <= 1e-6 && err_moe <= 1e-6 && ref.written == streamed.written;
        }
    }

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}

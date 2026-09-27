// LLAMA_QSA_UNION=1 (src/models/qwen4exp-qsa-union.h): the picks go straight to attention. Two contexts on
// one model, the masked path (LLAMA_QSA_UNION=0) and union attention (=1), both slicing 128 rows at a time,
// decode the same batches: 512 and 188 tokens (sliced), then 100 (one slice for union, the whole batch for
// the masked path), then four single tokens (neither route: the small host inputs, the per-cell picker).
//   picks     every row's picked blocks (indexer_top_blk) byte-identical: the bias is built the same way
//   visible   the visible-set oracle, exact: for every sliced row, the cells union attention receives (the
//             ids >= 0 of qsa_sel) are exactly the cells the masked path leaves finite in its mask (qsa_mask);
//             and the filter must have dropped cells after the token somewhere, or it proved nothing
//   outputs   the attention output before the gate (attn_pregate) and the logits agree within a tolerance:
//             the same cells and weights, summed in another order
//   ran       union floors were built in the union context (after its reserve), none in the masked one, and
//             no single-token graph took the route
// The head-256 fixture (-h256: heads of 256, two q heads on one kv head) is the shape the Metal union
// kernel takes; on the default fixture's heads of 128 the op runs on the CPU backend.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-memory-hybrid-idx.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml.h"

static int n_fail = 0;

#define FAIL(...) do { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } while (0)

struct capture {
    ggml_type            type;
    int64_t              ne0, ne1;
    std::vector<uint8_t> data;
};

// every capture of one decode by name: "<base>-<il>" or "<base>-<il>-s<i>"
using observer = std::map<std::string, capture>;

static bool observed(const char * name) {
    for (const char * p : { "indexer_top_blk-", "qsa_mask-", "qsa_sel-", "attn_pregate-" }) {
        const size_t n = strlen(p);
        if (strncmp(name, p, n) != 0) {
            continue;
        }
        const char * rest = name + n;
        const size_t digits = strspn(rest, "0123456789");
        return digits > 0 && (rest[digits] == '\0' ||
               (rest[digits] == '-' && rest[digits + 1] == 's' && strspn(rest + digits + 2, "0123456789") == strlen(rest + digits + 2)));
    }
    return false;
}

static bool observe(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return observed(t->name);
    }
    GGML_ASSERT(ggml_is_contiguous(t));
    capture c { t->type, t->ne[0], ggml_nelements(t)/t->ne[0], std::vector<uint8_t>(ggml_nbytes(t)) };
    ggml_backend_tensor_get(t, c.data.data(), 0, ggml_nbytes(t));
    (*(observer *) user_data)[t->name] = std::move(c);
    return true;
}

// the "qsa union:" startup line the context made last printed (after "qsa union: "): it must name the
// route the context takes, not the switch's value
static std::string g_union_line;

static void capture_log(ggml_log_level level, const char * text, void * user_data) {
    if (const char * p = strstr(text, "qsa union: ")) {
        g_union_line = p + strlen("qsa union: ");
        while (!g_union_line.empty() && g_union_line.back() == '\n') {
            g_union_line.pop_back();
        }
    }
    fputs(text, stderr);
    (void) level; (void) user_data;
}

struct run {
    llama_context *      ctx = nullptr;
    observer             obs;
    std::vector<float>   logits;
    llama_qsa_union_mode mode;
    uint64_t             attn0 = 0; // union floors built by the reserve
    std::string          line;      // its startup line

    run(const common_params & params, llama_model * model, llama_qsa_union_mode mode, bool kv_f16 = true) : mode(mode) {
        setenv("LLAMA_QSA_UNION", mode == LLAMA_QSA_UNION_ATTN ? "1" : "0", 1);
        setenv("LLAMA_QSA_SLICE", "128", 1);
        auto cparams = common_context_params_to_llama(params);
        cparams.n_ctx = 1024; cparams.n_batch = 1024; cparams.n_ubatch = 512; cparams.n_seq_max = 1;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // union attention needs it
        // ... and an F16 cache: with a quantised one "1" builds "bias"
        cparams.type_k = cparams.type_v = kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
        cparams.cb_eval = observe; cparams.cb_eval_user_data = &obs;
        g_union_line.clear();
        ctx = llama_init_from_model(model, cparams);
        line = g_union_line;
        unsetenv("LLAMA_QSA_UNION");
        unsetenv("LLAMA_QSA_SLICE");
        if (ctx == nullptr || mem()->qsa_union() != mode) {
            FAIL("LLAMA_QSA_UNION not picked up per context (asked %d)", (int) mode);
            return;
        }
        attn0 = mem()->qsa_union_attn_builds();
    }
    ~run() { llama_free(ctx); }

    llama_memory_hybrid_idx * mem() const { return dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx)); }

    bool decode(llama_pos p0, llama_pos p1) {
        obs.clear();
        logits.clear();
        llama_batch batch = llama_batch_init(p1 - p0, 0, 1);
        for (llama_pos p = p0; p < p1; ++p) {
            common_batch_add(batch, (llama_token) ((7*p + 1) % 97), p, { 0 }, true);
        }
        const bool ok = llama_decode(ctx, batch) == 0;
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        for (int i = 0; ok && i < batch.n_tokens; ++i) {
            const float * l = llama_get_logits_ith(ctx, i);
            logits.insert(logits.end(), l, l + n_vocab);
        }
        llama_batch_free(batch);
        return ok;
    }
};

// the captures of one layer named <base>-<il>[-s<i>], joined in slice order: {il: bytes}
static std::map<int, std::vector<uint8_t>> joined(const observer & obs, const std::string & base) {
    std::map<int, std::map<int, const capture *>> parts;
    for (const auto & [name, c] : obs) {
        if (name.rfind(base + "-", 0) != 0) {
            continue;
        }
        const char * rest = name.c_str() + base.size() + 1;
        const char * s    = strstr(rest, "-s");
        parts[atoi(rest)][s ? atoi(s + 2) : 0] = &c;
    }
    std::map<int, std::vector<uint8_t>> out;
    for (const auto & [il, slices] : parts) {
        for (const auto & [i, c] : slices) {
            out[il].insert(out[il].end(), c->data.begin(), c->data.end());
        }
    }
    return out;
}

// sum (a - b)^2 / sum a^2 over two float arrays of one size
static double nmse(const float * a, const float * b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t k = 0; k < n; ++k) {
        num += ((double) a[k] - b[k])*((double) a[k] - b[k]);
        den += (double) a[k]*a[k];
    }
    return den > 0.0 ? num/den : num;
}

// the oracle: every sliced row's union selection against the masked path's mask; counts rows and dropped ids
static void check_visible(const observer & ref, const observer & uni, const char * what, int64_t & n_rows, int64_t & n_dropped) {
    for (const auto & [name, sel] : uni) {
        if (name.rfind("qsa_sel-", 0) != 0) {
            continue;
        }
        const std::string mname = "qsa_mask-" + name.substr(strlen("qsa_sel-"));
        const auto it = ref.find(mname);
        if (it == ref.end()) {
            FAIL("%s: %s has no %s to hold it to", what, name.c_str(), mname.c_str());
            continue;
        }
        const capture & mask = it->second;
        if (sel.type != GGML_TYPE_I32 || sel.ne1 != mask.ne1 || (mask.type != GGML_TYPE_F16 && mask.type != GGML_TYPE_F32)) {
            FAIL("%s: %s and %s do not line up", what, name.c_str(), mname.c_str());
            continue;
        }
        for (int64_t row = 0; row < sel.ne1; ++row) {
            std::set<int32_t> want, got;
            for (int64_t c = 0; c < mask.ne0; ++c) {
                const float v = mask.type == GGML_TYPE_F16
                    ? ggml_fp16_to_fp32(((const ggml_fp16_t *) mask.data.data())[row*mask.ne0 + c])
                    : ((const float *) mask.data.data())[row*mask.ne0 + c];
                if (std::isfinite(v)) {
                    want.insert((int32_t) c);
                }
            }
            for (int64_t j = 0; j < sel.ne0; ++j) {
                const int32_t id = ((const int32_t *) sel.data.data())[row*sel.ne0 + j];
                if (id >= 0) {
                    got.insert(id);
                } else {
                    n_dropped++;
                }
            }
            if (want != got) {
                FAIL("%s: %s row %lld sees %zu cells, the masked path %zu", what, name.c_str(), (long long) row, got.size(), want.size());
                return;
            }
            n_rows++;
        }
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();
    llama_log_set(capture_log, nullptr);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (model == nullptr) {
        fprintf(stderr, "test-qsa-union-model: cannot load %s\n", params.model.path.c_str());
        return 1;
    }

    {
        run ref(params, model, LLAMA_QSA_UNION_OFF);
        run uni(params, model, LLAMA_QSA_UNION_ATTN);

        int64_t n_rows = 0, n_dropped = 0;
        double  worst_attn = 0.0, worst_logits = 0.0;

        const std::vector<std::pair<llama_pos, llama_pos>> batches = { { 0, 512 }, { 512, 700 }, { 700, 800 }, { 800, 801 }, { 801, 802 }, { 802, 803 }, { 803, 804 } };
        for (const auto & [p0, p1] : batches) {
            char what[64];
            snprintf(what, sizeof(what), "batch [%d, %d)", p0, p1);
            const uint64_t attn_before = uni.mem()->qsa_union_attn_builds();

            if (!ref.decode(p0, p1) || !uni.decode(p0, p1)) {
                FAIL("%s: decode failed", what);
                break;
            }

            // a single token takes the per-cell picker (its padded mask is not one row a token): no block picks
            const auto picks_ref = joined(ref.obs, "indexer_top_blk");
            const auto picks_uni = joined(uni.obs, "indexer_top_blk");
            if (picks_ref != picks_uni || (picks_ref.empty() && p1 - p0 >= LLAMA_QSA_UNION_MIN_ROWS)) {
                FAIL("%s: the picks differ (%zu / %zu layers captured)", what, picks_ref.size(), picks_uni.size());
            }

            if (p1 - p0 > 128) {
                check_visible(ref.obs, uni.obs, what, n_rows, n_dropped);
            } else if (p1 - p0 >= LLAMA_QSA_UNION_MIN_ROWS) {
                // one slice for union attention, the whole batch at once for the masked path (no named mask)
                if (joined(uni.obs, "qsa_sel").empty()) {
                    FAIL("%s: union attention did not take the batch in one slice", what);
                }
            } else if (uni.mem()->qsa_union_attn_builds() != attn_before) {
                FAIL("%s: a batch of %d rows took the union route", what, p1 - p0);
            }

            const auto out_ref = joined(ref.obs, "attn_pregate");
            const auto out_uni = joined(uni.obs, "attn_pregate");
            for (const auto & [il, bytes] : out_ref) {
                const auto it = out_uni.find(il);
                if (it == out_uni.end() || it->second.size() != bytes.size()) {
                    FAIL("%s: attn_pregate-%d missing or of another size", what, il);
                    continue;
                }
                worst_attn = std::max(worst_attn, nmse((const float *) bytes.data(), (const float *) it->second.data(), bytes.size()/sizeof(float)));
            }
            if (out_ref.empty()) {
                FAIL("%s: no attention output captured", what);
            }
            worst_logits = std::max(worst_logits, nmse(ref.logits.data(), uni.logits.data(), ref.logits.size()));
        }

        const uint64_t n_attn = uni.mem()->qsa_union_attn_builds() - uni.attn0;
        if (n_attn == 0 || ref.mem()->qsa_union_attn_builds() != 0) {
            FAIL("union attention did not run where asked (floors built: union %llu, masked %llu)",
                    (unsigned long long) n_attn, (unsigned long long) ref.mem()->qsa_union_attn_builds());
        }
        if (n_rows == 0 || n_dropped == 0) {
            FAIL("the oracle checked %lld rows and saw %lld cells dropped: it proved nothing", (long long) n_rows, (long long) n_dropped);
        }

        // F16 keys and values in both; the kernels differ only in the order they add
        const double tol = 1e-5;
        if (!(worst_attn < tol) || !(worst_logits < tol)) {
            FAIL("outputs outside tolerance: attention nmse %.3g, logits nmse %.3g (tolerance %.0e)", worst_attn, worst_logits, tol);
        }

        printf("test-qsa-union-model: union floors built %llu, visible sets equal on %lld sliced rows (%lld cells after "
               "the token dropped), attention nmse %.3g, logits nmse %.3g\n", (unsigned long long) n_attn,
               (long long) n_rows, (long long) n_dropped, worst_attn, worst_logits);

        // the startup line names the route each context takes
        if (ref.line.rfind("off,", 0) != 0 || uni.line.rfind("on,", 0) != 0) {
            FAIL("startup lines: the masked context's \"%s\", the union context's \"%s\"", ref.line.c_str(), uni.line.c_str());
        }
    }

    {
        // "1" in a context with a quantised cache builds "bias": the line says so in plain words, and its
        // batches build the GPU bias under the masked path, never union attention
        run fb(params, model, LLAMA_QSA_UNION_ATTN, /*kv_f16*/ false);
        if (fb.line.rfind("bias (", 0) != 0 || fb.line.find("F16") == std::string::npos) {
            FAIL("startup line of \"1\" with a quantised cache: \"%s\"", fb.line.c_str());
        }
        if (fb.ctx != nullptr) {
            const uint64_t bias0 = fb.mem()->qsa_union_bias_builds();
            if (!fb.decode(0, 128)) {
                FAIL("\"1\" with a quantised cache: decode failed");
            } else if (fb.mem()->qsa_union_attn_builds() != fb.attn0 || fb.mem()->qsa_union_bias_builds() == bias0) {
                FAIL("\"1\" with a quantised cache: union floors built %llu (want 0), GPU-bias floors %llu (want > 0)",
                        (unsigned long long) (fb.mem()->qsa_union_attn_builds() - fb.attn0),
                        (unsigned long long) (fb.mem()->qsa_union_bias_builds() - bias0));
            }
        }
    }

    llama_model_free(model);

    if (n_fail > 0) {
        fprintf(stderr, "test-qsa-union-model: %d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-qsa-union-model: all checks passed\n");
    return 0;
}

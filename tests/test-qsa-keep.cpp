// patch 4i: LLAMA_QSA_KEEP=1 keeps the qwen4exp indexer's finished block summaries instead of pooling
// every block again each ubatch. The output must not change by a single bit, so every scenario runs
// three contexts on one model - the reference (LLAMA_QSA_KEEP=0), keep, and keep with
// LLAMA_QSA_KEEP_CHECK=1 (which also rebuilds every summary and sums |kept - rebuilt|) - applies the
// same operations to all three, and compares every logit with memcmp. The plan counts show which
// path ran, so a planner that always falls back to the full rebuild cannot pass.
//
// The fixture's weights are N(0, 0.01), so attention adds less than a float's resolution to the
// residual stream and the logits do not move even when every block summary is garbage. Each scenario
// therefore runs a second time with the eval callback capturing the indexer's own tensors - the block
// summaries, their scores, the selection and the attention output - and compares those bit for bit.
//
// SHARE-PARTS-PLAN.md phase 3: with QSA_KEEP_SLICE=<rows> in the environment, every scenario runs at
// ubatch 4*<rows> and the keep and check contexts also slice the picker and attention (LLAMA_QSA_SLICE=<rows>)
// while the reference does the whole batch at once (LLAMA_QSA_SLICE=0). A slice's tensors are named
// <name>-<layer>-s<slice>; the observer joins them in slice order into the whole batch's tensor, and
// the run fails unless the sliced contexts really built sliced layers and the observer saw the slices.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-memory-hybrid-idx.h"

#include <clocale>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

enum variant {
    VARIANT_REF,   // switches unset: today's graph
    VARIANT_KEEP,  // LLAMA_QSA_KEEP=1
    VARIANT_CHECK, // LLAMA_QSA_KEEP=1 LLAMA_QSA_KEEP_CHECK=1
    VARIANT_COUNT,
};

static const char * variant_name[VARIANT_COUNT] = { "ref", "keep", "check" };

// the second pass: the indexer tensors each decode computed, in graph order
struct observer {
    std::vector<std::string>          names;
    std::vector<std::vector<uint8_t>> data;

    // QSA_KEEP_SLICE: slices joined onto an earlier entry, and the first slice seen out of order
    int         n_joined = 0;
    std::string bad;
};

// QSA_KEEP_SLICE's rows a slice, 0 for the plain runs
static uint32_t g_slice = 0;

// "indexer_score-3-s2" -> base "indexer_score-3", slice 2, and a view of it, "indexer_top_k-3-s2 (view)",
// -> "indexer_top_k-3 (view)"; -1 for a name without a slice suffix
static int slice_of(std::string name, std::string & base) {
    static const std::string view = " (view)";

    const bool is_view = name.size() > view.size() && name.compare(name.size() - view.size(), view.size(), view) == 0;
    if (is_view) {
        name.resize(name.size() - view.size());
    }

    const size_t dash = name.rfind('-');
    if (dash == std::string::npos || dash + 2 >= name.size() || name[dash + 1] != 's') {
        return -1;
    }
    for (size_t i = dash + 2; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return -1;
        }
    }
    base = name.substr(0, dash) + (is_view ? view : "");
    return std::atoi(name.c_str() + dash + 2);
}

static bool observed_name(const char * name) {
    for (const char * prefix : { "indexer_k-", "indexer_score-", "indexer_top_k-", "attn_pregate-" }) {
        if (strncmp(name, prefix, strlen(prefix)) == 0) {
            return true;
        }
    }
    return false;
}

static bool observe(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return observed_name(t->name);
    }

    auto * obs = (observer *) user_data;

    // a slice after the first is the next rows of the entry its first slice made
    std::string base;
    const int k = slice_of(t->name, base);
    if (k > 0) {
        int e = (int) obs->names.size() - 1;
        while (e >= 0 && obs->names[e] != base) {
            e--;
        }
        const int64_t n_seen = e >= 0 && ggml_is_contiguous(t) && ggml_nbytes(t) > 0 ? (int64_t) obs->data[e].size() : -1;
        if (n_seen <= 0) {
            if (obs->bad.empty()) {
                obs->bad = std::string(t->name) + " has no first slice before it";
            }
            return true;
        }
        const size_t n = ggml_nbytes(t);
        obs->data[e].resize(obs->data[e].size() + n);
        ggml_backend_tensor_get(t, obs->data[e].data() + n_seen, 0, n);
        obs->n_joined++;
        return true;
    }

    obs->names.emplace_back(k == 0 ? base : std::string(t->name));
    obs->data.emplace_back();

    // a view that is not contiguous is read as a mismatch of its own, below
    if (ggml_is_contiguous(t)) {
        obs->data.back().resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, obs->data.back().data(), 0, ggml_nbytes(t));
    }

    return true;
}

static bool g_observe = false;

struct ctx_cfg {
    uint32_t n_ctx     = 2048; // n_kv reaches 1024, where single-token decode takes the gather path
    uint32_t n_ubatch  = 64;
    uint32_t n_seq_max = 1;
    uint32_t n_rs_seq  = 0;
    bool     unified   = true;
};

static llama_context * make_ctx(const common_params & params, llama_model * model, const ctx_cfg & cfg, variant v, observer * obs) {
    // the memory reads the switches when it is made, so each context gets its own
    // the keep switch is on by default, so the reference turns it off explicitly
    unsetenv("LLAMA_QSA_KEEP_CHECK");
    setenv("LLAMA_QSA_KEEP", v == VARIANT_REF ? "0" : "1", 1);
    if (v == VARIANT_CHECK) {
        setenv("LLAMA_QSA_KEEP_CHECK", "1", 1);
    }
    // written in every run, "0" included: unset slices at 512 since the Q-slice rung, and the runs without
    // QSA_KEEP_SLICE check the whole batch at once
    setenv("LLAMA_QSA_SLICE", v == VARIANT_REF ? "0" : std::to_string(g_slice).c_str(), 1);

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx      = cfg.n_ctx;
    cparams.n_batch    = cfg.n_ctx;
    cparams.n_ubatch   = g_slice > 0 ? 4*g_slice : cfg.n_ubatch; // a slice run needs batches longer than a slice
    cparams.n_seq_max  = cfg.n_seq_max;
    cparams.n_rs_seq   = cfg.n_rs_seq;
    cparams.kv_unified = cfg.unified;

    if (g_observe) {
        cparams.cb_eval           = observe;
        cparams.cb_eval_user_data = obs;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);

    unsetenv("LLAMA_QSA_KEEP");
    unsetenv("LLAMA_QSA_KEEP_CHECK");
    unsetenv("LLAMA_QSA_SLICE");

    return ctx;
}

static llama_memory_hybrid_idx * get_mem(llama_context * ctx) {
    return dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx));
}

static llama_qsa_keep_stats get_stats(llama_context * ctx) {
    auto * mem = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx));
    GGML_ASSERT(mem != nullptr && "not a qwen4exp indexer memory");

    llama_synchronize(ctx);

    return mem->qsa_keep_stats();
}

// the three contexts of one scenario, driven in lock step
struct trio {
    const char * name;
    int n_vocab = 0;

    llama_context * ctx[VARIANT_COUNT] = {};
    observer        obs[VARIANT_COUNT];

    bool ok = true;

    int n_decode = 0;

    // QSA_KEEP_SLICE: layers built sliced when the contexts were made (the reserve's graphs), and whether
    // the scenario has a batch to slice (one sequence per stream and more than a slice of tokens)
    uint64_t slice_builds0[VARIANT_COUNT] = {};
    bool     slices_expected = true;

    trio(const char * name, const common_params & params, llama_model * model, const ctx_cfg & cfg) : name(name) {
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

        for (int v = 0; v < VARIANT_COUNT; ++v) {
            ctx[v] = make_ctx(params, model, cfg, (variant) v, &obs[v]);
            if (ctx[v] == nullptr) {
                fail("failed to create the %s context", variant_name[v]);
            }
        }

        // the reference must really be today's graph, and the others must really keep rows
        if (ok) {
            const bool keep[VARIANT_COUNT] = {
                dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx[VARIANT_REF]))->qsa_keep(),
                dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx[VARIANT_KEEP]))->qsa_keep(),
                dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx[VARIANT_CHECK]))->qsa_keep_check(),
            };
            if (keep[VARIANT_REF] || !keep[VARIANT_KEEP] || !keep[VARIANT_CHECK]) {
                fail("switches not picked up per context (ref %d, keep %d, check %d)",
                        keep[VARIANT_REF], keep[VARIANT_KEEP], keep[VARIANT_CHECK]);
            }

            for (int v = 0; v < VARIANT_COUNT; ++v) {
                const uint32_t want = v == VARIANT_REF ? 0 : g_slice;
                if (get_mem(ctx[v])->qsa_slice() != want) {
                    fail("LLAMA_QSA_SLICE not picked up per context: %s has %u, expected %u",
                            variant_name[v], get_mem(ctx[v])->qsa_slice(), want);
                }
                slice_builds0[v] = get_mem(ctx[v])->qsa_slice_builds();
            }
        }
    }

    ~trio() {
        for (auto * c : ctx) {
            llama_free(c);
        }
    }

    // reports the first failure only: the rest follow from it
    LLAMA_COMMON_ATTRIBUTE_FORMAT(2, 3)
    void fail(const char * fmt, ...) {
        if (ok) {
            va_list args;
            va_start(args, fmt);
            fprintf(stderr, "%s: FAIL: ", name);
            vfprintf(stderr, fmt, args);
            fprintf(stderr, "\n");
            va_end(args);
        }
        ok = false;
    }

    // decodes the batch in all three and compares every output logit bit for bit
    void decode(const llama_batch & batch, int expect = 0) {
        if (!ok) {
            return;
        }

        n_decode++;

        for (int v = 0; v < VARIANT_COUNT; ++v) {
            obs[v].names.clear();
            obs[v].data.clear();

            const int ret = llama_decode(ctx[v], batch);
            if (ret != expect) {
                fail("decode %d: %s returned %d, expected %d", n_decode, variant_name[v], ret, expect);
                return;
            }
        }

        if (expect != 0) {
            return;
        }

        if (g_observe) {
            if (obs[VARIANT_REF].names.empty()) {
                fail("decode %d: the eval callback saw no indexer tensor", n_decode);
                return;
            }

            for (int v = 0; v < VARIANT_COUNT; ++v) {
                if (!obs[v].bad.empty()) {
                    fail("decode %d: %s: %s", n_decode, variant_name[v], obs[v].bad.c_str());
                    return;
                }
            }

            for (int v = VARIANT_KEEP; v < VARIANT_COUNT; ++v) {
                if (obs[v].names != obs[VARIANT_REF].names) {
                    fail("decode %d: %s computed other indexer tensors than ref (%zu vs %zu)",
                            n_decode, variant_name[v], obs[v].names.size(), obs[VARIANT_REF].names.size());
                    return;
                }

                for (size_t k = 0; k < obs[v].names.size(); ++k) {
                    const auto & a = obs[VARIANT_REF].data[k];
                    const auto & b = obs[v].data[k];

                    if (a.empty() || a.size() != b.size() || memcmp(a.data(), b.data(), a.size()) != 0) {
                        fail("decode %d: %s differs from ref in %s", n_decode, variant_name[v], obs[v].names[k].c_str());
                        return;
                    }
                }
            }
        }

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i]) {
                continue;
            }

            const float * ref = llama_get_logits_ith(ctx[VARIANT_REF], i);

            for (int v = VARIANT_KEEP; v < VARIANT_COUNT; ++v) {
                const float * cur = llama_get_logits_ith(ctx[v], i);

                if (ref == nullptr || cur == nullptr) {
                    fail("decode %d: missing logits for output %d", n_decode, i);
                    return;
                }

                if (memcmp(ref, cur, n_vocab*sizeof(float)) != 0) {
                    int t = 0;
                    while (t < n_vocab && memcmp(&ref[t], &cur[t], sizeof(float)) == 0) {
                        t++;
                    }
                    fail("decode %d: %s logits differ at batch index %d (pos %d, seq %d), token %d: %.9g != %.9g",
                            n_decode, variant_name[v], i, batch.pos[i], batch.seq_id[i][0], t, (double) ref[t], (double) cur[t]);
                    return;
                }
            }
        }
    }

    void seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        for (int v = 0; ok && v < VARIANT_COUNT; ++v) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx[v]), seq_id, p0, p1)) {
                fail("seq_rm(%d, %d, %d) refused by %s", seq_id, p0, p1, variant_name[v]);
            }
        }
    }

    // the keep and check contexts plan alike; returns keep's counts
    llama_qsa_keep_stats stats() {
        const auto keep  = get_stats(ctx[VARIANT_KEEP]);
        const auto check = get_stats(ctx[VARIANT_CHECK]);

        if (keep.n_legacy != check.n_legacy || keep.n_full != check.n_full || keep.n_incr != check.n_incr) {
            fail("keep and check planned differently: %llu/%llu/%llu vs %llu/%llu/%llu",
                    (unsigned long long) keep.n_legacy,  (unsigned long long) keep.n_full,  (unsigned long long) keep.n_incr,
                    (unsigned long long) check.n_legacy, (unsigned long long) check.n_full, (unsigned long long) check.n_incr);
        }

        return keep;
    }

    // prints the counts and the check sums, which must all be 0
    bool finish() {
        const auto keep  = stats();
        const auto check = get_stats(ctx[VARIANT_CHECK]);

        std::string sums;
        for (const auto & [il, sum] : check.check) {
            sums += " l" + std::to_string(il) + "=" + std::to_string(sum);
            if (sum != 0.0) {
                fail("check sum of layer %d is %g", il, sum);
            }
        }
        if (check.check.empty()) {
            fail("the check context reported no layers");
        }

        // the sliced contexts must have sliced something since they were made (nothing, where the scenario
        // has no batch to slice), and the reference nothing; in the observed pass the observer must have
        // joined slices (the plain pass has no observer)
        std::string sliced;
        if (g_slice > 0) {
            uint64_t n_built[VARIANT_COUNT];
            for (int v = 0; v < VARIANT_COUNT; ++v) {
                n_built[v] = get_mem(ctx[v])->qsa_slice_builds() - slice_builds0[v];
            }
            const int n_joined = obs[VARIANT_KEEP].n_joined + obs[VARIANT_CHECK].n_joined;
            if (get_mem(ctx[VARIANT_REF])->qsa_slice_builds() != 0 || obs[VARIANT_REF].n_joined != 0) {
                fail("the reference sliced");
            }
            if (!slices_expected && (n_built[VARIANT_KEEP] != 0 || n_built[VARIANT_CHECK] != 0 || n_joined != 0)) {
                fail("sliced where nothing may be sliced (layers built sliced: keep %llu, check %llu; slices joined %d)",
                        (unsigned long long) n_built[VARIANT_KEEP], (unsigned long long) n_built[VARIANT_CHECK], n_joined);
            }
            if (slices_expected && (n_built[VARIANT_KEEP] == 0 || n_built[VARIANT_CHECK] == 0 || (g_observe && n_joined == 0))) {
                fail("slicing did not run (layers built sliced: keep %llu, check %llu; slices joined %d)",
                        (unsigned long long) n_built[VARIANT_KEEP], (unsigned long long) n_built[VARIANT_CHECK], n_joined);
            }
            sliced = ", sliced layers built " + std::to_string(n_built[VARIANT_KEEP]) + ", slices joined " + std::to_string(n_joined);
        }

        fprintf(stderr, "%s%s: %s: %d decodes, LEGACY %llu, FULL %llu, INCR %llu, check sums%s%s\n",
                name, g_observe ? " (observed)" : "", ok ? "ok" : "FAILED", n_decode,
                (unsigned long long) keep.n_legacy, (unsigned long long) keep.n_full, (unsigned long long) keep.n_incr,
                sums.c_str(), sliced.c_str());

        return ok;
    }
};

// the batch helpers take (token, pos, seq, output) rows
struct tok_row {
    llama_token  tok;
    llama_pos    pos;
    llama_seq_id seq;
    bool         out;
};

static void decode_rows(trio & t, const std::vector<tok_row> & rows, int expect = 0) {
    llama_batch batch = llama_batch_init((int32_t) rows.size(), 0, 1);
    for (const auto & r : rows) {
        common_batch_add(batch, r.tok, r.pos, { r.seq }, r.out);
    }
    t.decode(batch, expect);
    llama_batch_free(batch);
}

static llama_token tok_at(const trio & t, llama_seq_id seq, llama_pos pos, int salt = 0) {
    return (llama_token) ((7*(uint32_t) pos + 31*(uint32_t) seq + 13*(uint32_t) salt + 1) % (uint32_t) t.n_vocab);
}

// positions [p0, p1) of seq in one batch, every output requested
static void decode_range(trio & t, llama_seq_id seq, llama_pos p0, llama_pos p1, int salt = 0) {
    std::vector<tok_row> rows;
    for (llama_pos p = p0; p < p1; ++p) {
        rows.push_back({ tok_at(t, seq, p, salt), p, seq, true });
    }
    decode_rows(t, rows);
}

static void decode_singles(trio & t, llama_seq_id seq, llama_pos p0, llama_pos p1, int salt = 0) {
    for (llama_pos p = p0; p < p1 && t.ok; ++p) {
        decode_range(t, seq, p, p + 1, salt);
    }
}

static void expect(trio & t, bool cond, const char * what) {
    if (!cond) {
        t.fail("%s", what);
    }
}

// multi-ubatch prefill whose blocks straddle the ubatches, then decode across n_kv 768 and 1024:
// the block top-k path below 1024, the gather path from there
static bool test_prefill_decode(const common_params & params, llama_model * model) {
    trio t("prefill_decode", params, model, ctx_cfg());

    decode_range(t, 0, 0, 10);
    decode_range(t, 0, 10, 700);
    decode_singles(t, 0, 700, 1100);

    const auto st = t.stats();
    expect(t, st.n_incr > 0, "no INCR ubatch");
    expect(t, st.n_legacy == 0, "a LEGACY ubatch for one sequence");

    return t.finish();
}

// the speculative loop: verify 1 + d tokens, keep k of the drafts, trim the rest, go on from there.
// the trim is a seq_rm, which must cost no rebuild
static bool test_mtp_loop(const common_params & params, llama_model * model) {
    ctx_cfg cfg;
    cfg.n_rs_seq = 8;

    trio t("mtp_loop", params, model, cfg);

    decode_range(t, 0, 0, 300);

    const auto st0 = t.stats();

    llama_pos p = 300;
    for (int it = 0; it < 80 && t.ok; ++it) {
        const int d = 1 + it % 4;
        const int k = it % (d + 1);

        std::vector<tok_row> rows;
        rows.push_back({ tok_at(t, 0, p, 1), p, 0, true });
        for (int i = 1; i <= d; ++i) {
            rows.push_back({ tok_at(t, 0, p + i), p + i, 0, true });
        }
        decode_rows(t, rows);

        t.seq_rm(0, p + 1 + k, -1);
        p += 1 + k;
    }

    const auto st1 = t.stats();
    expect(t, st1.n_full == st0.n_full, "a trim forced a rebuild");
    expect(t, st1.n_incr == st0.n_incr + 80, "not every verify batch ran INCR");

    return t.finish();
}

// a recurrent-only checkpoint at an unaligned position, restored after the conversation went on,
// then cut back there and continued with different tokens
static bool test_partial_checkpoint(const common_params & params, llama_model * model) {
    trio t("partial_checkpoint", params, model, ctx_cfg());

    const llama_pos P = 301;

    decode_range(t, 0, 0, P);

    std::vector<std::vector<uint8_t>> ckpt(VARIANT_COUNT);
    for (int v = 0; v < VARIANT_COUNT; ++v) {
        ckpt[v].resize(llama_state_seq_get_size_ext(t.ctx[v], 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        const size_t n = llama_state_seq_get_data_ext(t.ctx[v], ckpt[v].data(), ckpt[v].size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        expect(t, n == ckpt[v].size() && n > 0, "partial checkpoint save failed");
    }

    decode_singles(t, 0, P, P + 50);

    const auto st0 = t.stats();

    for (int v = 0; v < VARIANT_COUNT && t.ok; ++v) {
        const size_t n = llama_state_seq_set_data_ext(t.ctx[v], ckpt[v].data(), ckpt[v].size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        expect(t, n == ckpt[v].size(), "partial checkpoint load failed");
    }
    t.seq_rm(0, P, -1);

    decode_range(t, 0, P, P + 7, 2);
    decode_singles(t, 0, P + 7, P + 40, 2);

    const auto st1 = t.stats();
    expect(t, st1.n_full == st0.n_full, "a partial restore forced a rebuild");
    expect(t, st1.n_incr > st0.n_incr, "no INCR after the partial restore");

    return t.finish();
}

// full state save/load, into a fresh context and in place: restored keys were never pooled, so the
// next ubatch must rebuild (FULL) before INCR resumes
static bool test_state_save_load(const common_params & params, llama_model * model) {
    trio a("state_save_load", params, model, ctx_cfg());

    decode_range(a, 0, 0, 500);
    decode_singles(a, 0, 500, 520);

    std::vector<std::vector<uint8_t>> state(VARIANT_COUNT);
    for (int v = 0; v < VARIANT_COUNT && a.ok; ++v) {
        state[v].resize(llama_state_get_size(a.ctx[v]));
        const size_t n = llama_state_get_data(a.ctx[v], state[v].data(), state[v].size());
        expect(a, n == state[v].size() && n > 0, "state save failed");
    }

    // with nothing saved, the loads below would leave empty contexts decoding at position 520
    if (!a.ok) {
        return a.finish();
    }

    // into a fresh context, which then only writes one token at a time: nothing to slice
    trio b("state_save_load/fresh", params, model, ctx_cfg());
    b.slices_expected = false;

    for (int v = 0; v < VARIANT_COUNT && b.ok; ++v) {
        const size_t n = llama_state_set_data(b.ctx[v], state[v].data(), state[v].size());
        expect(b, n == state[v].size(), "state load into a fresh context failed");
    }

    decode_singles(b, 0, 520, 550);

    const auto sb = b.stats();
    expect(b, sb.n_full >= 1, "no rebuild after loading into a fresh context");
    expect(b, sb.n_incr >= 1, "no INCR after the rebuild");

    // in place, after the conversation went elsewhere
    decode_singles(a, 0, 520, 545, 3);

    const auto sa0 = a.stats();

    for (int v = 0; v < VARIANT_COUNT && a.ok; ++v) {
        const size_t n = llama_state_set_data(a.ctx[v], state[v].data(), state[v].size());
        expect(a, n == state[v].size(), "state load in place failed");
    }

    decode_singles(a, 0, 520, 550);

    const auto sa1 = a.stats();
    expect(a, sa1.n_full > sa0.n_full, "no rebuild after loading in place");
    expect(a, sa1.n_incr > sa0.n_incr, "no INCR after the in-place rebuild");

    const bool ok_b = b.finish();
    const bool ok_a = a.finish();

    return ok_a && ok_b;
}

// one sequence's full state, as the server's prompt cache saves and restores it: the sequence is
// dropped, restored and continued, which needs one rebuild
static bool test_seq_state(const common_params & params, llama_model * model) {
    trio t("seq_state", params, model, ctx_cfg());

    decode_range(t, 0, 0, 333);

    std::vector<std::vector<uint8_t>> state(VARIANT_COUNT);
    for (int v = 0; v < VARIANT_COUNT && t.ok; ++v) {
        state[v].resize(llama_state_seq_get_size(t.ctx[v], 0));
        const size_t n = llama_state_seq_get_data(t.ctx[v], state[v].data(), state[v].size(), 0);
        expect(t, n == state[v].size() && n > 0, "sequence state save failed");
    }

    decode_singles(t, 0, 333, 350, 5);
    t.seq_rm(0, -1, -1);

    const auto st0 = t.stats();

    for (int v = 0; v < VARIANT_COUNT && t.ok; ++v) {
        const size_t n = llama_state_seq_set_data(t.ctx[v], state[v].data(), state[v].size(), 0);
        expect(t, n == state[v].size(), "sequence state load failed");
    }

    decode_range(t, 0, 333, 340);
    decode_singles(t, 0, 340, 370);

    const auto st1 = t.stats();
    expect(t, st1.n_full == st0.n_full + 1, "no single rebuild after the sequence was restored");
    expect(t, st1.n_incr > st0.n_incr, "no INCR after the rebuild");

    return t.finish();
}

// a middle range removed leaves partial blocks and renumbers the block ids after it. The M-RoPE batch
// check only lets positions move forward (Y > X), so the hole is not decoded again: decoding goes on
// past it, then jumps forward, including a batch that touches more blocks than INCR has room for
static bool test_remove_middle(const common_params & params, llama_model * model) {
    trio t("remove_middle", params, model, ctx_cfg());

    decode_range(t, 0, 0, 400);
    decode_singles(t, 0, 400, 410);

    const auto st0 = t.stats();

    t.seq_rm(0, 101, 203);
    decode_singles(t, 0, 410, 440);

    const auto st1 = t.stats();
    expect(t, st1.n_full == st0.n_full, "removing a middle range forced a rebuild");

    // a forward jump, then one batch over three blocks: n_slots is 2 for three tokens
    decode_singles(t, 0, 449, 452);
    decode_rows(t, { { tok_at(t, 0, 460), 460, 0, true }, { tok_at(t, 0, 468), 468, 0, true }, { tok_at(t, 0, 476), 476, 0, true } });
    decode_singles(t, 0, 477, 500);

    const auto st2 = t.stats();
    expect(t, st2.n_full == st1.n_full + 1, "the batch over three blocks did not rebuild");
    expect(t, st2.n_incr > st1.n_incr, "no INCR after the jumps");

    return t.finish();
}

// two sequences in one stream cannot keep rows (LEGACY); once one is gone the stream rebuilds (FULL)
// and keeps rows again (INCR). Then two streams, together and one at a time
static bool test_two_seqs(const common_params & params, llama_model * model) {
    bool ok = true;

    {
        ctx_cfg cfg;
        cfg.n_seq_max = 2;
        cfg.unified   = true;

        trio t("two_seqs/unified", params, model, cfg);

        std::vector<tok_row> rows;
        for (llama_pos p = 0; p < 100; ++p) {
            rows.push_back({ tok_at(t, 0, p), p, 0, p == 99 });
            rows.push_back({ tok_at(t, 1, p), p, 1, p == 99 });
        }
        decode_rows(t, rows);

        for (llama_pos p = 100; p < 110 && t.ok; ++p) {
            decode_rows(t, { { tok_at(t, 0, p), p, 0, true } });
            decode_rows(t, { { tok_at(t, 1, p), p, 1, true } });
        }

        const auto st0 = t.stats();
        expect(t, st0.n_legacy >= 1 && st0.n_full == 0 && st0.n_incr == 0, "two sequences in one stream did not stay LEGACY");

        t.seq_rm(1, -1, -1);
        decode_singles(t, 0, 110, 130);

        const auto st1 = t.stats();
        expect(t, st1.n_legacy == st0.n_legacy, "LEGACY with one sequence left");
        expect(t, st1.n_full == 1, "no single rebuild once one sequence was left");
        expect(t, st1.n_incr >= 1, "no INCR after the rebuild");

        ok = t.finish() && ok;
    }

    {
        ctx_cfg cfg;
        cfg.n_seq_max = 2;
        cfg.unified   = false;

        trio t("two_seqs/streams", params, model, cfg);
        t.slices_expected = false; // two streams in a batch: never sliced

        std::vector<tok_row> rows;
        for (llama_pos p = 0; p < 100; ++p) {
            rows.push_back({ tok_at(t, 0, p), p, 0, true });
            rows.push_back({ tok_at(t, 1, p), p, 1, true });
        }
        decode_rows(t, rows);

        for (llama_pos p = 100; p < 110 && t.ok; ++p) {
            decode_rows(t, { { tok_at(t, 0, p), p, 0, true }, { tok_at(t, 1, p), p, 1, true } });
        }

        // one stream at a time: the second one's rows sit at stream offset 1
        decode_singles(t, 1, 110, 125);
        decode_singles(t, 0, 110, 125);

        const auto st = t.stats();
        expect(t, st.n_legacy == 0, "LEGACY with one sequence per stream");
        expect(t, st.n_incr >= 30, "streams did not run INCR");

        ok = t.finish() && ok;
    }

    return ok;
}

// a position given twice repeats a block member, which the store cannot key: LEGACY until the
// repeat is gone, then one rebuild and INCR again
static bool test_same_pos_twice(const common_params & params, llama_model * model) {
    ctx_cfg cfg;
    cfg.n_rs_seq = 8;

    trio t("same_pos_twice", params, model, cfg);

    decode_range(t, 0, 0, 200);

    const auto st0 = t.stats();

    decode_rows(t, { { tok_at(t, 0, 200), 200, 0, true }, { tok_at(t, 0, 200, 1), 200, 0, true } });
    decode_singles(t, 0, 201, 205);

    // taking the tail off rescans the stream, and the repeat at 200 is still there
    t.seq_rm(0, 204, -1);
    decode_singles(t, 0, 204, 206);

    const auto st1 = t.stats();
    expect(t, st1.n_legacy == st0.n_legacy + 1 + 4 + 2, "the repeated position did not keep every ubatch LEGACY");
    expect(t, st1.n_incr == st0.n_incr && st1.n_full == st0.n_full, "rows kept over a repeated position");

    // start the sequence over, with other tokens so no row from before can pass for a rebuilt one
    t.seq_rm(0, -1, -1);
    decode_range(t, 0, 0, 150, 4);
    decode_singles(t, 0, 150, 170, 4);

    const auto st2 = t.stats();
    expect(t, st2.n_legacy == st1.n_legacy, "LEGACY after the repeat was gone");
    expect(t, st2.n_full == st1.n_full + 1, "no single rebuild after the repeat was gone");
    expect(t, st2.n_incr >= st1.n_incr + 20, "no INCR after the rebuild");

    return t.finish();
}

// a repeat made while the stream is shared must still be seen once its sequence is alone there
static bool test_repeat_while_shared(const common_params & params, llama_model * model) {
    ctx_cfg cfg;
    cfg.n_seq_max = 2;
    cfg.unified   = true;

    trio t("repeat_while_shared", params, model, cfg);
    t.slices_expected = false; // batches of 100 tokens at most, never longer than a slice (128 or more)

    decode_range(t, 0, 0, 100);

    const auto st0 = t.stats();

    decode_range(t, 1, 0, 40);

    // seq 1 gives position 40 twice while seq 0 shares the stream
    decode_rows(t, { { tok_at(t, 1, 40), 40, 1, true }, { tok_at(t, 1, 40, 1), 40, 1, true } });

    t.seq_rm(0, -1, -1);
    decode_singles(t, 1, 41, 50);

    const auto st1 = t.stats();
    expect(t, st1.n_full == st0.n_full && st1.n_incr == st0.n_incr, "rows kept over a repeated position");

    return t.finish();
}

static bool abort_now(void * data) {
    return *(const bool *) data;
}

// an aborted compute is rolled back by the context; its plan was never committed, so the next ubatch
// must rebuild
static bool test_abort(const common_params & params, llama_model * model) {
    ctx_cfg cfg;
    cfg.n_rs_seq = 8;

    trio t("abort", params, model, cfg);

    decode_range(t, 0, 0, 200);
    decode_singles(t, 0, 200, 205);

    const auto st0 = t.stats();

    bool do_abort = true;
    for (auto * c : t.ctx) {
        llama_set_abort_callback(c, abort_now, &do_abort);
    }

    // 2: aborted, and the context removed the ubatch's positions again
    decode_rows(t, { { tok_at(t, 0, 205), 205, 0, true } }, 2);

    do_abort = false;

    decode_singles(t, 0, 205, 230);

    const auto st1 = t.stats();
    expect(t, st1.n_full == st0.n_full + 1, "no rebuild after the aborted ubatch");
    expect(t, st1.n_incr > st0.n_incr, "no INCR after the rebuild");

    return t.finish();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (const char * e = getenv("QSA_KEEP_SLICE")) {
        g_slice = (uint32_t) atoi(e);
        fprintf(stderr, "%s: slice runs: the keep and check contexts slice %u rows at a time, at ubatch %u\n", __func__, g_slice, 4*g_slice);
    }

    ggml_backend_load_all();

    // the contexts are made one by one below; the model alone is loaded here
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "%s: failed to load the model\n", __func__);
        return 1;
    }

    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    if (strcmp(arch, "qwen4exp") != 0) {
        fprintf(stderr, "%s: needs a qwen4exp model, got '%s'\n", __func__, arch);
        llama_model_free(model);
        return 1;
    }

    bool ok = true;

    // first the graphs as they run in production, then with the indexer tensors captured
    for (bool obs : { false, true }) {
        g_observe = obs;

        ok = test_prefill_decode    (params, model) && ok;
        ok = test_mtp_loop          (params, model) && ok;
        ok = test_partial_checkpoint(params, model) && ok;
        ok = test_state_save_load   (params, model) && ok;
        ok = test_seq_state         (params, model) && ok;
        ok = test_remove_middle     (params, model) && ok;
        ok = test_two_seqs          (params, model) && ok;
        ok = test_same_pos_twice    (params, model) && ok;
        ok = test_repeat_while_shared(params, model) && ok;
        ok = test_abort             (params, model) && ok;
    }

    llama_model_free(model);

    fprintf(stderr, "%s: %s\n", __func__, ok ? "all scenarios bit-identical" : "FAILED");

    return ok ? 0 : 1;
}

// The block top-k picker never spends a pick on a block after the token (SHARE-PARTS-PLAN.md phase 1).
//
// A reading-in batch picks blocks before the attention mask hides the cells after each token, and the
// old rule gave those future blocks the tail's +1e9, so they filled the picks (llama-qsa-picks.h). On
// the tiny qwen4exp fixture (indexer_top_k 8, ratio 4: 3 blocks picked a token) this checks:
//   causal     every picked block the token can see is earlier or its own tail, the tail is always
//              picked, and no pick is wasted while a visible block is left out. Batch 10 -> 74 ends
//              mid-block, so its spare block holds cells after most of the batch. The old rule
//              (LLAMA_QSA_CAUSAL_PICKS=0), in the same process, must fail the same check, and the
//              counter must say so (and read 0 with the rule on).
//   invariant  (CPU only) batch 256, batch 64 and one token at a time pick the same visible blocks
//              for every position, below 1,024 cells, where one token at a time takes the same path.
//   shared     a stream shared by two sequences keeps the old rule: rule on and off byte-identical.
//   per-cell   with LLAMA_QWEN4EXP_BLOCK_TOPK=0 (ctest sets it; it is read once per process) the rule
//              must not move a byte: rule on and off identical, and a hash to hold against the build
//              before the fix. The block path prints its old-rule hash for the same comparison.
// The fixture's weights barely move the logits, so the picks themselves are captured and read.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-memory-hybrid-idx.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

static const int R = 4; // the fixture's compress ratio

// every capture of one decode, in graph order
struct observer {
    std::vector<std::string>          names;
    std::vector<std::vector<uint8_t>> data;
    std::vector<int64_t>              ne0, ne1;
};

static bool observe(ggml_tensor * t, bool ask, void * user_data) {
    // the tensors themselves, "<name>-<layer>", not their views ("... (view)", "... (reshaped)")
    static const char * prefixes[] = { "indexer_k-", "indexer_score-", "indexer_score_tokens-", "indexer_top_blk-",
                                       "indexer_top_k-", "attn_pregate-" };
    if (ask) {
        for (const char * p : prefixes) {
            const size_t n = strlen(p);
            if (strncmp(t->name, p, n) == 0 && t->name[n] != '\0' && strspn(t->name + n, "0123456789") == strlen(t->name + n)) {
                return true;
            }
        }
        return false;
    }
    auto * obs = (observer *) user_data;
    obs->names.emplace_back(t->name);
    obs->ne0.push_back(t->ne[0]);
    obs->ne1.push_back(t->ne[1]);
    obs->data.emplace_back(ggml_nbytes(t));
    GGML_ASSERT(ggml_is_contiguous(t));
    ggml_backend_tensor_get(t, obs->data.back().data(), 0, ggml_nbytes(t));
    return true;
}

// one context with the rule on or off, recording every capture and logit
struct run {
    llama_context * ctx = nullptr;
    observer        obs;
    uint64_t        hash = 1469598103934665603ull; // FNV-1a over captures and logits, in order
    bool            ok   = true;

    run(const common_params & params, llama_model * model, bool causal, uint32_t n_ubatch, uint32_t n_seq = 1) {
        setenv("LLAMA_QSA_CAUSAL_PICKS", causal ? "1" : "0", 1);
        auto cparams = common_context_params_to_llama(params);
        cparams.n_ctx = 1024; cparams.n_batch = 1024; cparams.n_ubatch = n_ubatch;
        cparams.n_seq_max = n_seq; cparams.kv_unified = true;
        cparams.cb_eval = observe; cparams.cb_eval_user_data = &obs;
        ctx = llama_init_from_model(model, cparams);
        unsetenv("LLAMA_QSA_CAUSAL_PICKS");
        if (ctx == nullptr || mem()->qsa_causal_picks() != causal) {
            fprintf(stderr, "FAIL: the rule switch was not picked up per context (asked %d)\n", causal);
            ok = false;
        }
    }
    ~run() { llama_free(ctx); }

    llama_memory_hybrid_idx * mem() const { return dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx)); }

    void mix(const void * p, size_t n) {
        for (size_t k = 0; k < n; ++k) { hash = (hash ^ ((const uint8_t *) p)[k])*1099511628211ull; }
    }

    // decodes positions [p0, p1) of sequences [0, n_seq) in one batch; the captures stay in obs until the next call
    void decode(llama_pos p0, llama_pos p1, int n_seq = 1) {
        obs = observer();
        llama_batch batch = llama_batch_init((p1 - p0)*n_seq, 0, 1);
        for (llama_pos p = p0; p < p1; ++p) {
            for (llama_seq_id seq = 0; seq < n_seq; ++seq) {
                common_batch_add(batch, (llama_token) ((7*p + 31*seq + 1) % 97), p, { seq }, true);
            }
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: decode [%d, %d) failed\n", p0, p1); ok = false; }
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        bool finite = true;
        for (int i = 0; i < batch.n_tokens; ++i) {
            const float * l = llama_get_logits_ith(ctx, i);
            for (int t = 0; l != nullptr && t < n_vocab; ++t) {
                finite = finite && std::isfinite(l[t]);
            }
            if (l != nullptr) { mix(l, n_vocab*sizeof(float)); }
        }
        if (!finite) { fprintf(stderr, "FAIL: a non-finite logit in [%d, %d)\n", p0, p1); ok = false; }
        for (const auto & d : obs.data) { mix(d.data(), d.size()); }
        llama_batch_free(batch);
    }
};

// the picked block ids of one layer, row by row, from every capture named indexer_top_blk-<il>
// since the last clear: {layer: rows}
static std::map<int, std::vector<std::vector<int32_t>>> picks_of(const observer & obs) {
    std::map<int, std::vector<std::vector<int32_t>>> out;
    for (size_t k = 0; k < obs.names.size(); ++k) {
        if (obs.names[k].rfind("indexer_top_blk-", 0) != 0) { continue; }
        const int il = atoi(obs.names[k].c_str() + strlen("indexer_top_blk-"));
        const auto * v = (const int32_t *) obs.data[k].data();
        for (int64_t row = 0; row < obs.ne1[k]; ++row) {
            out[il].emplace_back(v + row*obs.ne0[k], v + (row + 1)*obs.ne0[k]);
        }
    }
    return out;
}

// a fresh single-sequence context holds positions [0, n_cells) in cells of the same index, so block id b
// below n_bid = n_cells/R starts at R*b and id n_bid is the spare block, its cells from R*n_bid on.
// Returns the block starts a token at q picked and can see (the spare as R*n_bid), or -1 on a wasted pick
static int64_t visible_start(int32_t b, int64_t q, int64_t n_cells) {
    const int64_t n_bid = n_cells/R;
    const int64_t start = b <= n_bid ? R*(int64_t) b : n_cells;
    return start <= q ? start : -1;
}

// the context a row at q of a batch [p0, p1) cut into ubatches of ub saw: [0, the end of q's ubatch)
static int64_t cells_at(int64_t q, int64_t p0, int64_t p1, int64_t ub) {
    return std::min<int64_t>(p1, p0 + ((q - p0)/ub + 1)*ub);
}

// the causal check on one decode of positions [p0, p1): returns the number of rows that break it
static int count_bad_rows(const observer & obs, int64_t p0, int64_t p1, int64_t ub, bool verbose) {
    int bad = 0;
    const auto all = picks_of(obs);
    if (all.empty()) { fprintf(stderr, "FAIL: no indexer_top_blk capture\n"); return 1 << 20; }
    for (const auto & [il, rows] : all) {
        if ((int64_t) rows.size() != p1 - p0) { fprintf(stderr, "FAIL: layer %d gave %zu rows\n", il, rows.size()); return 1 << 20; }
        for (int64_t q = p0; q < p1; ++q) {
            const auto & row = rows[q - p0];
            std::set<int64_t> seen;
            for (int32_t b : row) {
                const int64_t s = visible_start(b, q, cells_at(q, p0, p1, ub));
                if (s >= 0) { seen.insert(s); }
            }
            // candidates: every block starting at or before q (full or the spare); the tail is q's own block
            const int64_t n_cand = q/R + 1;
            const int64_t want   = std::min<int64_t>((int64_t) row.size(), n_cand);
            const bool    tail   = q % R == R - 1 || seen.count(q/R*R) > 0;
            if ((int64_t) seen.size() != want || !tail) {
                if (verbose && bad < 3) {
                    fprintf(stderr, "  layer %d pos %lld: %zu visible picks of %lld wanted, tail %s\n",
                            il, (long long) q, seen.size(), (long long) want, tail ? "picked" : "missing");
                }
                bad++;
            }
        }
    }
    return bad;
}

static bool test_causal(const common_params & params, llama_model * model) {
    bool ok = true;
    for (bool causal : { true, false }) {
        run r(params, model, causal, 64);
        int bad = 0;
        const llama_pos ranges[][2] = { { 0, 10 }, { 10, 74 }, { 74, 330 } };
        for (const auto & rg : ranges) {
            r.decode(rg[0], rg[1]);
            bad += count_bad_rows(r.obs, rg[0], rg[1], 64, causal);
        }
        const auto st = r.mem()->qsa_pick_stats().total;
        fprintf(stderr, "causal (%s rule): %d rows break the check; counter: %lld rows every pick past the token, "
                "%lld of %lld picks past the token\n", causal ? "causal" : "old", bad,
                (long long) st.n_rows_future, (long long) st.n_picks_future, (long long) st.n_picks);
        if (causal && (bad != 0 || st.n_rows_future != 0 || st.n_picks_future != 0 || st.n_rows == 0)) {
            fprintf(stderr, "FAIL: the causal rule wastes picks\n"); ok = false;
        }
        if (!causal && (bad == 0 || st.n_rows_future == 0)) {
            fprintf(stderr, "FAIL: the old rule shows no waste: the check cannot see the bug\n"); ok = false;
        }
        if (!causal) { fprintf(stderr, "causal: old rule hash %016llx\n", (unsigned long long) r.hash); }
        ok = ok && r.ok;
    }
    return ok;
}

static bool test_invariant(const common_params & params, llama_model * model) {
    const llama_pos N = 300;
    // {layer: {pos: visible block starts}} for batch 256, batch 64, one token at a time
    std::vector<std::map<int, std::map<int64_t, std::set<int64_t>>>> vis(3);
    bool ok = true;
    for (int m = 0; m < 3; ++m) {
        run r(params, model, true, m == 0 ? 256 : 64);
        const llama_pos step = m == 2 ? 1 : N;
        for (llama_pos p0 = 0; p0 < N; p0 += step) {
            r.decode(p0, p0 + step);
            // the ubatches of one batch come in order, so rows run p0, p0 + 1, ...
            for (const auto & [il, rows] : picks_of(r.obs)) {
                for (size_t k = 0; k < rows.size(); ++k) {
                    const int64_t q = p0 + (int64_t) k;
                    const int64_t n_cells = cells_at(q, p0, p0 + step, m == 0 ? 256 : 64);
                    for (int32_t b : rows[k]) {
                        const int64_t s = visible_start(b, q, n_cells);
                        if (s >= 0) { vis[m][il][q].insert(s); }
                    }
                }
            }
        }
        ok = ok && r.ok;
    }
    int bad = 0;
    for (const auto & [il, by_pos] : vis[2]) {
        for (const auto & [q, set] : by_pos) {
            for (int m = 0; m < 2; ++m) {
                if (vis[m][il][q] != set) {
                    if (bad++ < 3) { fprintf(stderr, "  layer %d pos %lld: batch %d picks other visible blocks than one at a time\n",
                                             il, (long long) q, m == 0 ? 256 : 64); }
                }
            }
        }
    }
    fprintf(stderr, "invariant: %zu layers x %d positions, %d differ\n", vis[2].size(), N, bad);
    if (bad != 0 || vis[2].empty()) { fprintf(stderr, "FAIL: the picks depend on the batch\n"); ok = false; }
    return ok;
}

// rule on and off write the same bytes: for a shared stream (the old rule is kept there), and for
// the per-cell path (the mask is added before its top-k, so the rule cannot move it)
static bool test_same_bytes(const common_params & params, llama_model * model, const char * name, uint32_t n_seq) {
    uint64_t hash[2] = {};
    bool ok = true;
    for (int causal = 0; causal < 2; ++causal) {
        run r(params, model, causal, 64, n_seq);
        // every sequence from the first batch on, so the stream is never one sequence's alone
        r.decode(0, 150, n_seq);
        for (llama_pos p = 150; p < 160; ++p) { r.decode(p, p + 1, n_seq); }
        hash[causal] = r.hash;
        ok = ok && r.ok;
    }
    fprintf(stderr, "%s: hash %016llx with the old rule, %016llx with the causal rule\n", name,
            (unsigned long long) hash[0], (unsigned long long) hash[1]);
    if (hash[0] != hash[1]) { fprintf(stderr, "FAIL: %s: the rule moved a byte\n", name); ok = false; }
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (model == nullptr) {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }

    const char * topk_env = getenv("LLAMA_QWEN4EXP_BLOCK_TOPK");
    const bool   per_cell = topk_env != nullptr && strcmp(topk_env, "0") == 0;

    bool ok = true;
    if (per_cell) {
        ok = test_same_bytes(params, model, "per-cell", 1) && ok;
    } else {
        ok = test_causal(params, model) && ok;
        if (params.n_gpu_layers == 0) {
            ok = test_invariant(params, model) && ok;
        } else {
            fprintf(stderr, "invariant: skipped off the CPU (a GPU batch rounds differently from one token)\n");
        }
        ok = test_same_bytes(params, model, "shared", 2) && ok;
    }

    llama_model_free(model);
    fprintf(stderr, "%s: %s\n", __func__, ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}

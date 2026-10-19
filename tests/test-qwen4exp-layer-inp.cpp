// A floor's input for a draft (DFLASH-PLAN step 2), on the MTP fixture (qwen4exp-moe-e64mtp.gguf: two trunk floors
// and one MTP block, four hc streams). DFlash and EAGLE3 ask the library for some floors' inputs
// (llama_set_embeddings_layer_inp); qwen4exp keeps four copies of its notes on every floor (the hc streams), and the
// draft was trained on their plain mean. A read-in of 40 tokens at -ub 16, then 3 words written one at a time:
//   off      no floor asked for: no layer_inp node is built; the node list's and logits' hashes are printed to
//            compare with the build before the fix by hand (--off-only stops here, so the old build runs it too)
//   unseen   floor 1 asked for, the graph unsplit (the eval callback asks for nothing): logits and the apprentice's
//            notes (t_h_nextn) byte-identical to off, one layer_inp-1 node a graph and none for floor 0
//   mean     floors 0 and 1 asked for, the streams captured by exact name (hc_init into floor 0, l_last-0 into
//            floor 1; "<prefix>-<digits>" only, never a view): what llama_get_embeddings_layer_inp hands back is
//            n_embd wide and equals, byte for byte, stream 0 plus streams 1, 2 and 3 in turn, times 1/4; the same as
//            the layer_inp node and as the unseen run's; floor 1's streams differ, so stream 0 alone would fail
//   toggle   a context that reads in with no floor asked for, then asks for floor 1 and writes: the graph is
//            rebuilt (no abort), and the words written hand back the unseen run's numbers and logits

#include "common.h"
#include "llama.h"

#include "../src/llama-ext.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int N_FLOORS = 2;  // the fixture's trunk
constexpr int N_READ   = 40; // tokens read in, at -ub 16: three graphs
constexpr int N_WRITE  = 3;  // words written after, one graph each

std::string g_path;
int         g_ngl      = 99;
int         g_failures = 0;

void check(bool ok, const std::string & what) {
    printf("  %-90s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

uint64_t fnv(uint64_t h, const void * p, size_t n) {
    for (size_t k = 0; k < n; k++) {
        h = (h ^ ((const uint8_t *) p)[k])*1099511628211ull;
    }
    return h;
}
constexpr uint64_t FNV0 = 1469598103934665603ull;

// the floor in a node named exactly "<prefix>-<digits>", else -1: "l_last-0 (reshaped)" is a view (gotcha 30)
int floor_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-' || name[n + 1] == '\0') {
        return -1;
    }
    int il = 0;
    for (const char * c = name + n + 1; *c; c++) {
        if (*c < '0' || *c > '9') {
            return -1;
        }
        il = il*10 + (*c - '0');
    }
    return il;
}

// the streams going into a floor: hc_init into floor 0, the previous floor's l_last after that
int stream_floor(const char * name) {
    if (strcmp(name, "hc_init") == 0) {
        return 0;
    }
    const int il = floor_of(name, "l_last");
    return il >= 0 && il + 1 < N_FLOORS ? il + 1 : -1;
}

// one decode's numbers, rows in token order (one sequence, so its graphs run in order)
struct record {
    int n_tokens = 0;
    std::vector<float> streams[N_FLOORS]; // [n_embd, hc, n_tokens] per floor, captured (mean runs only)
    std::vector<float> node[N_FLOORS];    // the layer_inp-<il> nodes, captured (mean runs only)
    std::vector<float> api[N_FLOORS];     // llama_get_embeddings_layer_inp, for each floor asked for
    std::vector<float> logits;
    std::vector<float> nextn;
};

struct watch {
    bool     ask     = false; // capture (split the graph at) the streams and layer_inp nodes; else names only
    uint64_t nodes   = FNV0;  // every node's op, name and shape, in the order the scheduler hands them over
    int64_t  n_nodes = 0;
    int      built[N_FLOORS] = {};
    bool     wide    = true;  // every layer_inp node is [n_embd, rows] f32, contiguous
    int64_t  n_embd  = 0;
    record * rec     = nullptr;
};

void append(std::vector<float> & v, ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    const size_t n = v.size();
    v.resize(n + ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data() + n, 0, ggml_nbytes(t));
}

bool observe(ggml_tensor * t, bool ask, void * user_data) {
    auto * w = (watch *) user_data;
    const int il_mean   = floor_of(t->name, "layer_inp");
    const int il_stream = stream_floor(t->name);
    if (ask) {
        w->nodes = fnv(w->nodes, &t->op, sizeof(t->op));
        w->nodes = fnv(w->nodes, t->name, strlen(t->name));
        w->nodes = fnv(w->nodes, t->ne, sizeof(t->ne));
        w->n_nodes++;
        if (il_mean >= 0 && il_mean < N_FLOORS) {
            w->built[il_mean]++;
            w->wide = w->wide && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == w->n_embd &&
                    t->ne[2] == 1 && t->ne[3] == 1;
        }
        return w->ask && ((il_mean >= 0 && il_mean < N_FLOORS) || il_stream >= 0);
    }
    if (il_mean >= 0 && il_mean < N_FLOORS) {
        append(w->rec->node[il_mean], t);
    } else if (il_stream >= 0) {
        append(w->rec->streams[il_stream], t);
    }
    return true;
}

struct run_out {
    std::vector<record> recs;
    watch w;
};

// floors_at_start are asked for before the read-in, floors_after only for the words written after it
run_out run(llama_model * model, const std::vector<int> & floors_at_start, const std::vector<int> & floors_after,
        bool ask, bool last_only = false) {
    run_out out;
    out.w.ask    = ask;
    out.w.n_embd = llama_model_n_embd(model);
    out.recs.resize(1 + N_WRITE);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = 16; cp.n_seq_max = 1;
    cp.cb_eval = observe; cp.cb_eval_user_data = &out.w;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        check(false, "the context opens");
        return out;
    }
    llama_set_embeddings_nextn(ctx, true, /*masked*/ false); // the apprentice's notes, every row
    std::vector<bool> on(N_FLOORS, false);
    for (int il : floors_at_start) { llama_set_embeddings_layer_inp(ctx, il, true); on[il] = true; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_out   = llama_model_n_embd_out(model);
    llama_batch batch = llama_batch_init(N_READ, 0, 1);
    for (int d = 0, pos = 0; d <= N_WRITE; d++) {
        if (d == 1) {
            for (int il : floors_after) { llama_set_embeddings_layer_inp(ctx, il, true); on[il] = true; }
        }
        common_batch_clear(batch);
        const int n = d == 0 ? N_READ : 1;
        for (int i = 0; i < n; i++, pos++) {
            // last_only: the read-in asks for the last word's logits only, so the last floor drops the other rows
            common_batch_add(batch, 1 + (11*pos + 5) % 97, pos, { 0 }, !last_only || i == n - 1);
        }
        const int n_rows = last_only ? 1 : n;
        record & r = out.recs[d];
        r.n_tokens = n;
        out.w.rec  = &r;
        if (llama_decode(ctx, batch) != 0) {
            check(false, "decode " + std::to_string(d));
            break;
        }
        const float * lg = llama_get_logits(ctx);
        r.logits.assign(lg, lg + (size_t) n_rows*n_vocab);
        const float * nx = llama_get_embeddings_nextn(ctx);
        r.nextn.assign(nx, nx + (size_t) n*n_out);
        for (int il = 0; il < N_FLOORS; il++) {
            if (on[il]) {
                const float * v = llama_get_embeddings_layer_inp(ctx, il);
                r.api[il].assign(v, v + (size_t) n*out.w.n_embd);
            }
        }
    }
    llama_batch_free(batch);
    llama_free(ctx);
    return out;
}

uint64_t hash_of(const run_out & o, std::vector<float> record::* field) {
    uint64_t h = FNV0;
    for (const auto & r : o.recs) { h = fnv(h, (r.*field).data(), (r.*field).size()*sizeof(float)); }
    return h;
}

bool same(const run_out & a, const run_out & b, std::vector<float> record::* field) {
    for (size_t d = 0; d < a.recs.size(); d++) {
        const auto & x = a.recs[d].*field, & y = b.recs[d].*field;
        if (x.empty() || x.size() != y.size() || memcmp(x.data(), y.data(), x.size()*sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

// stream 0 plus streams 1, 2, ... in turn, times 1/hc: the graph's order of operations, so equal byte for byte
std::vector<float> host_mean(const std::vector<float> & streams, int64_t n_embd, int64_t hc, int n_tokens) {
    std::vector<float> m((size_t) n_embd*n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        for (int64_t k = 0; k < n_embd; k++) {
            float acc = streams[((size_t) t*hc)*n_embd + k];
            for (int64_t s = 1; s < hc; s++) {
                acc += streams[((size_t) t*hc + s)*n_embd + k];
            }
            m[(size_t) t*n_embd + k] = acc*(1.0f/hc);
        }
    }
    return m;
}

size_t n_differ(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return SIZE_MAX;
    }
    size_t n = 0;
    for (size_t i = 0; i < a.size(); i++) { n += a[i] != b[i]; }
    return n;
}

// mean: per decode and floor, the API's numbers against the captured streams' mean (exact), and the node
void check_mean(const run_out & m, int64_t n_embd, int64_t hc) {
    for (int il = 0; il < N_FLOORS; il++) {
        size_t bad_api = 0, bad_node = 0, bad_shape = 0, rows = 0, rows_not_s0 = 0, rows_streams_differ = 0;
        bool nonzero = false;
        for (const auto & r : m.recs) {
            const size_t want = (size_t) n_embd*r.n_tokens;
            if (r.streams[il].size() != want*hc || r.api[il].size() != want || r.node[il].size() != want) {
                bad_shape++;
                continue;
            }
            const auto mean = host_mean(r.streams[il], n_embd, hc, r.n_tokens);
            bad_api  += n_differ(r.api[il], mean);
            bad_node += n_differ(r.node[il], mean);
            for (int t = 0; t < r.n_tokens; t++) {
                const float * s0 = r.streams[il].data() + (size_t) t*hc*n_embd;
                bool not_s0 = false, differ = false;
                for (int64_t k = 0; k < n_embd; k++) {
                    const float x = mean[(size_t) t*n_embd + k];
                    nonzero = nonzero || x != 0.0f;
                    not_s0  = not_s0  || x != s0[k];
                    for (int64_t s = 1; s < hc; s++) { differ = differ || s0[s*n_embd + k] != s0[k]; }
                }
                rows++;
                rows_not_s0         += not_s0;
                rows_streams_differ += differ;
            }
        }
        const std::string f = "floor " + std::to_string(il) + ": ";
        check(bad_shape == 0 && rows == N_READ + N_WRITE, f + "streams, API and node all " + std::to_string(rows) +
                " rows, n_embd wide (" + std::to_string(bad_shape) + " decodes off)");
        check(bad_api == 0 && nonzero, f + "llama_get_embeddings_layer_inp is the mean of the 4 streams, byte for byte (" +
                std::to_string(bad_api) + " numbers differ)");
        check(bad_node == 0, f + "the layer_inp node is that mean too (" + std::to_string(bad_node) + " differ)");
        if (il == 0) {
            check(rows_streams_differ == 0 && rows_not_s0 == 0, f + "the streams start as copies, so the mean is stream 0");
        } else {
            // a row whose combine weights all saturate keeps 4 equal copies; most must not, or stream 0 alone would pass
            check(rows_not_s0 >= rows*3/4 && rows_not_s0 <= rows_streams_differ, f + "the streams differ on " +
                    std::to_string(rows_streams_differ) + "/" + std::to_string(rows) + " rows, the mean is not stream 0 on " +
                    std::to_string(rows_not_s0));
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    bool off_only = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            g_path = argv[++i];
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            g_ngl = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--off-only") == 0) {
            off_only = true;
        }
    }
    if (g_path.empty()) {
        fprintf(stderr, "usage: %s -m qwen4exp-moe-e64mtp.gguf [-ngl N] [--off-only]\n", argv[0]);
        return 1;
    }
    // TEST_LAYER_INP_LOG=1 prints every engine line (with GGML_METAL_GRAPH_DEBUG=1, the Metal fusion fingerprint)
    static const bool log_all = getenv("TEST_LAYER_INP_LOG") != nullptr;
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (log_all || level == GGML_LOG_LEVEL_ERROR) {
            fputs(text, log_all ? stdout : stderr);
        }
    }, nullptr);
    ggml_backend_load_all();
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers    = g_ngl;
    mp.load_mtp        = true; // the fixture's MTP block, or the loader's tensor count fails
    mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(g_path.c_str(), mp);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", g_path.c_str());
        return 1;
    }
    const int64_t n_embd = llama_model_n_embd(model);
    const int64_t hc     = llama_model_n_embd_out(model)/n_embd;

    printf("off: no floor asked for\n");
    const run_out off = run(model, {}, {}, false);
    check(off.w.built[0] == 0 && off.w.built[1] == 0, "no layer_inp node built");
    printf("  off hashes: nodes %016llx (%lld), logits %016llx, nextn %016llx\n", (unsigned long long) off.w.nodes,
            (long long) off.w.n_nodes, (unsigned long long) hash_of(off, &record::logits),
            (unsigned long long) hash_of(off, &record::nextn));
    if (!off_only) {
        printf("unseen: floor 1 asked for, the graph unsplit\n");
        const run_out unseen = run(model, { 1 }, {}, false);
        check(hc == 4, "the fixture keeps 4 hc streams (" + std::to_string(hc) + ")");
        check(unseen.w.built[1] == 6 && unseen.w.built[0] == 0, "one layer_inp-1 node a graph (" +
                std::to_string(unseen.w.built[1]) + "/6), none for floor 0, which was not asked for (" +
                std::to_string(unseen.w.built[0]) + ")");
        check(unseen.w.wide, "every layer_inp node is n_embd wide");
        check(same(unseen, off, &record::logits), "logits byte-identical to off");
        check(same(unseen, off, &record::nextn), "the apprentice's notes (t_h_nextn) byte-identical to off");

        printf("mean: floors 0 and 1 asked for, the streams captured\n");
        const run_out mean = run(model, { 0, 1 }, {}, true);
        check(mean.w.wide, "every layer_inp node is n_embd wide");
        check_mean(mean, n_embd, hc);
        check(same(mean, unseen, &record::logits), "logits byte-identical to unseen");
        bool api_same = true;
        for (size_t d = 0; d < mean.recs.size(); d++) {
            api_same = api_same && !mean.recs[d].api[1].empty() &&
                    n_differ(mean.recs[d].api[1], unseen.recs[d].api[1]) == 0;
        }
        check(api_same, "floor 1's numbers byte-identical to unseen's (watching does not change them)");

        // the last floor drops the rows that produce no output after the floor's input is taken: floor 1's input
        // must still be every row, the same numbers as when every row is an output
        printf("last-only: the read-in asks for the last word's logits only\n");
        const run_out lo = run(model, { 0, 1 }, {}, true, true);
        check_mean(lo, n_embd, hc);
        bool lo_same = true;
        for (size_t d = 0; d < lo.recs.size(); d++) {
            lo_same = lo_same && n_differ(lo.recs[d].api[1], mean.recs[d].api[1]) == 0 && !lo.recs[d].api[1].empty();
        }
        check(lo_same, "floor 1's numbers byte-identical to the all-rows run's");

        printf("toggle: floor 1 asked for only after the read-in\n");
        const run_out tog = run(model, {}, { 1 }, false);
        bool tog_same = tog.recs[0].api[1].empty();
        for (int d = 1; d <= N_WRITE; d++) {
            tog_same = tog_same && n_differ(tog.recs[d].api[1], unseen.recs[d].api[1]) == 0 &&
                    !tog.recs[d].api[1].empty();
        }
        check(tog.w.built[1] == N_WRITE, "built only for the words written (" + std::to_string(tog.w.built[1]) + ")");
        check(tog_same, "the words written hand back unseen's numbers");
        check(same(tog, off, &record::logits), "logits byte-identical to off");
    }
    llama_model_free(model);

    if (g_failures) {
        fprintf(stderr, "test-qwen4exp-layer-inp: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("test-qwen4exp-layer-inp: all checks passed\n");
    return 0;
}

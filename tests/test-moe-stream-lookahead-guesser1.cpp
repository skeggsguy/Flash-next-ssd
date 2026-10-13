// The trained one-floor guess (LLAMA_MOE_STREAM_GUESSER1, #43, src/llama-moe-stream-guesser.h) on the four-floor
// 64-book fixture, with every other guess on as the everyday library runs them (serve.py GUESSER_ENV: two floors
// ahead, every word of a check, the keep rule, and in the "both" arm the trained two-floor guesser at the cut 0.2).
// A guess only says which books are fetched, never which book a token reads, so the check is byte for byte: a
// 20-token prompt, 40 single written tokens, 20 checks of 4 tokens, a 40-token read-in (the reading room) and 10 more
// single tokens give the same logits and the same MoE output at every floor as no streaming, in every arm:
//   off       GUESSER1 unset: today's borrowed guess; no trained stop counted; the startup line says "borrowed"
//   random    a random one-floor guesser (F16, as the study exports it): trained stops counted, books fetched one
//             floor ahead; the startup line says "trained"
//   identity  a guesser whose floor n line is floor n-1's own router (F32) plus a bias: its scores at floor n-1,
//             captured from the graph (ffn_moe_guess1-n), are floor n-1's router scores (ffn_moe_logits-n) plus the
//             bias, byte for byte: the graph uses the guesser's matrix, its bias, and floor n-1's input for floor n
//   both      the one-floor and the two-floor guessers together: both count their stops, both startup lines
//   sidecar   the lookahead off (LLAMA_MOE_STREAM_LOOKAHEAD=0, a model with no floor guessing ahead, as the
//             apprentice's sidecar is) with both guessers set: it loads, nothing attached, and runs
//   refusals  a two-floor file in GUESSER1, a file of another width, a missing file: each fails the load

#include "test-moe-room-model.h"

#include "gguf.h"

#include <map>
#include <unistd.h>

namespace {

std::string g_la_log; // the "one-floor guess" and "two-floor guess" startup lines of the arm's load

struct capture {
    outputs out;
    std::map<int, std::vector<float>> logits;  // floor -> its last router scores
    std::vector<float> bias;                    // the identity guesser's bias
    int64_t guesses = 0, guesses_same = 0;      // ffn_moe_guess1 tensors seen, and equal to logits + bias
};

int floor_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] < '0' || name[n] > '9') {
        return -1;
    }
    char * end = nullptr;
    const long il = strtol(name + n, &end, 10);
    return *end == '\0' ? (int) il : -1;
}

bool capture_cb(ggml_tensor * t, bool ask, void * user_data) {
    const int il_moe = floor_of(t->name, "ffn_moe_out-");
    const int il_log = floor_of(t->name, "ffn_moe_logits-");
    const int il_gue = floor_of(t->name, "ffn_moe_guess1-");
    auto * c = (capture *) user_data;
    if (ask) {
        return il_moe >= 0 || il_gue >= 0 || (il_log >= 0 && !c->bias.empty());
    }
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    if (il_moe >= 0) {
        c->out.moe.insert(c->out.moe.end(), v.begin(), v.end());
    } else if (il_log >= 0) {
        c->logits[il_log] = v;
    } else if (il_gue >= 0) {
        c->guesses++;
        if (c->bias.empty()) {
            return true;
        }
        const auto & l = c->logits[il_gue];
        bool same = l.size() == v.size() && !v.empty();
        for (size_t i = 0; same && i < v.size(); i++) {
            const float want = l[i] + c->bias[i % c->bias.size()];
            same = memcmp(&want, &v[i], sizeof(float)) == 0;
        }
        c->guesses_same += same;
    }
    return true;
}

bool run_writing(llama_model * model, capture & cap) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 1024;
    cp.n_batch    = 512;
    cp.n_ubatch   = 512;
    cp.op_offload = false;
    cp.cb_eval           = capture_cb;
    cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        return false;
    }
    std::mt19937 rng(13);
    std::vector<size_t> sizes = { 20 };
    sizes.insert(sizes.end(), 40, 1);
    sizes.insert(sizes.end(), 20, 4);
    sizes.push_back(40);
    sizes.insert(sizes.end(), 10, 1);
    size_t total = 0;
    for (const size_t n : sizes) {
        total += n;
    }
    const std::vector<llama_token> toks = random_tokens(rng, total, n_vocab);
    llama_batch batch = llama_batch_init(512, 0, 1);
    bool ok = true;
    size_t pos = 0;
    for (size_t b = 0; ok && b < sizes.size(); b++) {
        common_batch_clear(batch);
        for (size_t i = 0; i < sizes[b]; i++, pos++) {
            common_batch_add(batch, toks[pos], (llama_pos) pos, { 0 }, true);
        }
        ok = llama_decode(ctx, batch) == 0;
        if (ok) {
            const float * l = llama_get_logits(ctx);
            cap.out.logits.insert(cap.out.logits.end(), l, l + sizes[b]*n_vocab);
        }
    }
    llama_batch_free(batch);
    llama_free(ctx);
    return ok;
}

// a guesser file at floor offset `gap` for floors gap..n_layer-1: random (F16 weights near 0, F32 biases spread),
// or (gap 1) floor n's line = floor n-1's own router (F32) plus `bias`
std::string write_guesser(const std::string & name, const llama_model * ref, uint32_t gap, bool identity,
        const std::vector<float> & bias, int64_t n_embd_key) {
    const int32_t n_layer = (int32_t) ref->layers.size();
    const int64_t n_embd = ref->hparams.n_embd, n_expert = ref->hparams.n_expert;
    gguf_context * g = gguf_init_empty();
    gguf_set_val_u32(g, "lookahead.floor_offset", gap);
    gguf_set_val_u32(g, "lookahead.first_floor", gap);
    gguf_set_val_u32(g, "lookahead.last_floor", (uint32_t) n_layer - 1);
    gguf_set_val_u32(g, "lookahead.n_embd", (uint32_t) n_embd_key);
    gguf_set_val_u32(g, "lookahead.n_expert", (uint32_t) n_expert);
    gguf_set_val_f32(g, "lookahead.confidence.scale", 10.0f);
    gguf_set_val_f32(g, "lookahead.confidence.max", 1.0f);
    ggml_init_params ip = { 16*ggml_tensor_overhead() + (size_t) n_layer*(n_embd*n_expert*4 + n_expert*4 + 256), nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(5 + gap);
    std::normal_distribution<float> small(0.0f, 0.02f), spread(0.0f, 2.0f);
    for (int32_t n = (int32_t) gap; n < n_layer; n++) {
        const ggml_tensor * router = ref->layers[n - gap].ffn_gate_inp;
        ggml_tensor * w = ggml_new_tensor_2d(ctx, identity ? router->type : GGML_TYPE_F16, n_embd, n_expert);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
        ggml_set_name(w, ("blk." + std::to_string(n) + ".la_guess.weight").c_str());
        ggml_set_name(b, ("blk." + std::to_string(n) + ".la_guess.bias").c_str());
        if (identity) {
            ggml_backend_tensor_get(router, w->data, 0, ggml_nbytes(w));
            memcpy(b->data, bias.data(), ggml_nbytes(b));
        } else {
            for (int64_t i = 0; i < n_embd*n_expert; i++) {
                ((ggml_fp16_t *) w->data)[i] = ggml_fp32_to_fp16(small(rng));
            }
            for (int64_t e = 0; e < n_expert; e++) {
                ((float *) b->data)[e] = spread(rng);
            }
        }
        gguf_add_tensor(g, w);
        gguf_add_tensor(g, b);
    }
    const std::string path = (std::filesystem::temp_directory_path() /
            ("test-moe-stream-lookahead-guesser1-" + std::to_string(getpid()) + "-" + name + ".gguf")).string();
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

struct counts {
    int64_t stops1 = 0, stops2 = 0, la1_issued = 0, la2_issued = 0;
    bool loaded = false, has1 = false, has2 = false;
    std::string log;
};

void set(const char * name, const char * v) {
    v ? setenv(name, v, 1) : unsetenv(name);
}

counts run_arm(const char * guesser1, const char * guesser2, capture & cap) {
    set("LLAMA_MOE_STREAM_GUESSER1", guesser1);
    set("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER", guesser2);
    set("LLAMA_MOE_STREAM_LOOKAHEAD_CONF", guesser2 ? "0.2" : nullptr);
    counts c;
    g_la_log.clear();
    llama_model * m = load(config());
    c.log = g_la_log;
    if (m == nullptr) {
        return c;
    }
    c.loaded = n_streamed_floors(m) >= 3 && run_writing(m, cap);
    llama_moe_stream * ms = m->moe_stream();
    {
        std::unique_lock<std::mutex> lk(ms->mtx);
        c.stops1 = ms->stats.n_guess1_stops; c.stops2 = ms->stats.n_guess_stops;
        c.la1_issued = ms->stats.n_la1_issued; c.la2_issued = ms->stats.n_la2_issued;
    }
    c.has1 = ms->guesser1 != nullptr;
    c.has2 = ms->guesser != nullptr;
    llama_model_free(m);
    return c;
}

std::string str(const counts & c) {
    char buf[200];
    snprintf(buf, sizeof(buf), "trained stops: one floor %lld, two floors %lld; fetched: one floor %lld, two floors %lld",
            (long long) c.stops1, (long long) c.stops2, (long long) c.la1_issued, (long long) c.la2_issued);
    return buf;
}

bool says(const counts & c, const std::string & what) {
    return c.log.find(what) != std::string::npos;
}

} // namespace

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    // silent, keeping the guessers' startup lines
    llama_log_set([](ggml_log_level, const char * text, void *) {
        if (strstr(text, "one-floor guess") != nullptr || strstr(text, "two-floor guess") != nullptr) {
            g_la_log += text;
        }
    }, nullptr);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", "4", 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_ALL", "1", 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_KEEP", "1", 1);
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD"); // the default width, as in every rung

    const int64_t n_expert = ref_model->hparams.n_expert;
    std::vector<float> bias(n_expert, 0.0f); // book 63 far over the rest, books 40..62 over the others
    for (int64_t e = n_expert - 24; e < n_expert - 1; e++) {
        bias[e] = 500.0f;
    }
    bias[n_expert - 1] = 1000.0f;
    const std::string random1 = write_guesser("random1", ref_model, 1, false, {}, ref_model->hparams.n_embd);
    const std::string ident1  = write_guesser("identity1", ref_model, 1, true, bias, ref_model->hparams.n_embd);
    const std::string wide1   = write_guesser("wide1", ref_model, 1, false, {}, ref_model->hparams.n_embd*2);
    const std::string random2 = write_guesser("random2", ref_model, 2, false, {}, ref_model->hparams.n_embd);
    const bool router_f32 = ref_model->layers[0].ffn_gate_inp->type == GGML_TYPE_F32;

    capture ref, off, rnd, ident, both, side;
    check(run_writing(ref_model, ref), "no streaming: the tokens ran");
    llama_model_free(ref_model);

    const counts c_off  = run_arm(nullptr, nullptr, off);
    const counts c_rnd  = run_arm(random1.c_str(), nullptr, rnd);
    ident.bias = bias;
    const counts c_ident = run_arm(ident1.c_str(), nullptr, ident);
    const counts c_both = run_arm(random1.c_str(), random2.c_str(), both);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD", "0", 1); // no floor guesses ahead: the apprentice's sidecar's case
    const counts c_side = run_arm(random1.c_str(), random2.c_str(), side);
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD");
    check(c_off.loaded && c_rnd.loaded && c_ident.loaded && c_both.loaded && c_side.loaded, "every arm loaded and its tokens ran");

    check(nonzero(ref.out.logits) && nonzero(ref.out.moe), "the reference logits and MoE outputs are not all zero");
    for (const auto & [name, got] : { std::pair<const char *, const capture *>{ "off", &off }, { "random one-floor", &rnd },
            { "identity one-floor", &ident }, { "both guessers", &both }, { "sidecar case", &side } }) {
        check(same(got->out.logits, ref.out.logits) && same(got->out.moe, ref.out.moe),
                std::string(name) + ": logits and every ffn_moe_out byte-identical to no streaming");
    }

    printf("  off:      %s\n  random:   %s\n  identity: %s\n  both:     %s\n  sidecar:  %s\n", str(c_off).c_str(),
            str(c_rnd).c_str(), str(c_ident).c_str(), str(c_both).c_str(), str(c_side).c_str());
    check(!c_off.has1 && c_off.stops1 == 0 && c_off.la1_issued > 0 && off.guesses == 0,
            "off: the borrowed guess fetches, no trained stop, no ffn_moe_guess1 in the graph");
    check(says(c_off, "one-floor guess: borrowed"), "off: the startup line says the one-floor guess is borrowed");
    check(c_rnd.has1 && c_rnd.stops1 > 0 && c_rnd.la1_issued > 0 && !c_rnd.has2 && c_rnd.stops2 == 0,
            "random: the trained one-floor guess scored stops and fetched; the two-floor guess untrained");
    check(rnd.guesses > 0, "random: ffn_moe_guess1 ran in the graph (" + std::to_string(rnd.guesses) + ")");
    check(says(c_rnd, "one-floor guess: trained, the guesser " + random1), "random: the startup line names the trained guesser's file");
    check(router_f32, "the fixture's routers are F32, so the identity guesser's matrix is their bytes");
    check(ident.guesses > 0 && ident.guesses_same == ident.guesses, "identity: every ffn_moe_guess1-n equals ffn_moe_logits-n "
            "+ bias byte for byte (" + std::to_string(ident.guesses_same) + " of " + std::to_string(ident.guesses) + ")");
    check(c_both.has1 && c_both.has2 && c_both.stops1 > 0 && c_both.stops2 > 0 && c_both.la1_issued > 0 && c_both.la2_issued > 0,
            "both: each guesser scored stops and fetched");
    check(says(c_both, "one-floor guess: trained") && says(c_both, "two-floor guess: the trained guesser " + random2),
            "both: both startup lines");
    check(!c_side.has1 && !c_side.has2 && c_side.stops1 == 0 && c_side.la1_issued == 0,
            "sidecar case: both switches set, nothing attached, nothing guessed");
    check(says(c_side, "one-floor guess: the trained guesser " + random1 + " is not loaded here"),
            "sidecar case: the startup line says the one-floor guesser is not loaded there");

    // refusals: each must fail the load, not run on something else
    auto refuses = [](const char * guesser1) {
        set("LLAMA_MOE_STREAM_GUESSER1", guesser1);
        llama_model * m = load(config());
        llama_model_free(m);
        return m == nullptr;
    };
    check(refuses(random2.c_str()), "a two-floor file in GUESSER1 refuses to load");
    check(refuses(wide1.c_str()), "a one-floor guesser of another width refuses to load");
    check(refuses("/nonexistent/guesser1.gguf"), "a missing one-floor guesser file refuses");

    for (const char * e : { "LLAMA_MOE_STREAM_GUESSER1", "LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER", "LLAMA_MOE_STREAM_LOOKAHEAD_CONF",
            "LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", "LLAMA_MOE_STREAM_LOOKAHEAD_ALL", "LLAMA_MOE_STREAM_LOOKAHEAD_KEEP" }) {
        unsetenv(e);
    }
    for (const auto & p : { random1, ident1, wide1, random2 }) {
        std::filesystem::remove(p);
    }
    return finish();
}

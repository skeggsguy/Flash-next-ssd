// The trained two-floor guesser (LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER and _CONF, src/llama-moe-stream-guesser.h) on
// the four-floor 64-book fixture, with every guess on as the K-c1 rung runs it (two floors ahead, every word of a
// check, the keep rule). A guess only says which books are fetched, never which book a token reads, so the check
// is byte for byte: a 20-token prompt, 40 single written tokens, 20 checks of 4 tokens, a 40-token read-in (the
// reading room) and 10 more single tokens give the same logits and the same MoE output at every floor as no
// streaming, with the guesser off, on with the fixed pick, and on with the confidence pick.
//
// And it must be seen working, or this would pass by guessing nothing:
//   - a random guesser (F16, as the study exports it) scores words at every floor stop that guesses two ahead;
//     at the cut 0.2 some of each word's 24 best are picked and some left under the cut;
//   - a guesser whose floor n+2 line is floor n's own router (F32) plus a bias of +1000 on book 63 and +500 on
//     books 40..62: its scores at floor n, captured from the graph (ffn_moe_guess-n), are floor n's router scores
//     (ffn_moe_logits-n) plus the bias, byte for byte, so the graph uses the guesser's matrix, its bias, and floor
//     n's input for floor n+2; and every word's 24 best are those books, of which book 63 alone reaches the cut,
//     so every floor stop picks exactly 1 book and leaves 23 under it;
//   - a file that does not fit the library, CONF without GUESSER, and a guesser with no two-floor guess refuse
//     to load.

#include "test-moe-room-model.h"

#include "gguf.h"

#include <map>
#include <unistd.h>

namespace {

struct capture {
    outputs out;
    std::map<int, std::vector<float>> logits;  // floor -> its last router scores
    std::vector<float> bias;                    // the identity guesser's bias
    int64_t guesses = 0, guesses_same = 0;      // ffn_moe_guess tensors seen, and equal to logits + bias
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
    const int il_gue = floor_of(t->name, "ffn_moe_guess-");
    auto * c = (capture *) user_data;
    if (ask) {
        return il_moe >= 0 || ((il_log >= 0 || il_gue >= 0) && !c->bias.empty());
    }
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    if (il_moe >= 0) {
        c->out.moe.insert(c->out.moe.end(), v.begin(), v.end());
    } else if (il_log >= 0) {
        c->logits[il_log] = v;
    } else if (il_gue >= 0) {
        c->guesses++;
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

// a guesser file for floors 2..n_layer-1: random (F16 weights near 0, F32 biases spread so confidences straddle
// the cut), or floor n's line = floor n-2's own router (F32) plus `bias`
std::string write_guesser(const std::string & name, const llama_model * ref, bool identity,
        const std::vector<float> & bias, int64_t n_embd_key) {
    const int32_t n_layer = (int32_t) ref->layers.size();
    const int64_t n_embd = ref->hparams.n_embd, n_expert = ref->hparams.n_expert;
    gguf_context * g = gguf_init_empty();
    gguf_set_val_u32(g, "lookahead.floor_offset", 2);
    gguf_set_val_u32(g, "lookahead.first_floor", 2);
    gguf_set_val_u32(g, "lookahead.last_floor", (uint32_t) n_layer - 1);
    gguf_set_val_u32(g, "lookahead.n_embd", (uint32_t) n_embd_key);
    gguf_set_val_u32(g, "lookahead.n_expert", (uint32_t) n_expert);
    gguf_set_val_f32(g, "lookahead.confidence.scale", 10.0f);
    gguf_set_val_f32(g, "lookahead.confidence.max", 1.0f);
    ggml_init_params ip = { 16*ggml_tensor_overhead() + (size_t) n_layer*(n_embd*n_expert*4 + n_expert*4 + 256), nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(5);
    std::normal_distribution<float> small(0.0f, 0.02f), spread(0.0f, 2.0f);
    for (int32_t n = 2; n < n_layer; n++) {
        const ggml_tensor * router = ref->layers[n - 2].ffn_gate_inp;
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
            ("test-moe-stream-lookahead-guesser-" + std::to_string(getpid()) + "-" + name + ".gguf")).string();
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

struct counts {
    int64_t stops = 0, words = 0, picked = 0, under = 0, la2_issued = 0, la2_used = 0;
    bool loaded = false;
};

counts run_arm(const char * guesser, const char * conf, capture & cap) {
    guesser ? setenv("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER", guesser, 1) : unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER");
    conf ? setenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF", conf, 1) : unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF");
    counts c;
    llama_model * m = load(config());
    if (m == nullptr) {
        return c;
    }
    c.loaded = n_streamed_floors(m) >= 3 && run_writing(m, cap);
    llama_moe_stream * ms = m->moe_stream();
    {
        std::unique_lock<std::mutex> lk(ms->mtx);
        c.stops = ms->stats.n_guess_stops; c.words = ms->stats.n_guess_words;
        c.picked = ms->stats.n_guess_picked; c.under = ms->stats.n_guess_under;
        c.la2_issued = ms->stats.n_la2_issued; c.la2_used = ms->stats.n_la2_used;
    }
    llama_model_free(m);
    return c;
}

std::string str(const counts & c) {
    char buf[200];
    snprintf(buf, sizeof(buf), "stops %lld, words %lld, picked %lld, under %lld; two floors: fetched %lld, used %lld",
            (long long) c.stops, (long long) c.words, (long long) c.picked, (long long) c.under,
            (long long) c.la2_issued, (long long) c.la2_used);
    return buf;
}

} // namespace

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", "4", 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_ALL", "1", 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_KEEP", "1", 1);
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD"); // the default width, as in every rung

    const int64_t n_expert = ref_model->hparams.n_expert;
    std::vector<float> bias(n_expert, 0.0f); // every word's 24 best: book 63, then books 40..62 far under it
    for (int64_t e = n_expert - 24; e < n_expert - 1; e++) {
        bias[e] = 500.0f;
    }
    bias[n_expert - 1] = 1000.0f;
    const std::string random_path = write_guesser("random", ref_model, false, {}, ref_model->hparams.n_embd);
    const std::string ident_path  = write_guesser("identity", ref_model, true, bias, ref_model->hparams.n_embd);
    const std::string wide_path   = write_guesser("wide", ref_model, false, {}, ref_model->hparams.n_embd*2);
    const bool router_f32 = ref_model->layers[0].ffn_gate_inp->type == GGML_TYPE_F32;

    capture ref, off, fixed, conf, ident;
    check(run_writing(ref_model, ref), "no streaming: the tokens ran");
    llama_model_free(ref_model);

    const counts c_off   = run_arm(nullptr, nullptr, off);
    const counts c_fixed = run_arm(random_path.c_str(), "0", fixed); // "0": the fixed pick (unset is 0.2 beside a guesser)
    const counts c_conf  = run_arm(random_path.c_str(), "0.2", conf);
    ident.bias = bias;
    const counts c_ident = run_arm(ident_path.c_str(), "0.2", ident);
    check(c_off.loaded && c_fixed.loaded && c_conf.loaded && c_ident.loaded, "every arm loaded and its tokens ran");

    check(nonzero(ref.out.logits) && nonzero(ref.out.moe), "the reference logits and MoE outputs are not all zero");
    for (const auto & [name, got] : { std::pair<const char *, const capture *>{ "guesser off", &off },
            { "guesser, fixed pick", &fixed }, { "guesser, cut 0.2", &conf }, { "identity guesser, cut 0.2", &ident } }) {
        check(same(got->out.logits, ref.out.logits) && same(got->out.moe, ref.out.moe),
                std::string(name) + ": logits and every ffn_moe_out byte-identical to no streaming");
    }

    printf("  off:      %s\n  fixed:    %s\n  cut 0.2:  %s\n  identity: %s\n", str(c_off).c_str(),
            str(c_fixed).c_str(), str(c_conf).c_str(), str(c_ident).c_str());
    check(c_off.stops == 0 && c_off.words == 0 && c_off.la2_issued > 0, "off: the untrained two-floor guess, no guesser counts");
    check(c_fixed.stops > 0 && c_fixed.words > c_fixed.stops && c_fixed.picked >= 4*c_fixed.stops && c_fixed.under == 0,
            "fixed pick: the guesser scored every word of the checks, at least top_k2 books a stop, none cut");
    check(c_conf.stops > 0 && c_conf.words > c_conf.stops && c_conf.picked > 0 && c_conf.under > 0 &&
            c_conf.picked < 24*c_conf.words, "cut 0.2: some books picked, some left under the cut");
    check(c_fixed.la2_issued > 0 && c_conf.la2_issued > 0, "both guesser arms fetched books two floors ahead");
    check(router_f32, "the fixture's routers are F32, so the identity guesser's matrix is their bytes");
    check(ident.guesses > 0 && ident.guesses_same == ident.guesses, "identity: every ffn_moe_guess-n equals ffn_moe_logits-n "
            "+ bias byte for byte (" + std::to_string(ident.guesses_same) + " of " + std::to_string(ident.guesses) + ")");
    check(c_ident.stops > 0 && c_ident.picked == c_ident.stops && c_ident.under == 23*c_ident.stops,
            "identity: book 63 alone reaches the cut: 1 picked and 23 under it at every stop");

    // refusals: each must fail the load, not run on something else
    auto refuses = [](const char * guesser, const char * conf, const char * depth2) {
        guesser ? setenv("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER", guesser, 1) : unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER");
        conf ? setenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF", conf, 1) : unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF");
        depth2 ? setenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", depth2, 1) : unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2");
        llama_model * m = load(config());
        llama_model_free(m);
        return m == nullptr;
    };
    check(refuses(wide_path.c_str(), "0.2", "4"), "a guesser of another width refuses to load");
    check(refuses(nullptr, "0.2", "4"), "CONF without GUESSER refuses to load");
    check(refuses(random_path.c_str(), "abc", "4"), "a CONF that is not a cut refuses to load");
    check(refuses(random_path.c_str(), nullptr, "0"), "a guesser without the two-floor guess (DEPTH2=0) refuses");
    check(!refuses(random_path.c_str(), nullptr, nullptr), "a guesser with DEPTH2 and CONF unset loads (4 and 0.2 beside it)");
    check(refuses("/nonexistent/guesser.gguf", nullptr, "4"), "a missing guesser file refuses");

    for (const char * e : { "LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER", "LLAMA_MOE_STREAM_LOOKAHEAD_CONF",
            "LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", "LLAMA_MOE_STREAM_LOOKAHEAD_ALL", "LLAMA_MOE_STREAM_LOOKAHEAD_KEEP" }) {
        unsetenv(e);
    }
    for (const auto & p : { random_path, ident_path, wide_path }) {
        std::filesystem::remove(p);
    }
    return finish();
}

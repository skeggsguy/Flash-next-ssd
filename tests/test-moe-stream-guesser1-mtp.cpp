// The trained one-floor guess (LLAMA_MOE_STREAM_GUESSER1, #43) beside the apprentice, on the MTP fixture
// (qwen4exp-moe-e64mtp.gguf: two trunk floors and one MTP block, 64 books of 8), streamed. A library and an
// apprentice context on one model, driven as common/speculative.cpp's MTP drafter drives them: the library reads a
// prompt in, the apprentice reads it in and drafts 3 words, the library checks the 4 (the first word and the draft),
// the apprentice drafts 2 more. Every guess switch the everyday library sets is on (serve.py GUESSER_ENV: every word
// of a check, two floors ahead, the keep rule, the trained two-floor guesser at the cut 0.2), so the apprentice's
// context opens and runs with both trained guessers set, as the server's would (gotchas: a load-time switch reaches
// the apprentice). Guesses never change words, so with GUESSER1 and GUESSER on:
//   same     the library's logits (prompt and check) and every draft's logits are byte-identical to both off
//   ran      both trained guesses scored floor stops (the one-floor counted at the library's floors, the two-floor too)

#include "common.h"
#include "llama.h"

#include "../src/llama-ext.h"
#include "../src/llama-model.h"
#include "../src/llama-moe-room.h"
#include "../src/llama-moe-stream.h"

#include "gguf.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

std::string g_path;
int         g_ngl      = 99;
int         g_failures = 0;

void check(bool ok, const std::string & what) {
    printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

llama_model * load(bool stream) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers    = g_ngl;
    mp.load_mtp        = true;
    mp.use_extra_bufts = false;
    if (stream) {
        mp.moe_stream            = true;
        mp.moe_stream_slots      = 55;
        mp.moe_stream_room_mode  = LLAMA_MOE_ROOM_AUTO;
        mp.moe_stream_room_parts = 4;
    }
    return llama_model_load_from_file(g_path.c_str(), mp);
}

llama_context * make_ctx(llama_model * model, bool mtp, llama_context * other) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 512;
    cp.n_batch    = 512;
    cp.n_ubatch   = 128;
    cp.n_seq_max  = 1;
    cp.op_offload = false; // streaming turns it off
    if (mtp) {
        cp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
        cp.ctx_other     = other;
        cp.n_outputs_max = 1;
        cp.n_rs_seq      = 0;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx != nullptr) {
        llama_set_embeddings_nextn(ctx, true, /*masked*/ mtp);
    }
    return ctx;
}

llama_token argmax(const float * row, int n) {
    llama_token best = 0;
    for (int v = 1; v < n; v++) {
        best = row[v] > row[best] ? v : best;
    }
    return best;
}

struct turn {
    bool               ran = false;
    std::vector<float> library;  // the library's logits: the prompt's last row, then the check's 4 rows
    std::vector<float> drafts;   // every draft step's logits
    int64_t stops1 = 0, stops2 = 0;
    bool    has1 = false, has2 = false;
};

// the apprentice drafts n words from the library's hidden state h after word id at position pos
bool draft(llama_context * dft, int n_embd, int n_vocab, llama_token id, int pos, std::vector<float> h, int n,
        std::vector<llama_token> & words, turn & r) {
    llama_batch b = llama_batch_init(1, n_embd, 1);
    b.token = (llama_token *) malloc(sizeof(llama_token));
    bool ok = true;
    for (int i = 0; ok && i < n; i++) {
        common_batch_clear(b);
        common_batch_add(b, id, pos + i, { 0 }, true);
        memcpy(b.embd, h.data(), n_embd*sizeof(float));
        ok = llama_decode(dft, b) == 0;
        const float * l  = ok ? llama_get_logits_ith(dft, 0) : nullptr;
        const float * hn = ok ? llama_get_embeddings_nextn_ith(dft, 0) : nullptr;
        ok = ok && l != nullptr && hn != nullptr;
        if (ok) {
            r.drafts.insert(r.drafts.end(), l, l + n_vocab);
            h.assign(hn, hn + n_embd);
            id = argmax(l, n_vocab);
            words.push_back(id);
        }
    }
    llama_batch_free(b);
    return ok;
}

turn run_turn(const std::vector<llama_token> & prompt) {
    turn r;
    llama_model * model = load(true);
    llama_context * tgt = model ? make_ctx(model, false, nullptr) : nullptr;
    llama_context * dft = tgt ? make_ctx(model, true, tgt) : nullptr;
    bool ok = dft != nullptr;
    const int n_embd  = model ? llama_model_n_embd_out(model) : 0;
    const int n_vocab = model ? llama_vocab_n_tokens(llama_model_get_vocab(model)) : 0;
    const int n       = (int) prompt.size();

    llama_batch b = llama_batch_init(512, 0, 1);
    std::vector<float> h_tgt;
    llama_token id = 0;
    if (ok) { // the library reads the prompt in
        for (int i = 0; i < n; i++) {
            common_batch_add(b, prompt[i], i, { 0 }, i == n - 1);
        }
        ok = llama_decode(tgt, b) == 0 && llama_get_embeddings_nextn(tgt) != nullptr;
    }
    if (ok) {
        const float * l = llama_get_logits_ith(tgt, -1);
        r.library.assign(l, l + n_vocab);
        id = argmax(l, n_vocab);
        h_tgt.assign(llama_get_embeddings_nextn(tgt), llama_get_embeddings_nextn(tgt) + (size_t) n*n_embd);
    }
    if (ok) { // the apprentice reads it in with no outputs: row 0 a zero state, row k the library's k-1
        llama_batch mb = llama_batch_init(n, n_embd, 1);
        mb.token = (llama_token *) malloc(sizeof(llama_token)*n);
        common_batch_clear(mb);
        for (int k = 0; k < n; k++) {
            common_batch_add(mb, prompt[k], k, { 0 }, false);
            if (k == 0) {
                memset(mb.embd, 0, n_embd*sizeof(float));
            } else {
                memcpy(mb.embd + (size_t) k*n_embd, h_tgt.data() + (size_t) (k - 1)*n_embd, n_embd*sizeof(float));
            }
        }
        ok = llama_decode(dft, mb) == 0;
        llama_batch_free(mb);
    }
    std::vector<llama_token> words;
    ok = ok && draft(dft, n_embd, n_vocab, id, n, std::vector<float>(h_tgt.end() - n_embd, h_tgt.end()), 3, words, r);
    if (ok) { // the library checks the first word and the draft: a 4-word batch, every row's logits
        common_batch_clear(b);
        common_batch_add(b, id, n, { 0 }, true);
        for (int i = 0; i < 3; i++) {
            common_batch_add(b, words[i], n + 1 + i, { 0 }, true);
        }
        ok = llama_decode(tgt, b) == 0 && llama_get_embeddings_nextn(tgt) != nullptr;
    }
    if (ok) {
        const float * l = llama_get_logits(tgt);
        r.library.insert(r.library.end(), l, l + 4*n_vocab);
        const float * h = llama_get_embeddings_nextn(tgt) + (size_t) 3*n_embd;
        std::vector<llama_token> more;
        ok = draft(dft, n_embd, n_vocab, argmax(l + 3*n_vocab, n_vocab), n + 4, std::vector<float>(h, h + n_embd), 2, more, r);
    }
    if (model != nullptr && model->moe_stream() != nullptr) {
        llama_moe_stream * ms = model->moe_stream();
        std::unique_lock<std::mutex> lk(ms->mtx);
        r.stops1 = ms->stats.n_guess1_stops;
        r.stops2 = ms->stats.n_guess_stops;
        r.has1   = ms->guesser1 != nullptr;
        r.has2   = ms->guesser != nullptr;
    }
    r.ran = ok;
    llama_batch_free(b);
    llama_free(dft);
    llama_free(tgt);
    llama_model_free(model);
    return r;
}

// a random guesser file at floor offset `gap` for floors gap..n_layer-1 (F16 weights near 0, F32 biases spread)
std::string write_guesser(const std::string & name, const llama_model * ref, uint32_t gap) {
    const int32_t n_layer = (int32_t) ref->layers.size();
    const int64_t n_embd = ref->hparams.n_embd, n_expert = ref->hparams.n_expert;
    gguf_context * g = gguf_init_empty();
    gguf_set_val_u32(g, "lookahead.floor_offset", gap);
    gguf_set_val_u32(g, "lookahead.first_floor", gap);
    gguf_set_val_u32(g, "lookahead.last_floor", (uint32_t) n_layer - 1);
    gguf_set_val_u32(g, "lookahead.n_embd", (uint32_t) n_embd);
    gguf_set_val_u32(g, "lookahead.n_expert", (uint32_t) n_expert);
    gguf_set_val_f32(g, "lookahead.confidence.scale", 10.0f);
    gguf_set_val_f32(g, "lookahead.confidence.max", 1.0f);
    ggml_init_params ip = { 16*ggml_tensor_overhead() + (size_t) n_layer*(n_embd*n_expert*2 + n_expert*4 + 256), nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(11 + gap);
    std::normal_distribution<float> small(0.0f, 0.02f), spread(0.0f, 2.0f);
    for (int32_t n = (int32_t) gap; n < n_layer; n++) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_expert);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
        ggml_set_name(w, ("blk." + std::to_string(n) + ".la_guess.weight").c_str());
        ggml_set_name(b, ("blk." + std::to_string(n) + ".la_guess.bias").c_str());
        for (int64_t i = 0; i < n_embd*n_expert; i++) {
            ((ggml_fp16_t *) w->data)[i] = ggml_fp32_to_fp16(small(rng));
        }
        for (int64_t e = 0; e < n_expert; e++) {
            ((float *) b->data)[e] = spread(rng);
        }
        gguf_add_tensor(g, w);
        gguf_add_tensor(g, b);
    }
    const std::string path = (std::filesystem::temp_directory_path() /
            ("test-moe-stream-guesser1-mtp-" + std::to_string(getpid()) + "-" + name + ".gguf")).string();
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

bool nonzero(const std::vector<float> & v) {
    for (const float x : v) {
        if (x != 0.0f) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            g_path = argv[++i];
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            g_ngl = atoi(argv[++i]);
        }
    }
    if (g_path.empty()) {
        fprintf(stderr, "usage: %s -m qwen4exp-moe-e64mtp.gguf [-ngl N]\n", argv[0]);
        return 1;
    }
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    ggml_backend_load_all();
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);
    setenv("LLAMA_QSA_UNION", "bias", 1); // the exact route, as the room's model tests hold it

    llama_model * ref = load(false);
    if (ref == nullptr) {
        fprintf(stderr, "failed to load %s\n", g_path.c_str());
        return 1;
    }
    const std::string one = write_guesser("one", ref, 1);
    const std::string two = write_guesser("two", ref, 2);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(ref));
    llama_model_free(ref);

    std::mt19937 rng(97);
    std::vector<llama_token> prompt(37);
    for (auto & t : prompt) {
        t = (llama_token) (rng() % n_vocab);
    }
    // the everyday library's guess switches, both arms
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_ALL", "1", 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", "4", 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_KEEP", "1", 1);
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD");

    unsetenv("LLAMA_MOE_STREAM_GUESSER1");
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER");
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF");
    const turn off = run_turn(prompt);
    setenv("LLAMA_MOE_STREAM_GUESSER1", one.c_str(), 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER", two.c_str(), 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF", "0.2", 1);
    const turn on = run_turn(prompt);

    check(off.ran && on.ran, "both arms: the library and the apprentice made their contexts and ran every step");
    check(nonzero(off.library) && nonzero(off.drafts) && off.drafts.size() == (size_t) 5*n_vocab,
            "the library's logits and the 5 drafts' logits carry signal");
    check(!off.has1 && !off.has2 && off.stops1 == 0 && off.stops2 == 0, "off: no trained guesser loaded, no trained stop");
    check(on.has1 && on.has2, "on: both trained guessers loaded beside the apprentice");
    printf("  on: trained stops, one floor %lld, two floors %lld\n", (long long) on.stops1, (long long) on.stops2);
    check(on.stops1 > 0 && on.stops2 > 0, "on: both trained guesses scored floor stops");
    check(!off.library.empty() && on.library.size() == off.library.size() &&
            memcmp(on.library.data(), off.library.data(), on.library.size()*sizeof(float)) == 0,
            "the library's logits (prompt and check) byte-identical with both guessers on");
    check(on.drafts.size() == off.drafts.size() &&
            memcmp(on.drafts.data(), off.drafts.data(), on.drafts.size()*sizeof(float)) == 0,
            "every draft's logits byte-identical with both guessers on");

    for (const char * e : { "LLAMA_MOE_STREAM_GUESSER1", "LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER", "LLAMA_MOE_STREAM_LOOKAHEAD_CONF",
            "LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", "LLAMA_MOE_STREAM_LOOKAHEAD_ALL", "LLAMA_MOE_STREAM_LOOKAHEAD_KEEP" }) {
        unsetenv(e);
    }
    std::filesystem::remove(one);
    std::filesystem::remove(two);
    printf("%s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}

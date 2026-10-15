// Desk books first (LLAMA_MOE_STREAM_DESK_FIRST, #39) beside the apprentice, on the MTP fixture (qwen4exp-moe-e64mtp:
// two trunk floors and one MTP block, 64 books of 8), streamed. The switch is read when a context opens, and both the
// library's and the apprentice's contexts open on the one streamed model (gotchas: a load-time switch reaches the
// apprentice), so the apprentice's context must leave it off: its graphs build no gate and its floors stop as before.
// Driven as common/speculative.cpp's MTP drafter drives them: the library reads a prompt in, the apprentice reads it
// in and drafts 3 words, then 4 rounds of the library checking 4 words (the apprentice's shape, desk books first) and
// the apprentice drafting 3 more. The runners are slowed (test_land_delay_us) so the checks' floors take trips, and the
// everyday library's guess switches are on (every word of a check, two floors ahead, the keep rule). Checked:
//   same   the library's logits (prompt and checks) and every draft's logits byte-identical with desk books first on
//   ran    only the library's check floors stopped desk-first (rounds x trunk floors), Metal ran one gate per stop,
//          some floors were split and their gates waited

#include "common.h"
#include "llama.h"

#include "../src/llama-ext.h"
#include "../src/llama-model.h"
#include "../src/llama-moe-room.h"
#include "../src/llama-moe-stream.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

std::string g_path;
int         g_ngl      = 99;
int         g_failures = 0;
const int   ROUNDS     = 4;

void (*g_gate_counts)(int64_t *, int64_t *) = nullptr;

void check(bool ok, const std::string & what) {
    printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

llama_model * load() {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers          = g_ngl;
    mp.load_mtp              = true;
    mp.use_extra_bufts       = false;
    mp.moe_stream            = true;
    mp.moe_stream_slots      = 55;
    mp.moe_stream_room_mode  = LLAMA_MOE_ROOM_AUTO;
    mp.moe_stream_room_parts = 4;
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
    std::vector<float> library, drafts;
    int64_t stops = 0, split = 0, waits = 0, calls = 0, trunk = 0;
};

// the apprentice drafts n words from the library's hidden state h after word id at position pos
bool draft(llama_context * dft, int n_embd, int n_vocab, llama_token id, int pos, std::vector<float> h, int n,
        std::vector<llama_token> & words, turn & r) {
    llama_batch b = llama_batch_init(1, n_embd, 1);
    b.token = (llama_token *) malloc(sizeof(llama_token));
    bool ok = true;
    words.clear();
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

turn run_turn(const std::vector<llama_token> & prompt, bool df) {
    turn r;
    if (df) {
        setenv("LLAMA_MOE_STREAM_DESK_FIRST", "1", 1);
    } else {
        unsetenv("LLAMA_MOE_STREAM_DESK_FIRST");
    }
    int64_t calls0 = 0, cut0 = 0;
    if (g_gate_counts) {
        g_gate_counts(&calls0, &cut0);
    }
    llama_model * model = load();
    if (model != nullptr) {
        model->moe_stream()->test_land_delay_us = 3000;
    }
    llama_context * tgt = model ? make_ctx(model, false, nullptr) : nullptr;
    llama_context * dft = tgt ? make_ctx(model, true, tgt) : nullptr;
    unsetenv("LLAMA_MOE_STREAM_DESK_FIRST");
    bool ok = dft != nullptr;
    const int n_embd  = model ? llama_model_n_embd_out(model) : 0;
    const int n_vocab = model ? llama_vocab_n_tokens(llama_model_get_vocab(model)) : 0;
    const int n       = (int) prompt.size();

    llama_batch b = llama_batch_init(512, 0, 1);
    std::vector<float> h;
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
        h.assign(llama_get_embeddings_nextn(tgt), llama_get_embeddings_nextn(tgt) + (size_t) n*n_embd);
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
                memcpy(mb.embd + (size_t) k*n_embd, h.data() + (size_t) (k - 1)*n_embd, n_embd*sizeof(float));
            }
        }
        ok = llama_decode(dft, mb) == 0;
        llama_batch_free(mb);
    }
    std::vector<llama_token> words;
    int pos = n;
    ok = ok && draft(dft, n_embd, n_vocab, id, pos, std::vector<float>(h.end() - n_embd, h.end()), 3, words, r);
    for (int round = 0; ok && round < ROUNDS; round++) {
        // the library checks the first word and the draft: a 4-word batch, every row's logits (all taken, so the
        // positions move on by 4 whatever the draft's luck)
        common_batch_clear(b);
        common_batch_add(b, id, pos, { 0 }, true);
        for (int i = 0; i < 3; i++) {
            common_batch_add(b, words[i], pos + 1 + i, { 0 }, true);
        }
        ok = llama_decode(tgt, b) == 0 && llama_get_embeddings_nextn(tgt) != nullptr;
        if (!ok) {
            break;
        }
        const float * l = llama_get_logits(tgt);
        r.library.insert(r.library.end(), l, l + 4*n_vocab);
        const float * hl = llama_get_embeddings_nextn(tgt) + (size_t) 3*n_embd;
        id   = argmax(l + 3*n_vocab, n_vocab);
        pos += 4;
        ok = draft(dft, n_embd, n_vocab, id, pos, std::vector<float>(hl, hl + n_embd), 3, words, r);
    }
    if (model != nullptr) {
        llama_moe_stream * ms = model->moe_stream();
        std::unique_lock<std::mutex> lk(ms->mtx);
        r.stops = ms->stats.n_df_stops;
        r.split = ms->stats.n_df_split;
        r.waits = ms->stats.n_df_gate_waits;
        r.trunk = (int64_t) model->hparams.n_layer(); // the trunk: nextn layers excluded
    }
    if (g_gate_counts) {
        int64_t cut = 0;
        g_gate_counts(&r.calls, &cut);
        r.calls -= calls0;
    }
    r.ran = ok;
    llama_batch_free(b);
    llama_free(dft);
    llama_free(tgt);
    llama_model_free(model);
    return r;
}

bool nonzero(const std::vector<float> & v) {
    for (const float x : v) {
        if (x != 0.0f) {
            return true;
        }
    }
    return false;
}

bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
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
    llama_log_set([](ggml_log_level, const char * text, void *) {
        if (getenv("GGML_SCHED_DEBUG") != nullptr) {
            fputs(text, stderr);
        }
    }, nullptr);
    ggml_backend_load_all();
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);
    setenv("LLAMA_QSA_UNION", "bias", 1); // the exact route, as the room's model tests hold it
    ggml_backend_reg_t mtl = ggml_backend_reg_by_name("MTL");
    if (mtl != nullptr) {
        g_gate_counts = (void (*)(int64_t *, int64_t *)) ggml_backend_reg_get_proc_address(mtl,
                "ggml_backend_metal_moe_gate_counts");
    }
    const bool gpu = g_ngl > 0 && mtl != nullptr;

    // the everyday library's guess switches, both arms
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_ALL", "1", 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", "4", 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_KEEP", "1", 1);
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD");

    std::mt19937 rng(23);
    std::vector<llama_token> prompt(37);
    for (auto & t : prompt) {
        t = (llama_token) (rng() % 128); // the fixture's vocabulary
    }
    const turn off = run_turn(prompt, false);
    const turn on  = run_turn(prompt, true);
    printf("  on: desk-first stops %lld (trunk floors %lld), split %lld, gate waits %lld, Metal gates %lld\n",
            (long long) on.stops, (long long) on.trunk, (long long) on.split, (long long) on.waits, (long long) on.calls);

    check(off.ran && on.ran, "both arms: the library and the apprentice made their contexts and ran every step");
    check(nonzero(off.library) && nonzero(off.drafts), "the library's and the drafts' logits carry signal");
    check(off.stops == 0 && off.calls == 0, "off: no desk-first stop, no gate");
    check(same(off.library, on.library), "the library's logits (prompt and checks) byte-identical with desk books first on");
    check(same(off.drafts, on.drafts), "every draft's logits byte-identical with desk books first on");
    if (gpu) {
        check(on.trunk > 0 && on.stops == ROUNDS*on.trunk,
                "on: only the library's checks stopped desk-first (rounds x trunk floors), not the apprentice's floor");
        check(on.calls == on.stops, "on: Metal ran one gate per desk-first stop");
        check(on.split > 0 && on.waits > 0, "on: floors split and their gates waited");
    } else {
        check(on.stops == 0 && on.calls == 0, "CPU desk: desk books first stays off");
    }

    for (const char * e : { "LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", "LLAMA_MOE_STREAM_LOOKAHEAD_ALL", "LLAMA_MOE_STREAM_LOOKAHEAD_KEEP" }) {
        unsetenv(e);
    }
    printf("%s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}

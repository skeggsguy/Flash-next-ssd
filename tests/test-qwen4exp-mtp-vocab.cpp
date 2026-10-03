// The apprentice's word list (LLAMA_MTP_VOCAB, src/llama-mtp-vocab.h) on the MTP fixture
// (qwen4exp-moe-e64mtp.gguf: two trunk floors and one MTP block, 64 books of 8, a vocabulary of 128). A library
// and an apprentice context on one model, driven as common/speculative.cpp's MTP drafter drives them: the
// library reads the prompt in, the apprentice reads it in (records only, the default), then six chained guesses.
// The switch is read when a context is made, so each turn sets it before making its two contexts:
//   today    unset and "0" give the same bytes (scores, hidden states, records), and say "word list: off,"
//   exact    a list holding every top word the whole head picked plus ~40% of the rest: every listed word's
//            score byte-identical to the whole head's, every other word -INF, the same six guesses, the
//            hidden states and the apprentice's records byte-identical; the library's own head untouched; the
//            startup line says "word list: on, guesses from N of 128 words"; the copy shows in the memory
//            breakdown as the head's listed rows
//   unlisted a list without the first guess's top word: that guess is the best listed word instead, its listed
//            scores still the whole head's
//   refused  a missing file, an id past the vocabulary, a word that is not an id or an empty list fail the
//            apprentice's context (a mistyped path never runs silently on the whole head)

#include "common.h"
#include "llama.h"

#include "../src/llama-ext.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

std::string g_path;
int         g_ngl      = 99;
int         g_failures = 0;
std::string g_log; // the engine's log lines while a turn makes its contexts

void check(bool ok, const std::string & what) {
    printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

llama_context * make_ctx(llama_model * model, bool mtp, llama_context * other) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 512;
    cp.n_batch    = 512;
    cp.n_ubatch   = 64;
    cp.n_seq_max  = 1;
    cp.op_offload = true;
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

struct mtp_batch {
    llama_batch b;
    mtp_batch(int n, int n_embd) : b(llama_batch_init(n, n_embd, 1)) { b.token = (llama_token *) malloc(sizeof(llama_token)*n); }
    ~mtp_batch() { llama_batch_free(b); }
};

struct turn {
    bool                     made = false; // both contexts made
    bool                     ran  = false; // and every decode ran
    std::vector<float>       library;      // the library's scores for its last prompt row
    std::vector<float>       scores;       // every guess's scores, [n_guess][n_vocab]
    std::vector<float>       hidden;       // every guess's hidden state
    std::vector<llama_token> guesses;
    std::vector<uint8_t>     records;      // the apprentice's state after the guesses
    size_t                   model_bytes = 0; // the apprentice's memory breakdown, model part, all devices
    std::string              log;
};

// value: LLAMA_MTP_VOCAB for this turn's contexts (nullptr: unset)
turn run_turn(llama_model * model, const std::vector<llama_token> & prompt, const char * value, int n_guess) {
    turn r;
    if (value) {
        setenv("LLAMA_MTP_VOCAB", value, 1);
    } else {
        unsetenv("LLAMA_MTP_VOCAB");
    }
    g_log.clear();
    llama_context * tgt = make_ctx(model, false, nullptr);
    llama_context * dft = tgt ? make_ctx(model, true, tgt) : nullptr;
    r.log  = g_log;
    r.made = dft != nullptr;
    unsetenv("LLAMA_MTP_VOCAB");

    const int n_embd  = llama_model_n_embd_out(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n       = (int) prompt.size();
    bool ok = r.made;

    llama_batch b = llama_batch_init(512, 0, 1);
    std::vector<float> h_tgt;
    llama_token id = 0;
    if (ok) { // the library reads the prompt in: its last row's scores, every row's hidden state
        common_batch_clear(b);
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
    if (ok) { // the apprentice reads it in with no outputs: row 0 gets a zero state, row k the library's k-1
        mtp_batch mb(n, n_embd);
        common_batch_clear(mb.b);
        for (int k = 0; k < n; k++) {
            common_batch_add(mb.b, prompt[k], k, { 0 }, false);
            if (k == 0) {
                memset(mb.b.embd, 0, n_embd*sizeof(float));
            } else {
                memcpy(mb.b.embd + (size_t) k*n_embd, h_tgt.data() + (size_t) (k - 1)*n_embd, n_embd*sizeof(float));
            }
        }
        ok = llama_decode(dft, mb.b) == 0;
    }
    std::vector<float> h(h_tgt.end() - (ok ? n_embd : 0), h_tgt.end());
    mtp_batch mb(1, n_embd);
    for (int i = 0; ok && i < n_guess; i++) {
        common_batch_clear(mb.b);
        common_batch_add(mb.b, id, n + i, { 0 }, true);
        memcpy(mb.b.embd, h.data(), n_embd*sizeof(float));
        ok = llama_decode(dft, mb.b) == 0;
        const float * l  = ok ? llama_get_logits_ith(dft, 0) : nullptr;
        const float * hn = ok ? llama_get_embeddings_nextn_ith(dft, 0) : nullptr;
        ok = ok && l != nullptr && hn != nullptr;
        if (ok) {
            r.scores.insert(r.scores.end(), l, l + n_vocab);
            r.hidden.insert(r.hidden.end(), hn, hn + n_embd);
            h.assign(hn, hn + n_embd);
            id = argmax(l, n_vocab);
            r.guesses.push_back(id);
        }
    }
    if (ok) {
        r.records.resize(llama_state_seq_get_size(dft, 0));
        r.records.resize(llama_state_seq_get_data(dft, r.records.data(), r.records.size(), 0));
        for (const auto & e : llama_get_memory_breakdown(dft)) {
            r.model_bytes += e.second.model;
        }
    }
    r.ran = ok;
    llama_batch_free(b);
    llama_free(dft);
    llama_free(tgt);
    return r;
}

std::string write_list(const std::string & name, const std::string & text) {
    const char * tmp = getenv("TMPDIR");
    const std::string path = std::string(tmp ? tmp : "/tmp") + "/test-mtp-vocab-" + std::to_string(getpid()) + "-" + name;
    std::ofstream(path) << text;
    return path;
}

std::string list_text(const std::vector<int> & ids) {
    std::string s = "# test list\n";
    for (const int id : ids) {
        s += std::to_string(id) + "\n";
    }
    return s;
}

bool same_bytes(const std::vector<float> & a, const std::vector<float> & b) {
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

void run_prompt(llama_model * model, std::mt19937 & rng, int n_prompt) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_guess = 6;
    std::vector<llama_token> prompt(n_prompt);
    std::generate(prompt.begin(), prompt.end(), [&] { return (llama_token) (rng() % (uint32_t) n_vocab); });
    const std::string p = "prompt " + std::to_string(n_prompt) + ": ";

    // today: unset and "0"
    const turn unset = run_turn(model, prompt, nullptr, n_guess);
    const turn full  = run_turn(model, prompt, "0", n_guess);
    check(unset.ran && full.ran, p + "today's turns ran");
    check(same_bytes(unset.scores, full.scores) && same_bytes(unset.hidden, full.hidden) && unset.records == full.records,
          p + "unset and \"0\" give the same scores, hidden states and records");
    const std::string off = "word list: off, guesses from all " + std::to_string(n_vocab) + " words";
    check(unset.log.find(off) != std::string::npos && full.log.find(off) != std::string::npos,
          p + "both say \"" + off + "\"");
    check(full.log.find(off) == full.log.rfind(off), p + "once: the library's own context says nothing");
    bool signal = full.scores.size() == (size_t) n_guess*n_vocab;
    for (int g = 0; signal && g < n_guess; g++) {
        const float * s = full.scores.data() + (size_t) g*n_vocab;
        signal = std::isfinite(s[0]) && s[0] != s[1] && s[argmax(s, n_vocab)] > s[0] - 1e9f;
    }
    check(signal, p + "the whole head's scores carry signal");

    // exact: every top word listed, plus ~40% of the rest
    std::vector<int> ids;
    std::vector<bool> listed(n_vocab, false);
    for (const llama_token t : full.guesses) {
        listed[t] = true;
    }
    for (int v = 0; v < n_vocab; v++) {
        listed[v] = listed[v] || rng() % 5 < 2;
        if (listed[v]) {
            ids.push_back(v);
        }
    }
    const std::string path = write_list("exact.txt", list_text(ids));
    const turn trim = run_turn(model, prompt, path.c_str(), n_guess);
    check(trim.ran, p + "the listed turn ran (" + std::to_string(ids.size()) + " of " + std::to_string(n_vocab) + " words)");
    bool kept_same = trim.scores.size() == full.scores.size(), rest_inf = kept_same;
    for (size_t i = 0; kept_same && i < trim.scores.size(); i++) {
        if (listed[i % n_vocab]) {
            kept_same = memcmp(&trim.scores[i], &full.scores[i], sizeof(float)) == 0;
        } else {
            rest_inf = rest_inf && std::isinf(trim.scores[i]) && trim.scores[i] < 0;
        }
    }
    check(kept_same, p + "every listed word's score byte-identical to the whole head's, every guess");
    check(rest_inf, p + "every other word's score -INF");
    check(!trim.guesses.empty() && trim.guesses == full.guesses, p + "the same six guesses");
    check(same_bytes(trim.hidden, full.hidden), p + "every guess's hidden state byte-identical");
    check(!trim.records.empty() && trim.records == full.records, p + "the apprentice's records byte-identical");
    check(same_bytes(trim.library, full.library), p + "the library's own head untouched (its scores byte-identical)");
    const std::string on = "word list: on, guesses from " + std::to_string(ids.size()) + " of " + std::to_string(n_vocab) + " words";
    check(trim.log.find(on) != std::string::npos && trim.log.find("word list: off") == std::string::npos,
          p + "says \"" + on + "\"");
    const ggml_tensor * head = model->output;
    const size_t rows = head ? ggml_row_size(head->type, head->ne[0])*ids.size() : 0;
    const size_t grew = trim.model_bytes - full.model_bytes;
    check(head && grew >= rows + ids.size()*sizeof(int64_t) && grew < rows + ids.size()*sizeof(int64_t) + 4096,
          p + "the memory breakdown grew by the listed rows (" + std::to_string(grew) + " B)");
    unlink(path.c_str());

    // unlisted: without the first guess's top word
    std::vector<int> ids2;
    for (int v = 0; v < n_vocab; v++) {
        if (v != full.guesses[0] && rng() % 5 < 2) {
            ids2.push_back(v);
        }
    }
    const std::string path2 = write_list("unlisted.txt", list_text(ids2));
    const turn miss = run_turn(model, prompt, path2.c_str(), 1);
    int best = ids2[0];
    for (const int v : ids2) {
        best = full.scores[v] > full.scores[best] ? v : best;
    }
    check(miss.ran && miss.guesses.size() == 1 && miss.guesses[0] == best && best != full.guesses[0],
          p + "a list without the top word guesses the best listed word");
    bool kept2 = miss.ran;
    for (size_t k = 0; kept2 && k < ids2.size(); k++) {
        kept2 = memcmp(&miss.scores[ids2[k]], &full.scores[ids2[k]], sizeof(float)) == 0;
    }
    check(kept2, p + "its listed scores still the whole head's");
    unlink(path2.c_str());
}

void refused(llama_model * model) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::vector<llama_token> prompt = { 1, 2, 3, 4, 5 };
    const std::vector<std::pair<std::string, std::string>> bad = {
        { "an id past the vocabulary", "1\n" + std::to_string(n_vocab) + "\n" },
        { "a word that is not an id",  "1\nabc\n" },
        { "an empty list",             "# nothing\n" },
    };
    const turn missing = run_turn(model, prompt, "/nonexistent/list-96k.txt", 1);
    check(!missing.made && missing.log.find("cannot read /nonexistent/list-96k.txt") != std::string::npos,
          "refused: a missing file fails the apprentice's context, naming it");
    for (const auto & [what, text] : bad) {
        const std::string path = write_list("bad.txt", text);
        const turn r = run_turn(model, prompt, path.c_str(), 1);
        check(!r.made, "refused: " + what);
        unlink(path.c_str());
    }
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
    llama_log_set([](ggml_log_level, const char * text, void *) { g_log += text; }, nullptr);
    ggml_backend_load_all();
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers    = g_ngl;
    mp.load_mtp        = true;
    mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(g_path.c_str(), mp);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", g_path.c_str());
        return 1;
    }
    printf("  the head: %s\n", model->output ? ggml_type_name(model->output->type) : "(none)");
    std::mt19937 rng(2468);
    for (const int n_prompt : { 37, 120 }) {
        run_prompt(model, rng, n_prompt);
    }
    refused(model);
    llama_model_free(model);

    printf("%s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}

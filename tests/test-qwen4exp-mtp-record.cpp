// The apprentice records only (LLAMA_MTP_RECORD_ONLY, src/llama-mtp-record.h), on the MTP fixture
// (qwen4exp-moe-e64mtp.gguf: two trunk floors and one MTP block, 64 books of 8, weights around 1). A library
// and an apprentice context on one model, driven as common/speculative.cpp's MTP drafter drives them: the
// apprentice reads the prompt in with no outputs, then check cycles of 3, 0 and 1 guesses (each checked batch
// read in with no outputs; 0 makes a one-token graph without outputs that the next guess must not reuse).
// Switch 0 and 1 in one process (read per context), prompts of 37, 64 and 200 tokens at -ub 64:
//   exact    the apprentice's whole state (llama_state_seq_get_data) after every step, and every guess's
//            logits and hidden state, byte-identical and carrying signal; the off turn's hashes are printed
//            to compare with the build before step 4 by hand ("0" is today's graph)
//   ran      switch on: reading in and the checked batches built no attention, books or head (the eval
//            callback sees every node's name and asks for none, so nothing is split); off, they did
//   reserve  the apprentice's working space (its compute buffers) is smaller with the switch on
//   update   the reserve after a memory update (llama_context::memory_update) asks for no outputs too, so the
//            working space stays the record-only graph's (ggml's allocator only grows): a cross-stream copy on
//            a two-stream apprentice, applied by the next decode; the guess and records byte-identical after it
//   room     the graph-reuse regression (results/engine/gates.md, B's review) with a real apprentice build
//            between two of the library's same-shape batches through the room: reused, same logits, both ways

#include "common.h"
#include "llama.h"

#include "../src/llama-ext.h"
#include "../src/llama-model.h"
#include "../src/llama-moe-room.h"
#include "../src/llama-moe-stream.h"

#include <algorithm>
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
std::string g_log;             // the engine's log lines while g_capture is set
bool        g_capture  = false;

void check(bool ok, const std::string & what) {
    printf("  %-78s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

// the apprentice's node names, by kind, as the scheduler walks its graph
struct seen {
    int attn = 0; // mtp_attn_out: the dense attention
    int moe  = 0; // mtp_ffn_moe_out: the books
    int head = 0; // result_output: the head
    void clear() { *this = {}; }
    void add(const seen & o) { attn += o.attn; moe += o.moe; head += o.head; }
    bool none() const { return attn == 0 && moe == 0 && head == 0; }
    bool all()  const { return attn > 0 && moe > 0 && head > 0; }
};

bool starts(const char * s, const char * p) { return strncmp(s, p, strlen(p)) == 0; }

bool observe(ggml_tensor * t, bool ask, void * user_data) {
    auto * s = (seen *) user_data;
    if (ask) {
        s->attn += starts(t->name, "mtp_attn_out");
        s->moe  += starts(t->name, "mtp_ffn_moe_out");
        s->head += starts(t->name, "result_output");
    }
    return false; // never ask for a tensor: the graph runs unsplit, as without the callback
}

struct run_result {
    std::vector<std::vector<uint8_t>> records; // the apprentice's state after each phase
    std::vector<float> guesses;                // every guess step's logits, then its hidden state
    seen reading, checked, guessing;           // what reading in, the checked batch and guessing built
    size_t compute = 0;                        // the apprentice's compute buffers, all devices
};

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

// n_seq: the context's sequences (streams); an apprentice gets as many outputs, as common_speculative gives
// its draft (n_parallel)
llama_context * make_ctx(llama_model * model, bool mtp, llama_context * other, seen * obs, bool stream, int n_seq = 1) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 512;
    cp.n_batch    = 512;
    cp.n_ubatch   = stream ? 128 : 64;
    cp.n_seq_max  = n_seq;
    cp.no_perf    = false;
    cp.op_offload = !stream; // streaming turns it off
    if (mtp) {
        cp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
        cp.ctx_other     = other;
        cp.n_outputs_max = n_seq;
        cp.n_rs_seq      = 0;
        cp.cb_eval           = observe;
        cp.cb_eval_user_data = obs;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx != nullptr) {
        llama_set_embeddings_nextn(ctx, true, /*masked*/ mtp);
    }
    return ctx;
}

std::vector<uint8_t> records_of(llama_context * dft) {
    std::vector<uint8_t> buf(llama_state_seq_get_size(dft, 0));
    buf.resize(llama_state_seq_get_data(dft, buf.data(), buf.size(), 0));
    return buf;
}

// the context's compute buffers (its working space), all devices
size_t compute_of(llama_context * ctx) {
    size_t total = 0;
    for (const auto & e : llama_get_memory_breakdown(ctx)) {
        total += e.second.compute;
    }
    return total;
}

llama_token argmax(const float * row, int n) {
    llama_token best = 0;
    for (int v = 1; v < n; v++) {
        best = row[v] > row[best] ? v : best;
    }
    return best;
}

// the MTP batch carries tokens and hidden states both (common/speculative.cpp does the same)
struct mtp_batch {
    llama_batch b;
    mtp_batch(int n, int n_embd) : b(llama_batch_init(n, n_embd, 1)) { b.token = (llama_token *) malloc(sizeof(llama_token)*n); }
    ~mtp_batch() { llama_batch_free(b); }
};

// the apprentice reads toks in at pos0.. with no outputs: row 0's hidden state is `pending`, row k's the
// library's row k-1 (the shift speculative.cpp's process() makes)
bool read_in(llama_context * dft, const std::vector<llama_token> & toks, llama_pos pos0, const float * pending,
        const float * h_tgt, int n_embd) {
    mtp_batch mb((int) toks.size(), n_embd);
    common_batch_clear(mb.b);
    for (size_t k = 0; k < toks.size(); k++) {
        common_batch_add(mb.b, toks[k], pos0 + (llama_pos) k, { 0 }, false);
        memcpy(mb.b.embd + k*n_embd, k == 0 ? pending : h_tgt + (k - 1)*n_embd, n_embd*sizeof(float));
    }
    return llama_decode(dft, mb.b) == 0;
}

// n guesses one token at a time from (id, pos, h); appends logits and hidden states, returns the guesses
std::vector<llama_token> guess(llama_context * dft, llama_token id, llama_pos pos, std::vector<float> h, int n,
        int n_embd, int n_vocab, run_result & r, bool & ok) {
    std::vector<llama_token> out;
    mtp_batch mb(1, n_embd);
    for (int i = 0; ok && i < n; i++) {
        common_batch_clear(mb.b);
        common_batch_add(mb.b, id, pos + i, { 0 }, true);
        memcpy(mb.b.embd, h.data(), n_embd*sizeof(float));
        ok = llama_decode(dft, mb.b) == 0;
        const float * l  = ok ? llama_get_logits_ith(dft, 0) : nullptr;
        const float * hn = ok ? llama_get_embeddings_nextn_ith(dft, 0) : nullptr;
        ok = ok && l != nullptr && hn != nullptr;
        if (ok) {
            r.guesses.insert(r.guesses.end(), l, l + n_vocab);
            r.guesses.insert(r.guesses.end(), hn, hn + n_embd);
            h.assign(hn, hn + n_embd);
            id = argmax(l, n_vocab);
            out.push_back(id);
        }
    }
    return out;
}

// the library checks id + the guesses at pos in one batch (every row out); the apprentice drops its guesses'
// records and reads the checked batch in. id, pos and pending move on to the next cycle
bool check_cycle(llama_context * tgt, llama_context * dft, llama_batch & b, const std::vector<llama_token> & guesses,
        llama_token & id, llama_pos & pos, std::vector<float> & pending, int n_embd, int n_vocab, seen & obs,
        run_result & r) {
    std::vector<llama_token> chk = { id };
    chk.insert(chk.end(), guesses.begin(), guesses.end());
    common_batch_clear(b);
    for (size_t i = 0; i < chk.size(); i++) {
        common_batch_add(b, chk[i], pos + (llama_pos) i, { 0 }, true);
    }
    if (llama_decode(tgt, b) != 0 || !llama_memory_seq_rm(llama_get_memory(dft), 0, pos, -1)) {
        return false;
    }
    const float * h = llama_get_embeddings_nextn(tgt);
    if (h == nullptr) {
        return false;
    }
    const std::vector<float> h_chk(h, h + chk.size()*n_embd);
    id = argmax(llama_get_logits_ith(tgt, -1), n_vocab);
    obs.clear();
    const bool ok = read_in(dft, chk, pos, pending.data(), h_chk.data(), n_embd);
    r.checked.add(obs);
    r.records.push_back(records_of(dft));
    pending.assign(h_chk.end() - n_embd, h_chk.end());
    pos += (llama_pos) chk.size();
    return ok;
}

// the library reads the prompt in: the last row's logits (argmax to id) out, every row's hidden state
bool library_reads(llama_context * tgt, llama_batch & b, const std::vector<llama_token> & prompt,
        std::vector<float> & h_tgt, llama_token & id) {
    const int n_embd  = llama_model_n_embd_out(llama_get_model(tgt));
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(tgt)));
    const int n       = (int) prompt.size();
    common_batch_clear(b);
    for (int i = 0; i < n; i++) {
        common_batch_add(b, prompt[i], i, { 0 }, i == n - 1);
    }
    if (llama_decode(tgt, b) != 0) {
        return false;
    }
    const float * h = llama_get_embeddings_nextn(tgt);
    if (h == nullptr) {
        return false;
    }
    h_tgt.assign(h, h + (size_t) n*n_embd);
    id = argmax(llama_get_logits_ith(tgt, -1), n_vocab);
    return true;
}

// one conversation turn with the switch at `on`: the prompt read in, then check cycles of 3, 0 and 1 guesses
// (0: the library checks the last word alone, so the apprentice reads in one token without outputs, a
// graph of the guess's own shape that must not be reused for the next guess), then a last guess
bool run_turn(llama_model * model, const std::vector<llama_token> & prompt, bool on, run_result & r) {
    setenv("LLAMA_MTP_RECORD_ONLY", on ? "1" : "0", 1);
    seen obs;
    llama_context * tgt = make_ctx(model, false, nullptr, nullptr, false);
    llama_context * dft = tgt ? make_ctx(model, true, tgt, &obs, false) : nullptr;
    const int n_embd  = llama_model_n_embd_out(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n       = (int) prompt.size();
    bool ok = tgt != nullptr && dft != nullptr;

    llama_batch b = llama_batch_init(512, 0, 1);
    std::vector<float> h_tgt, pending(n_embd, 0.0f);
    llama_token id  = 0;
    llama_pos   pos = n;
    ok = ok && library_reads(tgt, b, prompt, h_tgt, id);
    if (ok) { // the apprentice's records of the prompt
        obs.clear();
        ok = read_in(dft, prompt, 0, pending.data(), h_tgt.data(), n_embd);
        r.reading = obs;
        pending.assign(h_tgt.end() - n_embd, h_tgt.end());
        r.records.push_back(records_of(dft));
    }
    for (const int k : { 3, 0, 1 }) {
        std::vector<llama_token> guesses;
        if (ok && k > 0) {
            obs.clear();
            guesses = guess(dft, id, pos, pending, k, n_embd, n_vocab, r, ok);
            r.guessing.add(obs);
            r.records.push_back(records_of(dft));
        }
        ok = ok && check_cycle(tgt, dft, b, guesses, id, pos, pending, n_embd, n_vocab, obs, r);
    }
    if (ok) {
        guess(dft, id, pos, pending, 1, n_embd, n_vocab, r, ok);
        r.records.push_back(records_of(dft));
    }
    if (ok) {
        r.compute = compute_of(dft);
    }
    llama_batch_free(b);
    llama_free(dft);
    llama_free(tgt);
    return ok;
}

// the reserve after a memory update. When a context's memory changes under it, llama_context::memory_update
// applies the change and reserves the big graph again; ggml's allocator only grows, so with the switch on
// that reserve must ask for no outputs too, or the working space grows back to the whole floor's on the first
// update. The update here is a cross-stream copy on a two-stream apprentice (a K-shift is not available to
// an mrope model), queued by llama_memory_seq_cp and applied by the next decode, a guess
struct update_result {
    size_t before = 0, after = 0; // the apprentice's compute buffers before and after the update
    std::vector<float>   guess;   // the guess decoded with the update: its logits and hidden state
    std::vector<uint8_t> records; // the apprentice's state after it
    std::string          reserves; // graph_reserve's debug lines during that decode
};

bool run_update(llama_model * model, const std::vector<llama_token> & prompt, bool on, update_result & r) {
    setenv("LLAMA_MTP_RECORD_ONLY", on ? "1" : "0", 1);
    seen obs;
    llama_context * tgt = make_ctx(model, false, nullptr, nullptr, false);
    llama_context * dft = tgt ? make_ctx(model, true, tgt, &obs, false, /*n_seq*/ 2) : nullptr;
    const int n_embd  = llama_model_n_embd_out(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n       = (int) prompt.size();
    bool ok = tgt != nullptr && dft != nullptr;

    llama_batch b = llama_batch_init(512, 0, 1);
    std::vector<float> h_tgt, pending(n_embd, 0.0f);
    llama_token id = 0;
    ok = ok && library_reads(tgt, b, prompt, h_tgt, id);
    if (ok) {
        ok = read_in(dft, prompt, 0, pending.data(), h_tgt.data(), n_embd);
        pending.assign(h_tgt.end() - n_embd, h_tgt.end());
        r.before = compute_of(dft);
    }
    if (ok) {
        llama_memory_seq_cp(llama_get_memory(dft), 0, 1, -1, -1); // queued: the next decode applies it
        run_result tmp;
        g_log.clear();
        g_capture = true;
        guess(dft, id, n, pending, 1, n_embd, n_vocab, tmp, ok);
        g_capture = false;
        r.guess   = tmp.guesses;
        r.records = records_of(dft);
        r.after   = compute_of(dft);
        for (size_t at = 0; (at = g_log.find("reserving a graph for ubatch", at)) != std::string::npos; ) {
            const size_t end = g_log.find('\n', at);
            r.reserves += g_log.substr(at, end == std::string::npos ? end : end - at + 1);
            at = end == std::string::npos ? end : end + 1;
        }
    }
    llama_batch_free(b);
    llama_free(dft);
    llama_free(tgt);
    return ok;
}


// FNV-1a over bytes, for the off turn's printed hashes
uint64_t fnv(const void * data, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; i++) {
        h = (h ^ ((const uint8_t *) data)[i])*1099511628211ull;
    }
    return h;
}

bool mostly_nonzero(const std::vector<uint8_t> & v) {
    return !v.empty() && (size_t) std::count_if(v.begin(), v.end(), [](uint8_t x) { return x != 0; })*2 >= v.size();
}

void exact_and_ran(llama_model * model, std::mt19937 & rng, int n_prompt) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> prompt(n_prompt);
    std::generate(prompt.begin(), prompt.end(), [&] { return (llama_token) (rng() % (uint32_t) n_vocab); });
    const std::string p = "prompt " + std::to_string(n_prompt) + ": ";
    run_result off, on;
    check(run_turn(model, prompt, false, off) && run_turn(model, prompt, true, on), p + "both turns ran");

    // reading in, then per cycle (3, 0, 1 guesses) the guesses' and the checked batch's, then the last guess
    bool same_records = off.records.size() == 7 && on.records.size() == 7;
    for (size_t i = 0; same_records && i < off.records.size(); i++) {
        same_records = off.records[i] == on.records[i];
    }
    check(same_records, p + "records byte-identical after reading in, each guess and each checked batch");
    check(same_records && mostly_nonzero(on.records[0]), p + "the records carry signal (mostly nonzero bytes)");
    check(!off.guesses.empty() && off.guesses.size() == on.guesses.size() &&
          memcmp(off.guesses.data(), on.guesses.data(), off.guesses.size()*sizeof(float)) == 0,
          p + "every guess's logits and hidden state byte-identical");

    check(off.reading.all() && off.checked.all(), p + "off: reading in and the checked batch ran the whole floor");
    check(on.reading.none() && on.checked.none(), p + "on: reading in and the checked batch built no attention, books or head");
    check(off.guessing.all() && on.guessing.all(), p + "guessing runs the whole floor either way");
    uint64_t h_rec = fnv(nullptr, 0);
    for (const auto & rec : off.records) {
        h_rec = fnv(rec.data(), rec.size(), h_rec);
    }
    printf("  %-78s records %016llx guesses %016llx\n", (p + "off turn's hashes (printed)").c_str(),
            (unsigned long long) h_rec, (unsigned long long) fnv(off.guesses.data(), off.guesses.size()*sizeof(float)));
    printf("  %-78s %.1f -> %.1f KiB\n", (p + "apprentice compute buffers, off -> on").c_str(),
            off.compute/1024.0, on.compute/1024.0);
    check(on.compute > 0 && on.compute < off.compute, p + "the apprentice's working space is smaller with it on");
}

void after_update(llama_model * model, std::mt19937 & rng, int n_prompt) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> prompt(n_prompt);
    std::generate(prompt.begin(), prompt.end(), [&] { return (llama_token) (rng() % (uint32_t) n_vocab); });
    const std::string p = "update: ";
    update_result off, on;
    check(run_update(model, prompt, false, off) && run_update(model, prompt, true, on), p + "both turns ran");
    // the update did reserve, once, and the record-only one asked for no outputs (the whole floor's for its two)
    check(off.reserves.find("n_outputs =    2") != std::string::npos && off.reserves.find('\n') == off.reserves.size() - 1,
          p + "off: the decode after the copy reserved once, the whole floor with its outputs");
    check(on.reserves.find("n_outputs =    0")  != std::string::npos && on.reserves.find('\n')  == on.reserves.size() - 1,
          p + "on: the decode after the copy reserved once, asking for no outputs");
    check(!off.guess.empty() && off.guess == on.guess, p + "the guess after the update byte-identical");
    check(mostly_nonzero(on.records) && off.records == on.records, p + "the records after the update byte-identical");
    printf("  %-78s %.1f -> %.1f KiB (on), %.1f KiB (off)\n", (p + "apprentice compute buffers, before -> after").c_str(),
            on.before/1024.0, on.after/1024.0, off.after/1024.0);
    check(on.before > 0 && on.after == on.before, p + "the working space stays the record-only graph's after the update");
    check(on.after < off.after, p + "and smaller than the whole floor's");
}

// the library reads F in twice through the room, clearing its memory between, with the apprentice
// reading F in between (the apprentice shares the model's book manager)
void room_reuse(bool on, const std::vector<llama_token> & F) {
    setenv("LLAMA_MTP_RECORD_ONLY", on ? "1" : "0", 1);
    const std::string p = std::string("room, record-only ") + (on ? "on: " : "off: ");
    llama_model * model = load(true);
    seen obs;
    llama_context * tgt = model ? make_ctx(model, false, nullptr, nullptr, true) : nullptr;
    llama_context * dft = tgt ? make_ctx(model, true, tgt, &obs, true) : nullptr;
    bool ok = dft != nullptr;
    const int n_vocab = model ? llama_vocab_n_tokens(llama_model_get_vocab(model)) : 0;
    const int n_embd  = model ? llama_model_n_embd_out(model) : 0;
    const int n       = (int) F.size();
    std::vector<float> logits[2];
    llama_batch b = llama_batch_init(512, 0, 1);
    for (int pass = 0; ok && pass < 2; pass++) {
        llama_memory_clear(llama_get_memory(tgt), true);
        common_batch_clear(b);
        for (int i = 0; i < n; i++) {
            common_batch_add(b, F[i], i, { 0 }, true);
        }
        ok = llama_decode(tgt, b) == 0;
        if (ok) {
            logits[pass].assign(llama_get_logits(tgt), llama_get_logits(tgt) + (size_t) n*n_vocab);
        }
        if (ok && pass == 0) {
            const std::vector<float> h(llama_get_embeddings_nextn(tgt), llama_get_embeddings_nextn(tgt) + (size_t) n*n_embd);
            const std::vector<float> pending(n_embd, 0.0f);
            obs.clear();
            ok = read_in(dft, F, 0, pending.data(), h.data(), n_embd);
        }
    }
    const int64_t reused = ok ? llama_perf_context(tgt).n_reused : 0;
    // both of the library's batches went through the room on each trunk floor (desk + 4 parts); the
    // apprentice's floor never takes it, and with the switch off its books ran beside it
    const llama_moe_room * room = model && model->moe_stream() ? model->moe_stream()->room.get() : nullptr;
    const int64_t groups = room ? room->stats.n_groups : 0;
    const int64_t want   = model ? 2*(int64_t) llama_model_n_layer(model)*(1 + 4) : -1;
    check(ok, p + "ran");
    check(groups == want, p + "room groups ran " + std::to_string(groups) + " of " + std::to_string(want));
    check(on ? obs.none() : obs.all(), p + (on ? "the apprentice built no book sweep" : "the apprentice built its whole floor"));
    check(reused >= 1, p + "the second batch reused the library's graph (" + std::to_string(reused) + ")");
    check(!logits[0].empty() && logits[0] == logits[1], p + "both batches wrote the same logits");
    llama_batch_free(b);
    llama_free(dft);
    llama_free(tgt);
    llama_model_free(model);
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
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
        if (g_capture) {
            g_log += text;
        }
    }, nullptr);
    ggml_backend_load_all();
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);

    llama_model * model = load(false);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", g_path.c_str());
        return 1;
    }
    std::mt19937 rng(4321);
    for (const int n_prompt : { 37, 64, 200 }) {
        exact_and_ran(model, rng, n_prompt);
    }
    after_update(model, rng, 64);
    llama_model_free(model);

    // the room takes batches of 35+ here (the rung's override, read at each load), so F's 128 go through it
    setenv("LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS", "35", 1);
    std::vector<llama_token> F(128);
    std::generate(F.begin(), F.end(), [&] { return (llama_token) (rng() % 128u); });
    room_reuse(false, F);
    room_reuse(true, F);

    printf("%s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}

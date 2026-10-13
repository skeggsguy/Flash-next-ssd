// LLAMA_MOE_STREAM_STATE_LOG (src/llama-moe-stream-statelog.h) on the four-floor 64-book fixture. The log must leave
// the words alone and must hold what each floor's router saw and said. A 20-word read-in (waves: not logged), 24
// written words, three checks of 4 words (every row an output), a 3-word read-in asking only for the last word's
// output (its last floor sees one row, so that floor is not logged), then 4 more written words, on fresh models
// (the switch is read at load):
//   - the logits and every floor's MoE output are byte-identical with the log off, on, and with no streaming; the
//     shared expert, built before the floor stop when the log is on, gives the same bytes as after it;
//   - off writes no file; on, the file holds one record per word per streamed floor of every single-wave call;
//   - each record's router scores, router input (the floor's second hc_mixed: the FFN's, after attention's) and
//     shared expert output are the f16 of what the graph computed (caught by the eval callback), its slip is the
//     top n_expert_used of those scores, and a second run without the callback writes the same file;
//   - that second run guesses two floors ahead (LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2) and fetches books by it: the
//     logging op still runs the lookahead it replaced.

#include "test-moe-room-model.h"

#include <algorithm>
#include <map>
#include <mutex>

#include <unistd.h>

namespace {

const char * const g_env = "LLAMA_MOE_STREAM_STATE_LOG";

struct floor_cap {
    std::vector<float> scores, shexp, moe;
    std::vector<std::vector<float>> mix; // every hc_mixed of the floor, in compute order
};
struct decode_cap {
    int n_tok = 0;
    std::map<int, floor_cap> floors;
};
std::vector<decode_cap> * g_caps = nullptr;
bool g_cap_bad = false;

// exactly "prefix-il" (not a view of it, such as "hc_mixed-0 (reshaped)"): il, else -1
int wanted(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-' || name[n + 1] == '\0') {
        return -1;
    }
    for (const char * p = name + n + 1; *p; p++) {
        if (*p < '0' || *p > '9') {
            return -1;
        }
    }
    return atoi(name + n + 1);
}

bool capture_states(ggml_tensor * t, bool ask, void *) {
    int il = -1;
    int which = -1;
    const char * names[4] = { "ffn_moe_logits", "ffn_shexp_gated", "hc_mixed", "ffn_moe_out" };
    for (int k = 0; k < 4 && il < 0; k++) {
        il = wanted(t->name, names[k]);
        which = k;
    }
    if (ask || il < 0) {
        return il >= 0;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || g_caps == nullptr || g_caps->empty()) {
        g_cap_bad = true;
        return true;
    }
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    floor_cap & f = g_caps->back().floors[il];
    switch (which) {
        case 0: f.scores = std::move(v); break;
        case 1: f.shexp  = std::move(v); break;
        case 2: f.mix.push_back(std::move(v)); break;
        default: f.moe   = std::move(v); break;
    }
    return true;
}

struct batch_plan {
    int  n;
    bool all; // every row an output, or only the last
};

const std::vector<batch_plan> & script() {
    static std::vector<batch_plan> s;
    if (s.empty()) {
        s.push_back({ 20, false });
        for (int i = 0; i < 24; i++) s.push_back({ 1, true });
        for (int i = 0; i < 3;  i++) s.push_back({ 4, true });
        s.push_back({ 3, false });
        for (int i = 0; i < 4;  i++) s.push_back({ 1, true });
    }
    return s;
}

bool run_script(llama_model * model, std::vector<float> & logits, std::vector<decode_cap> * caps) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 1024;
    cp.n_batch    = 512;
    cp.n_ubatch   = 512;
    cp.op_offload = false;
    if (caps) {
        cp.cb_eval = capture_states;
    }
    g_caps = caps;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        return false;
    }
    std::mt19937 rng(11);
    const std::vector<llama_token> toks = random_tokens(rng, 128, n_vocab);
    llama_batch batch = llama_batch_init(512, 0, 1);
    bool ok = true;
    size_t pos = 0;
    for (const batch_plan & b : script()) {
        common_batch_clear(batch);
        for (int i = 0; i < b.n; i++, pos++) {
            common_batch_add(batch, toks[pos], (llama_pos) pos, { 0 }, b.all || i + 1 == b.n);
        }
        if (caps) {
            caps->push_back({});
            caps->back().n_tok = b.n;
        }
        ok = ok && llama_decode(ctx, batch) == 0;
        for (int i = 0; ok && i < (b.all ? b.n : 1); i++) {
            const float * l = llama_get_logits_ith(ctx, b.all ? i : -1);
            logits.insert(logits.end(), l, l + n_vocab);
        }
    }
    llama_batch_free(batch);
    llama_free(ctx);
    g_caps = nullptr;
    return ok;
}

std::vector<uint8_t> slurp(const std::string & path) {
    std::vector<uint8_t> b;
    if (FILE * f = fopen(path.c_str(), "rb")) {
        uint8_t buf[65536];
        size_t  n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
            b.insert(b.end(), buf, buf + n);
        }
        fclose(f);
    }
    return b;
}

template <typename T> T at(const std::vector<uint8_t> & b, size_t off) {
    T v{};
    memcpy(&v, b.data() + off, sizeof(T));
    return v;
}

// row r of a captured [width, rows] matrix, against n f16 values in the file: bit for bit
bool same_f16(const std::vector<uint8_t> & b, size_t off, const std::vector<float> & m, int64_t width, int64_t r) {
    if (m.size() < (size_t) (width*(r + 1))) {
        return false;
    }
    for (int64_t c = 0; c < width; c++) {
        if (at<ggml_fp16_t>(b, off + 2*c) != ggml_fp32_to_fp16(m[r*width + c])) {
            return false;
        }
    }
    return true;
}

std::vector<float> all_moe(const std::vector<decode_cap> & caps) {
    std::vector<float> v;
    for (const auto & d : caps) {
        for (const auto & f : d.floors) {
            v.insert(v.end(), f.second.moe.begin(), f.second.moe.end());
        }
    }
    return v;
}

} // namespace

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    const char * tmp = getenv("TMPDIR");
    const std::string dir = std::string(tmp && *tmp ? tmp : "/tmp/") + "/";
    const std::string path_on   = dir + "statelog-on-"   + std::to_string(getpid()) + ".mssl";
    const std::string path_nocb = dir + "statelog-nocb-" + std::to_string(getpid()) + ".mssl";
    remove(path_on.c_str());
    remove(path_nocb.c_str());

    unsetenv(g_env);
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD"); // the lookahead at its default width, as in every rung
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2");
    std::vector<float> ref, off, on, nocb;
    std::vector<decode_cap> c_ref, c_off, c_on;
    check(run_script(ref_model, ref, &c_ref), "no streaming: the words ran");
    llama_model_free(ref_model);

    config c; // a 40-slot desk of 64 books: writing and the checks take one wave, the 20-word read-in does not
    llama_model * m_off = load(c);
    check(m_off && n_streamed_floors(m_off) >= 3, "streamed, log off: loaded, three floors or more streamed");
    if (m_off) {
        check(m_off->moe_stream()->state_log == nullptr, "log off: the manager has no log");
        check(run_script(m_off, off, &c_off), "log off: the words ran");
        llama_model_free(m_off);
    }
    check(slurp(path_on).empty() && slurp(path_nocb).empty(), "log off: no file written");

    setenv(g_env, path_on.c_str(), 1);
    llama_model * m_on = load(c);
    std::vector<int> floors;
    int last_floor = -1;
    if (m_on) {
        for (int il = 0; il < (int) m_on->moe_stream()->layers.size(); il++) {
            if (m_on->moe_stream()->layer(il)) {
                floors.push_back(il);
            }
        }
        last_floor = (int) llama_model_n_layer(m_on) - 1;
        check(run_script(m_on, on, &c_on), "log on: the words ran");
        llama_model_free(m_on); // closes the log
    }
    // the second run with the log on also guesses two floors ahead: the logging op must still hand the lookahead
    // its logits (its fifth input), or the log would silently cost every rung's prefetch
    setenv(g_env, path_nocb.c_str(), 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", "4", 1);
    llama_model * m_nocb = load(c);
    int64_t la2_issued = 0;
    if (m_nocb) {
        check(run_script(m_nocb, nocb, nullptr), "log on, no eval callback, two floors ahead: the words ran");
        {
            llama_moe_stream * ms = m_nocb->moe_stream();
            std::unique_lock<std::mutex> lk(ms->mtx);
            la2_issued = ms->stats.n_la2_issued;
        }
        llama_model_free(m_nocb);
    }
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2");
    unsetenv(g_env);
    check(la2_issued > 0, "the logging stop still runs the lookahead: " + std::to_string(la2_issued) +
            " books fetched two floors ahead with the log on");

    check(!g_cap_bad, "every caught tensor was a contiguous F32");
    check(nonzero(ref) && nonzero(all_moe(c_ref)), "the reference logits and MoE outputs carry signal");
    check(same(off, ref) && same(on, ref) && same(nocb, ref), "the same logits: no streaming, log off, log on (both)");
    check(same(all_moe(c_off), all_moe(c_ref)) && same(all_moe(c_on), all_moe(c_ref)),
            "the same MoE output on every floor: no streaming, log off, log on");
    bool shexp_same = c_on.size() == c_off.size();
    for (size_t d = 0; shexp_same && d < c_on.size(); d++) {
        for (int il : floors) {
            shexp_same = shexp_same && same(c_on[d].floors[il].shexp, c_off[d].floors[il].shexp);
        }
    }
    check(shexp_same, "the shared expert built before the stop: the same bytes as after it");

    const std::vector<uint8_t> b = slurp(path_on);
    check(b == slurp(path_nocb), "the eval callback changes nothing in the log");
    check(b.size() >= 40 && memcmp(b.data(), "MSSL", 4) == 0 && at<uint32_t>(b, 4) == 1, "MSSL, version 1");
    if (b.size() < 40) {
        return finish();
    }
    const uint32_t rec = at<uint32_t>(b, 12), n_slip = at<uint32_t>(b, 20), n_cur = at<uint32_t>(b, 24);
    const uint32_t n_sc = at<uint32_t>(b, 28), n_sh = at<uint32_t>(b, 32);
    check(at<uint32_t>(b, 8) == 40 && rec == 12 + 2*n_slip + 2*(n_cur + n_sc + n_sh) && n_slip == 8 &&
          n_sc == 64 && n_cur == n_sh && at<uint32_t>(b, 36) == 1, "the header's sizes: slip 8, 64 scores, f16");

    // the calls the log should hold: every one-wave call of the script, its floors, its rows
    int64_t want_records = 0, want_calls = 0;
    for (const batch_plan & p : script()) {
        if (p.n == 20) {
            continue; // waves
        }
        want_calls++;
        for (int il : floors) {
            want_records += (il == last_floor && !p.all) ? 0 : p.n;
        }
    }
    const int64_t n_rec = (int64_t) ((b.size() - 40)/rec);
    check((b.size() - 40) % rec == 0 && n_rec == want_records,
            "one record per word per floor: " + std::to_string(n_rec) + " of " + std::to_string(want_records));

    // each record against what the graph computed (c_on, decode d = call + 1: the 20-word read-in is not logged)
    bool fields = true, scores = true, cur = true, shexp = true, slip = true;
    int64_t calls_seen = 0;
    size_t off_r = 40;
    for (int64_t k = 0; k < n_rec; k++, off_r += rec) {
        const uint32_t call = at<uint32_t>(b, off_r), ntok = at<uint32_t>(b, off_r + 4);
        const uint16_t row  = at<uint16_t>(b, off_r + 8);
        const int      il   = at<uint8_t>(b, off_r + 10);
        calls_seen = std::max<int64_t>(calls_seen, call + 1);
        if ((size_t) call + 1 >= c_on.size() || c_on[call + 1].n_tok != (int) ntok || row >= ntok) {
            fields = false;
            continue;
        }
        floor_cap & f = c_on[call + 1].floors[il];
        const size_t s_off = off_r + 12, c_off_ = s_off + 2*n_slip, sc_off = c_off_ + 2*n_cur, sh_off = sc_off + 2*n_sc;
        scores = scores && same_f16(b, sc_off, f.scores, n_sc, row);
        cur    = cur && f.mix.size() == 2 && same_f16(b, c_off_, f.mix[1], n_cur, row) &&
                 !same_f16(b, c_off_, f.mix[0], n_cur, row); // not attention's
        shexp  = shexp && same_f16(b, sh_off, f.shexp, n_sh, row);
        // the slip: the router's top n_slip of its f32 scores, as a set (top-k does not promise an order)
        if (f.scores.size() >= (size_t) n_sc*(row + 1)) {
            std::vector<int> idx(n_sc);
            for (uint32_t e = 0; e < n_sc; e++) idx[e] = (int) e;
            const float * sr = f.scores.data() + (size_t) row*n_sc;
            std::partial_sort(idx.begin(), idx.begin() + n_slip, idx.end(), [&](int x, int y) { return sr[x] > sr[y]; });
            std::vector<int> want(idx.begin(), idx.begin() + n_slip), got;
            for (uint32_t s = 0; s < n_slip; s++) got.push_back(at<int16_t>(b, s_off + 2*s));
            std::sort(want.begin(), want.end());
            std::sort(got.begin(), got.end());
            slip = slip && want == got;
        } else {
            slip = false;
        }
    }
    check(fields && calls_seen == want_calls, "call, words and row match the script: " + std::to_string(calls_seen) +
            " calls of " + std::to_string(want_calls));
    check(scores, "every record's scores are the router's (f16 of ffn_moe_logits)");
    check(cur,    "every record's router input is the FFN's hc_mixed");
    check(shexp,  "every record's shared expert is ffn_shexp_gated");
    check(slip,   "every record's slip is the top 8 of its scores");
    check(n_rec > 0, "the log is not empty");

    remove(path_on.c_str());
    remove(path_nocb.c_str());
    return finish();
}

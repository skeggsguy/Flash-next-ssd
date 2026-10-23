// The lenient check (LLAMA_SPEC_LENIENT, common/speculative-lenient.h, FAIR-CHECK-PLAN L3) on the MTP fixture
// (qwen4exp-moe-e64mtp.gguf), with the library's own draws, driven as the server drives speculation
// (tests/speculative-pq-record-run.h):
//   law       one check of one drafted word, over and over on the same logits: the written word's frequencies are
//             p' (w* gets a = min(1, gamma p*), any other word p (1 - a) / (1 - p*)), by chi-square, at gamma 1.3, 2
//             and inf; and the sampler is told the word written (its history ends with the check's words)
//   gamma 1   writes today's words: on at gamma 1, a whole generation at 1.0 (seeded) gives the same tokens and rounds
//             as off, through the plain loop and through the p/q record's loop
//   temp 0    the switch on at gamma 1.3 changes nothing at temperature 0: the server's request test leaves it off,
//             and the loop forced on at gamma inf still writes the same (the chain leaves p* at 0 or 1)
//   dial      at 1.0 with gamma inf the dial keeps words today's check would not (the generation differs); at 3 and inf
//             the record's loop and the plain loop write the same words from the same stream, the record names each
//             kept word (`len`: the drafted word written, the draw another), and its kept counts still match the rate
//             trace's
// Mutation check (each seen red): telling the sampler the library's draw in place of the word written, in the plain
// loop or in the record's, turns the history check under "law" red; keeping a miss with chance a in place of
// (a - p*) / (1 - p*) turns the chi-squares red (here and in test-speculative-lenient, which holds the fall-backs'
// mutants).

#include "speculative-pq-record-run.h"

#include <map>

using namespace pq_test;

namespace {

common_spec_lenient dial(double gamma) {
    common_spec_lenient lp;
    lp.on    = true;
    lp.inf   = std::isinf(gamma);
    lp.gamma = lp.inf ? 1.0 : gamma;
    return lp;
}

bool same(const run_out & a, const run_out & b) {
    if (a.text != b.text || a.rounds.size() != b.rounds.size()) {
        return false;
    }
    for (size_t i = 0; i < a.rounds.size(); i++) {
        if (a.rounds[i].draft != b.rounds[i].draft || a.rounds[i].ids != b.rounds[i].ids) {
            return false;
        }
    }
    return true;
}

double chi2_crit(int df) { // alpha 1e-4, Wilson-Hilferty
    const double z = 3.719, k = df;
    return k * std::pow(1.0 - 2.0 / (9.0 * k) + z * std::sqrt(2.0 / (9.0 * k)), 3);
}

// one check of [w*] after the prompt, `n` times on the same logits through common_spec_lenient_accept_n
void test_law(const llama_tokens & prompt) {
    common_params params;
    params.model.path   = g_path;
    params.n_gpu_layers = g_ngl;
    params.fit_params   = false;
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    params.warmup   = false;
    params.n_ctx    = 512;
    params.n_batch  = 512;
    params.n_ubatch = 64;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP }; // the fixture's head loads with it
    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    if (ctx == nullptr) {
        check(false, "law: the fixture loads");
        return;
    }
    // the candidates at the prompt's end. The fixture's library is near uniform (19 candidates at about 0.053 at any
    // temperature), so a logit bias on five of them (part of the sampler's chain, ahead of the rest) makes a spread
    common_params_sampling sp;
    sp.temp  = 1.0f;
    sp.top_k = 20;
    sp.top_p = 0.95f;
    sp.min_p = 0.05f;
    sp.seed  = 42;
    std::vector<std::pair<llama_token, float>> p;
    std::vector<llama_token> ws; // w*: the library's favourite, and its third
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int pass = 0; pass < 2; pass++) {
        common_batch_clear(batch);
        for (size_t i = 0; i < prompt.size(); i++) {
            common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
        }
        llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);
        if (llama_decode(ctx, batch) != 0) {
            break;
        }
        common_sampler_ptr probe(common_sampler_init(llama_get_model(ctx), sp));
        common_sampler_sample(probe.get(), ctx, (int) prompt.size() - 1); // a batch index, not an output's
        const llama_token_data_array * cur = common_sampler_get_candidates(probe.get(), true);
        p.clear();
        for (size_t i = 0; i < cur->size; i++) {
            if (cur->data[i].p > 0.0f) {
                p.push_back({ cur->data[i].id, cur->data[i].p });
            }
        }
        if (pass == 0) {
            for (size_t i = 0; i < 5 && i < p.size(); i++) {
                sp.logit_bias.push_back({ p[i].first, 3.0f - 0.5f * (float) i });
            }
            continue;
        }
        printf("  law: %zu candidates, the top three at %.3f %.3f %.3f\n", p.size(),
               p.size() > 0 ? p[0].second : 0.0f, p.size() > 1 ? p[1].second : 0.0f, p.size() > 2 ? p[2].second : 0.0f);
        if (p.size() >= 3 && p[0].second <= 0.6f && p[2].second >= 0.05f) {
            ws = { p[0].first, p[2].first };
        }
    }
    check(!ws.empty(), "law: a spread of candidates to draw from");
    for (const llama_token w : ws) {
    // the prompt, then w*: outputs at the prompt's last word (the drafted word's position) and at w* (the word after)
    common_batch_clear(batch);
    for (size_t i = 0; i < prompt.size(); i++) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
    }
    common_batch_add(batch, w, (llama_pos) prompt.size(), { 0 }, true);
    llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);
    if (llama_decode(ctx, batch) != 0) {
        check(false, "law: the check decodes");
        break;
    }
    const std::vector<int32_t> idxs = { (int32_t) prompt.size() - 1, (int32_t) prompt.size() };
    double p_star = 0.0;
    for (const auto & c : p) {
        p_star = c.first == w ? c.second : p_star;
    }
    const int N = 60000;
    for (const double gamma : { 1.3, 2.0, (double) INFINITY }) {
        const auto lp = dial(gamma);
        common_sampler_ptr smpl(common_sampler_init(llama_get_model(ctx), sp));
        std::mt19937 rng;
        common_spec_lenient_seed(rng, 42);
        const common_spec_lenient_draw len = { &lp, &rng };
        const double a = lp.inf ? 1.0 : std::min(1.0, gamma * p_star);
        std::map<llama_token, int> obs;
        int dial_kept = 0;
        for (int n = 0; n < N; n++) {
            std::vector<int> by_dial;
            const llama_tokens ids = common_spec_lenient_accept_n(len, smpl.get(), ctx, idxs, { w }, &by_dial);
            obs[ids[0]]++;
            dial_kept += (int) by_dial.size();
        }
        double x2 = 0.0;
        int df = -1;
        bool stray = false;
        for (const auto & o : obs) {
            bool known = false;
            for (const auto & c : p) {
                known = known || c.first == o.first;
            }
            stray = stray || !known;
        }
        for (const auto & c : p) {
            const double e = (c.first == w ? a : c.second * (1.0 - a) / (1.0 - p_star)) * N;
            if (e <= 0.0) {
                stray = stray || obs.count(c.first) > 0;
                continue;
            }
            const double o = obs.count(c.first) ? obs[c.first] : 0;
            x2 += (o - e) * (o - e) / e;
            df++;
        }
        char what[200];
        snprintf(what, sizeof(what), "law: %s, p* %.3f: chi-square %.1f on %d df (critical %.1f), %d kept by the dial",
                 lp.describe().c_str(), p_star, x2, df, df > 0 ? chi2_crit(df) : 0.0, dial_kept);
        check(!stray && (df > 0 ? x2 < chi2_crit(df) : obs.size() == 1), what);

        // the sampler's history: with a presence penalty on the last word only (so heavy no word is drawn twice running),
        // the word after a kept w* (the bonus, drawn from the sampler that accepted it) is never w*. Told the draw
        // instead, it would be w* about as often as its chance there. (The penalty moves position 0's chances too, by
        // the word before, so the law is read above, without it; the fixture has no text, so prev_str cannot.)
        common_params_sampling sp_hist = sp;
        sp_hist.penalty_last_n  = 1;
        sp_hist.penalty_present = 50.0f;
        for (const bool through_record : { false, true }) { // the plain loop, then the p/q record's
            common_sampler_ptr hist(common_sampler_init(llama_get_model(ctx), sp_hist));
            FILE * f = std::tmpfile();
            common_spec_pq_record rec(f, 1, 0); // closes f
            int kept_then = 0, repeated = 0;
            for (int n = 0; n < N / 10; n++) {
                std::vector<int> by_dial;
                llama_tokens ids;
                if (through_record) {
                    ids = rec.accept_n(hist.get(), ctx, idxs, { w }, 0, (int) prompt.size() - 1, false, &len);
                    rec.check_end(0, false);
                    by_dial.resize(ids.size() == 2 && ids[0] == w ? 1 : 0); // a kept w*: the dial's or the draw's
                } else {
                    ids = common_spec_lenient_accept_n(len, hist.get(), ctx, idxs, { w }, &by_dial);
                }
                if (!by_dial.empty()) {
                    kept_then++;
                    repeated += ids.size() > 1 && ids[1] == w;
                }
            }
            check(kept_then > 0 && repeated == 0, lp.describe() + (through_record ? " (the record's loop)" : "") +
                  ": the sampler is told the word written (after " + std::to_string(kept_then) + " keeps of w*, w* "
                  "drawn next " + std::to_string(repeated) + " times)");
        }
    }
    }
    llama_batch_free(batch);
}

// the record's lines without their clock ("wall_us"), which differs run to run
std::vector<std::string> unclocked(const std::vector<std::string> & lines) {
    std::vector<std::string> out;
    for (std::string l : lines) {
        const size_t at = l.find("\"wall_us\":");
        if (at != std::string::npos) {
            l.erase(at, l.find_first_of(",}", at) - at);
        }
        out.push_back(l);
    }
    return out;
}

// the words the dial kept, named on the rounds' lines (not the tail's); each named word checked on every line
int count_len(const run_out & r, bool * ok) {
    int n = 0;
    for (size_t li = 0; li < r.lines.size(); li++) {
        const std::string & l = r.lines[li];
        const std::string f = field(l, "len");
        if (f.empty()) {
            continue;
        }
        const llama_tokens draft = [&] { llama_tokens t; for (double v : nums(field(l, "draft"))) t.push_back((llama_token) v); return t; }();
        const llama_tokens words = [&] { llama_tokens t; for (double v : nums(field(l, "words"))) t.push_back((llama_token) v); return t; }();
        const auto p = rows(field(l, "p"));
        for (double v : nums(f)) {
            const size_t i = (size_t) v;
            n += li < r.rounds.size();
            *ok = *ok && i < words.size() && i < draft.size() && i < p.size() && words[i] == draft[i] && p[i][0] != words[i];
        }
    }
    return n;
}

void test_words(const llama_tokens & prompt) {
    // oracle drafts (a text the same seed wrote), so checks keep words and miss after some
    const run_out base = run(1.0f, 42, prompt, "", false);
    const llama_tokens * oracle = &base.text;
    const run_out off = run(1.0f, 42, prompt, "", false, oracle);
    check(base.ok && off.ok && off.text.size() > 20, "the fixture writes at 1.0");

    const auto g1 = dial(1.0), g13 = dial(1.3), ginf = dial(INFINITY);
    const run_out on1 = run(1.0f, 42, prompt, "", false, oracle, &g1);
    check(same(off, on1) && on1.dial == 0, "gamma 1: the same tokens and rounds as off at 1.0 (the plain loop)");
    const run_out off_rec = run(1.0f, 42, prompt, "off-rec", true, oracle);
    const run_out on1_rec = run(1.0f, 42, prompt, "g1-rec", true, oracle, &g1);
    bool ok1 = true;
    check(same(off_rec, on1_rec) && unclocked(off_rec.lines) == unclocked(on1_rec.lines) && !off_rec.lines.empty() &&
          count_len(on1_rec, &ok1) == 0,
          "gamma 1: the same tokens, rounds and record lines as off (the record's loop)");

    const run_out t0_off = run(0.0f, 42, prompt, "", false, oracle);
    const run_out t0_on  = run(0.0f, 42, prompt, "", false, oracle, &g13);
    const run_out t0_inf = run(0.0f, 42, prompt, "", false, oracle, &ginf, /* force */ true);
    check(same(t0_off, t0_on) && same(t0_off, t0_inf) && t0_inf.dial == 0,
          "temperature 0: gamma 1.3 left off by the request test, and gamma inf forced on, write the same");

    // gamma 3 and inf: the fixture's library is near uniform (p* about 0.05) and the oracle's wrong words mostly
    // outside its top 20 (p* 0), so only inf is sure to act here; the law above covers the finite dials' chances
    for (const auto & lp : { dial(3.0), ginf }) {
        const run_out plain = run(1.0f, 42, prompt, "", false, oracle, &lp);
        const run_out rec   = run(1.0f, 42, prompt, "dial-rec", true, oracle, &lp);
        bool ok = true;
        const int n_len = count_len(rec, &ok);
        if (lp.inf) {
            check(plain.ok && rec.ok && !same(plain, off) && plain.dial > 0, lp.describe() + ": the dial keeps words (" +
                  std::to_string(plain.dial) + "), so the generation differs from off");
        }
        check(same(plain, rec) && n_len == plain.dial, lp.describe() + ": the record's loop writes the plain loop's words, "
              "and names the " + std::to_string(n_len) + " words the dial kept");
        check(rec.len_resumed && plain.len_resumed, lp.describe() + ": after a restored check's replay the dial's stream "
              "stands where the check left it, not back over rolls it used");
        check(ok, lp.describe() + ": each named word is the drafted word, written over another draw");
        std::vector<int> kept; // the rounds' check lines (the tail's restore and replay are not traced)
        for (size_t r = 0; r < rec.rounds.size() && r < rec.lines.size(); r++) {
            if (rec.lines[r].find("\"line\":\"check\"") != std::string::npos) {
                kept.push_back(std::atoi(field(rec.lines[r], "kept").c_str()));
            }
        }
        check(kept == rec.trace_kept, lp.describe() + ": the record's kept counts are the rate trace's");
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) {
            g_path = argv[++i];
        } else if (a == "-ngl" && i + 1 < argc) {
            g_ngl = std::atoi(argv[++i]);
        }
    }
    if (g_path.empty()) {
        fprintf(stderr, "usage: %s -m qwen4exp-moe-e64mtp.gguf [-ngl N]\n", argv[0]);
        return 1;
    }
    common_init();
    llama_backend_init();

    std::mt19937 rng(42);
    llama_tokens prompt;
    for (int i = 0; i < 37; i++) {
        prompt.push_back((llama_token) (1 + rng() % 100)); // the fixture has 128 words
    }
    test_law(prompt);
    test_words(prompt);

    llama_backend_free();
    printf("%s\n", g_failures ? "test-speculative-lenient-mtp: FAILED" : "test-speculative-lenient-mtp: OK");
    return g_failures ? 1 : 0;
}

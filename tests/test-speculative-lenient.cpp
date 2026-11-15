// The lenient check's rule (LLAMA_SPEC_LENIENT, common/speculative-lenient.h, FAIR-CHECK-PLAN L3), with no model: a
// sampler on a real vocabulary (gpt-2, vocab only) whose candidates are set by hand, so the rule reads the chances a
// draw would have left. The draws here are the test's own; the MTP fixture's test makes the library's.
//   parse     unset, "", "0", "off": off; "1": on at gamma 1; "1.3", "inf"; below 1, not a number, trailing junk,
//             nan: refused
//   extra     the chance a miss is kept anyway: (a - p*) / (1 - p*), a = min(1, gamma p*); 0 at gamma 1, p* 0 or 1
//   request   only at temperature > 0, no mirostat, no adaptive-p, backend sampling off, the switch on
//   law       the written word's frequencies are p' (w* gets a, any other x p(x) (1 - a) / (1 - p*)), by chi-square,
//             at gamma 1.3, 2 and inf, w* the library's favourite and a less likely word; gamma 1 never keeps a
//             miss and never touches its stream
//   falls     back to today's check: a grammar that refuses a word the draw could take (one refusing only words at
//             chance 0, or a lazy one awaiting its trigger, does not), a draw redone grammar first, the reasoning
//             budget forcing, a word the chain left out
// Mutation check: keeping a miss with chance a instead of (a - p*) / (1 - p*) turns "law" red; dropping the grammar
// test turns "falls" red.
#include "common.h"
#include "llama.h"
#include "reasoning-budget.h"
#include "sampling.h"
#include "speculative-lenient.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const std::string & what) {
    printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

static bool refused(const char * v) {
    try {
        common_spec_lenient_parse(v);
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

static common_spec_lenient dial(double gamma) {
    common_spec_lenient lp;
    lp.on    = true;
    lp.inf   = std::isinf(gamma);
    lp.gamma = lp.inf ? 1.0 : gamma;
    return lp;
}

// the sampler's candidates as a draw of `drawn` would leave them (the chain's p over `cand`)
struct fake_draw {
    std::vector<llama_token_data> data;
    void set(common_sampler * smpl, const std::vector<std::pair<llama_token, float>> & cand, llama_token drawn) {
        data.clear();
        int64_t sel = -1;
        for (const auto & c : cand) {
            if (c.first == drawn) {
                sel = (int64_t) data.size();
            }
            data.push_back({ c.first, std::log(std::max(c.second, 1e-30f)), c.second });
        }
        *common_sampler_get_candidates(smpl, false) = { data.data(), data.size(), sel, false };
    }
};

// chi-square's critical value at alpha 1e-4 (Wilson-Hilferty)
static double chi2_crit(int df) {
    const double z = 3.719, k = df;
    return k * std::pow(1.0 - 2.0 / (9.0 * k) + z * std::sqrt(2.0 / (9.0 * k)), 3);
}

// sum (obs - exp)^2 / exp over the words; *df the words less one
static double chi2(const std::map<llama_token, int> & obs, const std::map<llama_token, double> & expect, int n, int * df) {
    double x = 0.0;
    for (const auto & e : expect) {
        const auto it = obs.find(e.first);
        const double o = it == obs.end() ? 0.0 : it->second, m = e.second * n;
        x += (o - m) * (o - m) / m;
    }
    for (const auto & o : obs) {
        if (expect.find(o.first) == expect.end()) {
            x += 1e9; // a word p' never writes
        }
    }
    *df = (int) expect.size() - 1;
    return x;
}

static void test_parse_extra_request() {
    check(!common_spec_lenient_parse(nullptr).on && !common_spec_lenient_parse("").on &&
          !common_spec_lenient_parse("0").on && !common_spec_lenient_parse("off").on, "parse: unset, \"\", \"0\", \"off\" are off");
    const auto g1 = common_spec_lenient_parse("1"), g13 = common_spec_lenient_parse("1.3"), gi = common_spec_lenient_parse("inf");
    check(g1.on && !g1.inf && g1.gamma == 1.0, "parse: \"1\" is on at gamma 1");
    check(g13.on && !g13.inf && std::fabs(g13.gamma - 1.3) < 1e-12 && g13.describe() == "gamma 1.3", "parse: \"1.3\"");
    check(gi.on && gi.inf && gi.describe() == "gamma inf", "parse: \"inf\"");
    check(refused("0.5") && refused("-2") && refused("abc") && refused("1.3x") && refused("nan") && refused("Infinity"),
          "parse: below 1, negative, not a number, trailing junk, nan, a spelled infinity refused");

    check(common_spec_lenient_extra(dial(1.0), 0.5) == 0.0 && common_spec_lenient_extra(dial(1.0), 0.01) == 0.0,
          "extra: gamma 1 keeps no miss");
    check(std::fabs(common_spec_lenient_extra(dial(1.3), 0.5) - 0.3) < 1e-12, "extra: gamma 1.3 at p* 0.5: (0.65 - 0.5) / 0.5");
    check(std::fabs(common_spec_lenient_extra(dial(1.3), 0.8) - 1.0) < 1e-12, "extra: gamma 1.3 at p* 0.8: a = 1, always");
    check(common_spec_lenient_extra(dial(1.3), 0.0) == 0.0 && common_spec_lenient_extra(dial(INFINITY), 0.0) == 0.0,
          "extra: a word at chance 0 is never kept");
    check(common_spec_lenient_extra(dial(INFINITY), 1.0) == 0.0 && common_spec_lenient_extra(dial(INFINITY), 0.001) == 1.0,
          "extra: inf keeps any word with chance; p* 1 cannot miss");
    check(common_spec_lenient_extra(common_spec_lenient(), 0.5) == 0.0, "extra: off keeps no miss");

    common_params_sampling sp;
    sp.temp = 1.0f;
    const auto on = dial(1.3);
    check(common_spec_lenient_request(on, sp, false), "request: temperature 1.0, the switch on");
    check(!common_spec_lenient_request(common_spec_lenient(), sp, false), "request: the switch off");
    check(!common_spec_lenient_request(on, sp, true), "request: backend sampling");
    common_params_sampling t0 = sp;
    t0.temp = 0.0f;
    check(!common_spec_lenient_request(on, t0, false), "request: temperature 0");
    common_params_sampling mir = sp;
    mir.mirostat = 2;
    check(!common_spec_lenient_request(on, mir, false), "request: mirostat");
    common_params_sampling ap = sp;
    ap.samplers.push_back(COMMON_SAMPLER_TYPE_ADAPTIVE_P);
    check(!common_spec_lenient_request(on, ap, false), "request: adaptive-p");
}

// the rule's law on a hand-set p: x drawn from p by the test, the dial's choice, the written word counted
static void test_law(llama_model * model, const std::vector<llama_token> & ids) {
    const std::vector<float> ps = { 0.45f, 0.25f, 0.15f, 0.10f, 0.05f };
    std::vector<std::pair<llama_token, float>> cand;
    for (size_t i = 0; i < ps.size(); i++) {
        cand.push_back({ ids[i], ps[i] });
    }
    common_params_sampling sp;
    common_sampler_ptr smpl(common_sampler_init(model, sp));
    fake_draw fd;
    std::mt19937 draw_rng(1234);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const int N = 200000;
    for (const double gamma : { 1.0, 1.3, 2.0, (double) INFINITY }) {
        for (const size_t wi : { (size_t) 0, (size_t) 2 }) {
            const llama_token w = ids[wi];
            const double p_star = ps[wi];
            const double a = std::isinf(gamma) ? 1.0 : std::min(1.0, gamma * p_star);
            std::map<llama_token, double> expect;
            for (size_t i = 0; i < ps.size(); i++) {
                const double e = i == wi ? a : ps[i] * (1.0 - a) / (1.0 - p_star);
                if (e > 0.0) {
                    expect[ids[i]] = e;
                }
            }
            const auto lp = dial(gamma);
            std::mt19937 rng;
            common_spec_lenient_seed(rng, 42);
            const std::mt19937 rng0 = rng;
            std::map<llama_token, int> obs;
            int kept_miss = 0;
            for (int n = 0; n < N; n++) {
                double r = u(draw_rng), acc = 0.0;
                llama_token x = ids.back();
                for (size_t i = 0; i < ps.size(); i++) {
                    acc += ps[i];
                    if (r < acc) {
                        x = ids[i];
                        break;
                    }
                }
                fd.set(smpl.get(), cand, x);
                const uint64_t r0 = common_sampler_get_draw_state(smpl.get()).n_resample;
                const bool keep = common_spec_lenient_keep(lp, smpl.get(), w, x, r0, rng);
                kept_miss += keep;
                obs[keep ? w : x]++;
            }
            int df = 0;
            const double x2 = chi2(obs, expect, N, &df);
            char what[160];
            snprintf(what, sizeof(what), "law: %s, w* at p %.2f: chi-square %.1f on %d df (critical %.1f), %d misses kept",
                     lp.describe().c_str(), p_star, x2, df, df > 0 ? chi2_crit(df) : 0.0, kept_miss);
            check(df > 0 ? x2 < chi2_crit(df) : obs.size() == 1, what);
            if (gamma == 1.0) {
                check(kept_miss == 0 && rng == rng0, "law: gamma 1 keeps no miss and never rolls its stream");
            }
        }
    }
}

static void test_falls_back(llama_model * model, const std::vector<llama_token> & ids) {
    const llama_token a = ids[0], b = ids[1], c = ids[2], d = ids[3];
    const auto inf = dial(INFINITY);
    std::mt19937 rng(7);
    fake_draw fd;
    auto keep_at = [&](common_sampler * smpl, const std::vector<std::pair<llama_token, float>> & cand, llama_token w,
                       llama_token x, int64_t r0_shift = 0) {
        fd.set(smpl, cand, x);
        const uint64_t r0 = common_sampler_get_draw_state(smpl).n_resample + r0_shift;
        return common_spec_lenient_keep(inf, smpl, w, x, r0, rng);
    };
    {
        common_params_sampling sp;
        common_sampler_ptr smpl(common_sampler_init(model, sp));
        check(keep_at(smpl.get(), { { a, 0.6f }, { b, 0.4f } }, b, a), "no grammar: inf keeps the drafted word");
        check(!keep_at(smpl.get(), { { a, 0.6f }, { b, 0.4f } }, c, a), "a word the chain left out is never kept");
        check(!keep_at(smpl.get(), { { a, 0.6f }, { b, 0.4f } }, b, a, -1), "a draw redone grammar first: today's check");
        check(!keep_at(smpl.get(), { { a, 0.6f }, { b, 0.4f } }, a, a), "the draw itself is no miss (today's keep)");
    }
    {
        common_params_sampling sp;
        sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, "root ::= [abc] [abc]");
        common_sampler_ptr smpl(common_sampler_init(model, sp));
        check(keep_at(smpl.get(), { { a, 0.5f }, { b, 0.3f }, { c, 0.2f } }, b, a),
              "a grammar refusing none of the candidates: the dial runs");
        check(keep_at(smpl.get(), { { a, 0.5f }, { b, 0.5f }, { d, 0.0f } }, b, a),
              "a grammar refusing only a word at chance 0: the dial runs");
        check(!keep_at(smpl.get(), { { a, 0.5f }, { b, 0.3f }, { d, 0.2f } }, b, a),
              "a grammar refusing a word the draw could take: today's check");
    }
    {
        common_params_sampling sp;
        sp.grammar      = common_grammar(COMMON_GRAMMAR_TYPE_USER, "root ::= \"ab\"");
        sp.grammar_lazy = true;
        sp.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "zzz", LLAMA_TOKEN_NULL });
        common_sampler_ptr smpl(common_sampler_init(model, sp));
        check(keep_at(smpl.get(), { { a, 0.5f }, { b, 0.3f }, { c, 0.2f } }, c, a),
              "a lazy grammar awaiting its trigger: the dial runs");
    }
    {
        common_params_sampling sp;
        sp.reasoning_budget_tokens = 0;
        sp.reasoning_budget_start  = { a };
        sp.reasoning_budget_end    = { { b } };
        sp.reasoning_budget_forced = { b };
        common_sampler_ptr smpl(common_sampler_init(model, sp));
        check(keep_at(smpl.get(), { { c, 0.6f }, { d, 0.4f } }, d, c), "the reasoning budget idle: the dial runs");
        common_sampler_accept(smpl.get(), a, true); // the start, with no budget: forcing at once
        check(common_sampler_get_draw_state(smpl.get()).budget == REASONING_BUDGET_FORCING, "the budget is forcing");
        check(!keep_at(smpl.get(), { { c, 0.6f }, { d, 0.4f } }, d, c), "the reasoning budget forcing: today's check");
    }
}

static void test_restore() {
    std::mt19937 rng;
    common_spec_lenient_seed(rng, 5);
    const std::mt19937 before = rng;
    std::optional<std::mt19937> after;
    rng(); rng();
    const std::mt19937 check_end = rng;
    common_spec_lenient_restore(rng, before, after);
    check(rng == before && after && *after == check_end, "restore: the stream goes back for the replay, the check's end kept");
    rng();
    common_spec_lenient_restore(rng, before, after); // a replay that restored again
    check(rng == before && *after == check_end, "restore: a second restore keeps the first check's end");
    common_spec_lenient_replayed(rng, after);
    check(rng == check_end && !after, "replayed: the stream stands where the check left it");
    common_spec_lenient_replayed(rng, after);
    check(rng == check_end, "replayed: no restore pending, nothing moves");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s ggml-vocab-gpt-2.gguf\n", argv[0]);
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> ids;
    for (const char * s : { "a", "b", "c", "d", "e" }) {
        ids.push_back(common_tokenize(vocab, s, false, false)[0]);
    }

    test_parse_extra_request();
    test_law(model, ids);
    test_falls_back(model, ids);
    test_restore();

    llama_model_free(model);
    llama_backend_free();
    printf("%s\n", g_failures ? "test-speculative-lenient: FAILED" : "test-speculative-lenient: OK");
    return g_failures ? 1 : 0;
}

// The p/q record's wider chances for DFlash 2 (common/speculative-pq-record.h, FAIR-CHECK-PLAN step 3d): beside each
// lattice, the top TOP_L (id, logit) of every drafted position's whole-vocabulary logits and their log-sum-exp. No test
// model has DFlash 2's selector, so the made-up lattice of test-speculative-pq-record-mtp.cpp is handed over with real
// logits: the library's own, from the MTP fixture's check the round before (qwen4exp-moe-e64mtp.gguf, 128 words, so
// the top 64 is half the vocabulary and the log-sum-exp is not theirs alone):
//   logits_top   against a full sort and a double log-sum-exp: ids, order, ties by id, a masked word, one word far above
//                the rest, a row shorter than k, no row and a row of nothing (empty, lse NaN)
//   words        at temperature 0, and at 1.0 seeded, the same words and rounds with the record on (logits handed) as off
//   lines        a lattice handed with logits carries "lse" and "top": a row a lattice position, exactly the plain way's
//                ids and logits (written %.9g, a float read back whole), lse within 1e-6, its 64 words' chances summing
//                under 1; a lattice handed without (the round before checked too few words, the restore and replay's)
//                carries neither
// Mutation check (FAIR-CHECK-PLAN): the log-sum-exp over the top TOP_L only turns logits_top and "lines" red; reading
// the rows from output i + 1 instead of idx0 + i (the DFlash 2 hook's beg + 1) turns "lines" red.

#include "speculative-pq-record-run.h"

using namespace pq_test;

namespace {

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

bool lse_near(double got, double want, int n) {
    return std::isfinite(got) && std::fabs(got - want) <= 1e-6 + n * std::exp(-30.0);
}

void test_logits_top() {
    const int k = common_spec_pq_record::TOP_L;
    double lse = 0.0;
    bool ok = common_spec_pq_record::logits_top(nullptr, 10, k, lse).empty() && std::isnan(lse);
    std::vector<float> none(50, -INFINITY);
    ok = ok && common_spec_pq_record::logits_top(none.data(), (int) none.size(), k, lse).empty() && std::isnan(lse);
    check(ok, "logits_top: no row, or a row of masked words: nothing, lse NaN");

    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    ok = true;
    int n_short = 0;
    for (int trial = 0; trial < 80; trial++) {
        const int n = trial < 2 ? 10 + trial * 20 : 1 + (int) (rng() % 4000); // the first two shorter than 64
        n_short += n < k;
        std::vector<float> l(n);
        for (auto & v : l) {
            v = nd(rng);
        }
        if (trial % 4 == 1) {
            l[rng() % n] = -INFINITY;
        } else if (trial % 4 == 2 && n > 3) {
            l[1] = l[2] = 20.0f; // a tie at the top: the lower id first
        } else if (trial % 4 == 3) {
            l[n - 1] = 200.0f;   // one word far above the rest: the rest are under SKIP
        }
        double want_lse = 0.0;
        auto want = plain_top(l.data(), n, k, want_lse);
        while (!want.empty() && !std::isfinite(want.back().second)) {
            want.pop_back();     // a masked word is never among the top
        }
        const auto got = common_spec_pq_record::logits_top(l.data(), n, k, lse);
        bool same_ids = got.size() == want.size(); // a far-above word's row too: the sum skips, the top does not
        for (size_t i = 0; same_ids && i < got.size(); i++) {
            same_ids = got[i].first == want[i].first && got[i].second == want[i].second;
        }
        ok = ok && same_ids && lse_near(lse, want_lse, n);
    }
    check(ok && n_short > 0, "logits_top: the top 64 of a full sort (ties by id), lse within 1e-6, over 80 spreads (" +
          std::to_string(n_short) + " shorter than 64)");
}

void test_wide_lines(const run_out & on, const std::string & t) {
    const size_t n = on.rounds.size();
    check(on.lines.size() == n + 2 && on.wide.size() == n, t + ": one line a round, a handover a round");
    if (on.lines.size() != n + 2 || on.wide.size() != n) {
        return;
    }
    bool exact = true, narrow_ok = true, under = true;
    int n_wide = 0, n_narrow = 0;
    for (size_t r = 0; r < n; r++) {
        if (r == 3) {
            continue; // its lattice came from another position: on no line
        }
        const std::string lat = field(on.lines[r], "lat");
        const auto lse = nums(field(lat, "lse"));
        const auto top = rows(field(lat, "top"));
        if (!on.wide[r]) {
            n_narrow++;
            narrow_ok = narrow_ok && !lat.empty() && field(lat, "lse").empty() && field(lat, "top").empty();
            continue;
        }
        n_wide++;
        if (lse.size() != (size_t) N_POS || top.size() != (size_t) N_POS) {
            exact = false;
            continue;
        }
        for (int i = 0; i < N_POS; i++) {
            const auto & want = on.wide_top[r][i];
            exact = exact && top[i].size() == 2 * want.size() && want.size() == (size_t) common_spec_pq_record::TOP_L &&
                    lse_near(lse[i], on.wide_lse[r][i], 128);
            double sum = 0.0;
            for (size_t c = 0; exact && c < want.size(); c++) {
                exact = top[i][2 * c] == want[c].first && (float) top[i][2 * c + 1] == want[c].second;
                sum += std::exp(top[i][2 * c + 1] - lse[i]);
            }
            under = under && sum < 0.999;
        }
    }
    check(exact && n_wide > 0, t + ": " + std::to_string(n_wide) + " lattices carry lse and the top 64 a position, "
          "exactly the plain way's (ids, logits, lse within 1e-6)");
    check(under, t + ": each position's 64 words' chances sum under 1 (lse is the whole vocabulary's)");
    const std::string rs = field(on.lines[n], "lat");
    check(narrow_ok && n_narrow > 0 && !rs.empty() && field(rs, "lse").empty(), t + ": " + std::to_string(n_narrow) +
          " lattices handed without logits, and the restoring check's, carry neither lse nor top");
}

void test_prompt(const llama_tokens & prompt) {
    printf("prompt of %zu tokens, ngl %d\n", prompt.size(), g_ngl);
    for (float temp : { 0.0f, 1.0f }) {
        const std::string t = "temperature " + std::to_string(temp).substr(0, 3);
        const run_out base = run(temp, 42, prompt, "", false);
        const run_out off  = run(temp, 42, prompt, "", false, &base.text);
        const run_out on   = run(temp, 42, prompt, "wide", true, &base.text);
        check(base.ok && off.ok && on.ok && same(on, off), t + ": the same words and rounds with the record on, "
              "logits handed (" + std::to_string(on.rounds.size()) + " rounds)");
        if (on.ok) {
            test_wide_lines(on, t);
        }
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

    test_logits_top();
    std::mt19937 rng(42);
    for (int n : { 37, 120 }) {
        llama_tokens prompt;
        for (int i = 0; i < n; i++) {
            prompt.push_back((llama_token) (1 + rng() % 100)); // the fixture has 128 words
        }
        test_prompt(prompt);
    }

    llama_backend_free();
    printf("%s\n", g_failures ? "test-speculative-pq-record-wide: FAILED" : "test-speculative-pq-record-wide: OK");
    return g_failures ? 1 : 0;
}

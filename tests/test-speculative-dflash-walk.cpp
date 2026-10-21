// The DFlash 2 selector walk (common/speculative-dflash-walk.h, CHAIN-PLAN step 1) on made-up lattices, since no
// test model has a selector: at p-min 0 with the trace on it reports every position's confidence; the words are
// the same with the trace on or off; the p-min cut lands on the first word under it, which is walked but not
// drafted, and the trace walks on into the tail; n_min clears a short draft; and over generated lattices the
// walk's invariants hold (a draft at any p-min is a prefix of the p-min 0 draft, every confidence in (0, 1]).
// With the chain stop (common/speculative-chain.h, CHAIN-PLAN step 2): the cut lands on the first word that takes the
// product under the threshold, a product exactly at it is kept, the earlier of p-min and the chain wins (p-min named
// when both fall on one word), and over generated lattices every drafted prefix's product holds the threshold.

#include "speculative-dflash-walk.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_failures++; } } while (0)

using trace_t = common_speculative_rate_trace;
using words_t = std::vector<llama_token>;

// rows of [top_k ids][top_k x top_k scores][pad], as the draft's h_nextn holds them
struct lattice {
    int32_t top_k = 0, n_pos = 0;
    size_t  stride = 0;
    std::vector<float> data;

    lattice(int32_t top_k, int32_t n_pos, int32_t pad = 3)
        : top_k(top_k), n_pos(n_pos), stride((size_t) (top_k + top_k * top_k + pad)), data(stride * n_pos, -7.0f) {
        for (int32_t i = 0; i < n_pos; i++) {
            for (int32_t k = 0; k < top_k; k++) {
                id(i, k) = (float) (100 + 10 * i + k);
            }
        }
    }
    float & id(int32_t pos, int32_t k) { return data[pos * stride + k]; }
    float & score(int32_t pos, int32_t pred, int32_t k) { return data[pos * stride + top_k + pred * top_k + k]; }
    common_dflash2_lattice view() const { return { data.data(), stride, n_pos, top_k }; }
};

static trace_t::stop_reason walk(const lattice & l, float p_min, int32_t n_min, words_t & words,
        common_dflash2_walk_seen * seen, double chain_p = 0.0) {
    return common_dflash2_walk(l.view(), common_speculative_cut(p_min, chain_p), n_min, words, seen);
}

// the product of the first n confidences, in double as the chain runs it
static double product(const std::vector<float> & conf, size_t n) {
    double prod = 1.0;
    for (size_t i = 0; i < n && i < conf.size(); i++) {
        prod *= (double) conf[i];
    }
    return prod;
}

static bool near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

static bool near_all(const std::vector<float> & got, const std::vector<float> & want) {
    if (got.size() != want.size()) {
        return false;
    }
    for (size_t i = 0; i < got.size(); i++) {
        if (!near(got[i], want[i])) {
            return false;
        }
    }
    return true;
}

// two candidates; after candidate 0 the scores (0, ln 3) pick candidate 1 at 0.75, after candidate 1 the scores
// (ln 9, 0) pick candidate 0 at 0.9, so the walk alternates (a walk that ignored the predecessor would not);
// position 3 (seen after candidate 1) is a tie (0, 0): candidate 0 at 0.5
static lattice crafted() {
    lattice l(2, 7);
    for (int32_t i = 0; i < 7; i++) {
        l.score(i, 0, 0) = 0.0f; l.score(i, 0, 1) = std::log(3.0f);
        l.score(i, 1, 0) = std::log(9.0f); l.score(i, 1, 1) = 0.0f;
    }
    l.score(3, 1, 0) = 0.0f;
    return l;
}

static void test_p_min_zero_reports_every_position() {
    const lattice l = crafted();
    words_t on, off;
    common_dflash2_walk_seen seen;
    const auto why = walk(l, 0.0f, 0, on, &seen);
    walk(l, 0.0f, 0, off, nullptr);
    CHECK(why == trace_t::STOP_CAP && seen.stop == trace_t::STOP_CAP, "p-min 0 never cuts: cap");
    CHECK(seen.conf.size() == 7 && seen.tail.empty(), "p-min 0 with the trace on: 7 confidences, no tail (got %zu, %zu)",
          seen.conf.size(), seen.tail.size());
    CHECK(near_all(seen.conf, {0.75f, 0.9f, 0.75f, 0.5f, 0.75f, 0.9f, 0.75f}), "each word's softmax at its pick");
    CHECK(on == words_t({101, 110, 121, 130, 141, 150, 161}), "the walk follows its picks (alternates, ties to the first)");
    CHECK(on == off, "the same words with the trace on and off");
}

static void test_cut_lands_on_the_first_word_under_p_min() {
    const lattice l = crafted();
    words_t on, off;
    common_dflash2_walk_seen seen;
    const auto why = walk(l, 0.6f, 0, on, &seen);
    walk(l, 0.6f, 0, off, nullptr);
    CHECK(why == trace_t::STOP_PMIN && seen.stop == trace_t::STOP_PMIN, "p-min 0.6 cuts at position 3's 0.5");
    CHECK(on == words_t({101, 110, 121}) && on == off, "three words drafted, trace on or off (got %zu, %zu)", on.size(), off.size());
    CHECK(near_all(seen.conf, {0.75f, 0.9f, 0.75f, 0.5f}), "conf holds the dropped word's confidence last");
    CHECK(near_all(seen.tail, {0.75f, 0.9f, 0.75f}), "the tail walks on from the dropped word's pick (%zu)", seen.tail.size());

    words_t first;
    common_dflash2_walk_seen seen0;
    walk(l, 0.8f, 0, first, &seen0);
    CHECK(first.empty() && seen0.conf.size() == 1 && seen0.tail.size() == 6 && seen0.stop == trace_t::STOP_PMIN,
          "p-min 0.8 drops the first word: nothing drafted, one confidence, six in the tail");

    words_t edge;
    walk(l, 0.5f, 0, edge, nullptr); // position 3's tie is exactly 1 / (1 + 1)
    CHECK(edge.size() == 7, "a word at exactly p-min is kept (p < p_min cuts), got %zu", edge.size());
}

static void test_n_min_clears_the_draft() {
    const lattice l = crafted();
    words_t on, off;
    common_dflash2_walk_seen seen;
    walk(l, 0.6f, 4, on, &seen);
    walk(l, 0.6f, 4, off, nullptr);
    CHECK(on.empty() && off.empty(), "three words under n_min 4: cleared");
    CHECK(seen.conf.size() == 4 && seen.tail.size() == 3 && seen.stop == trace_t::STOP_PMIN,
          "the trace still sees the walk and its stop");
    words_t kept;
    walk(l, 0.6f, 3, kept, nullptr);
    CHECK(kept.size() == 3, "three words at n_min 3 stay");
}

static void test_chain_cuts_at_the_first_product_under_the_threshold() {
    const lattice l = crafted();
    common_dflash2_walk_seen all;
    words_t full;
    walk(l, 0.0f, 0, full, &all); // the walk's own confidences: 0.75, 0.9, 0.75, 0.5, ... as floats
    // products 0.75, 0.675, 0.506, 0.253, ...: a threshold of 0.6 cuts at position 2
    words_t on, off;
    common_dflash2_walk_seen seen;
    const auto why = walk(l, 0.0f, 0, on, &seen, 0.6);
    walk(l, 0.0f, 0, off, nullptr, 0.6);
    CHECK(why == trace_t::STOP_CHAIN && seen.stop == trace_t::STOP_CHAIN, "a chain of 0.6 stops the draft: chain");
    CHECK(on == words_t({101, 110}) && on == off, "two words drafted, trace on or off (got %zu, %zu)", on.size(), off.size());
    CHECK(near_all(seen.conf, {0.75f, 0.9f, 0.75f}) && near_all(seen.tail, {0.5f, 0.75f, 0.9f, 0.75f}),
          "conf ends on the word the chain dropped; the tail walks on (%zu, %zu)", seen.conf.size(), seen.tail.size());

    // the exact edge: a threshold equal to a product keeps that word (the draft goes on while product >= threshold)
    for (size_t n = 1; n <= 3; n++) {
        words_t edge;
        common_dflash2_walk_seen seen_edge;
        const auto why_edge = walk(l, 0.0f, 0, edge, &seen_edge, product(all.conf, n));
        CHECK(edge.size() == n && why_edge == trace_t::STOP_CHAIN, "a threshold exactly at the product of %zu words keeps "
              "them all and cuts the next (got %zu)", n, edge.size());
    }
    words_t one;
    walk(l, 0.0f, 0, one, nullptr, 1.0); // the first word's 0.75 is under 1: nothing drafted
    CHECK(one.empty(), "a threshold of 1 drops a first word under 1");

    words_t none;
    common_dflash2_walk_seen seen_off;
    walk(l, 0.0f, 0, none, &seen_off, 0.0);
    CHECK(none == full && seen_off.stop == trace_t::STOP_CAP, "a threshold of 0 is off: the whole draft");
}

static void test_the_earlier_of_p_min_and_the_chain_wins() {
    const lattice l = crafted();
    struct { float p_min; double chain; size_t words; trace_t::stop_reason stop; const char * what; } cases[] = {
        { 0.6f, 0.52, 2, trace_t::STOP_CHAIN, "the chain at position 2 before p-min's 0.5 at 3" },
        { 0.6f, 0.2,  3, trace_t::STOP_PMIN,  "p-min at position 3 before the chain's 0.19 at 4" },
        { 0.6f, 0.3,  3, trace_t::STOP_PMIN,  "both at position 3 (0.5 < 0.6, 0.253 < 0.3): p-min is named" },
        { 0.8f, 0.3,  0, trace_t::STOP_PMIN,  "p-min drops the first word" },
        { 0.0f, 0.8,  0, trace_t::STOP_CHAIN, "the chain drops the first word (0.75 < 0.8)" },
    };
    for (const auto & c : cases) {
        words_t on, off;
        common_dflash2_walk_seen seen;
        const auto why = walk(l, c.p_min, 0, on, &seen, c.chain);
        walk(l, c.p_min, 0, off, nullptr, c.chain);
        CHECK(why == c.stop && on.size() == c.words && on == off && seen.conf.size() == c.words + 1 &&
              seen.conf.size() + seen.tail.size() == 7, "p-min %.2f, chain %.2f: %s (got %zu words, stop %d)",
              (double) c.p_min, c.chain, c.what, on.size(), (int) why);
    }
    words_t short_draft;
    walk(l, 0.0f, 3, short_draft, nullptr, 0.6);
    CHECK(short_draft.empty(), "a chain-cut draft under n_min is cleared too");
}

// generated lattices, ties included: the invariants a correct walk keeps at any p-min, n_min and trace setting
static void test_generated_lattices() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-4.0f, 4.0f);
    int checked = 0, cut = 0, chained = 0;
    for (int t = 0; t < 2000; t++) {
        const int32_t top_k = 1 + (int32_t) (rng() % 8);
        const int32_t n_pos = (int32_t) (rng() % 9);
        lattice l(top_k, n_pos, (int32_t) (rng() % 4));
        for (int32_t i = 0; i < n_pos; i++) {
            for (int32_t p = 0; p < top_k; p++) {
                for (int32_t k = 0; k < top_k; k++) {
                    l.score(i, p, k) = rng() % 5 == 0 ? 1.0f : u(rng); // some ties
                }
            }
        }
        words_t full;
        common_dflash2_walk_seen seen_full;
        walk(l, 0.0f, 0, full, &seen_full);
        bool ok = full.size() == (size_t) n_pos && seen_full.conf.size() == (size_t) n_pos && seen_full.tail.empty();
        for (float p : seen_full.conf) {
            ok = ok && p > 0.0f && p <= 1.0f && (top_k > 1 || p == 1.0f);
        }
        const float p_min = rng() % 3 == 0 ? 0.0f : std::uniform_real_distribution<float>(0.05f, 1.0f)(rng);
        const int32_t n_min = (int32_t) (rng() % 3);
        const double chain = rng() % 2 == 0 ? 0.0 : std::uniform_real_distribution<double>(0.01, 1.0)(rng);
        words_t on, off;
        common_dflash2_walk_seen seen;
        const auto why = walk(l, p_min, n_min, on, &seen, chain);
        walk(l, p_min, n_min, off, nullptr, chain);
        ok = ok && on == off && seen.conf.size() + seen.tail.size() == (size_t) n_pos;
        std::vector<float> whole = seen.conf; // the walk past the stop follows the same picks: conf + tail is p-min 0's
        whole.insert(whole.end(), seen.tail.begin(), seen.tail.end());
        ok = ok && near_all(whole, seen_full.conf);
        const bool cut_here = why == trace_t::STOP_PMIN || why == trace_t::STOP_CHAIN;
        const size_t walked = seen.conf.size() - (cut_here ? 1 : 0);
        ok = ok && (on.empty() || on.size() == walked) && (on.size() >= (size_t) n_min || on.empty());
        ok = ok && std::equal(on.begin(), on.end(), full.begin());
        for (size_t j = 0; j < walked; j++) {
            ok = ok && seen.conf[j] >= p_min && (chain == 0.0 || product(seen.conf, j + 1) >= chain);
        }
        ok = ok && (why != trace_t::STOP_PMIN || seen.conf.back() < p_min);
        ok = ok && (why != trace_t::STOP_CHAIN || (chain > 0.0 && seen.conf.back() >= p_min &&
                                                   product(seen.conf, seen.conf.size()) < chain));
        ok = ok && (cut_here || seen.conf.size() == (size_t) n_pos);
        CHECK(ok, "lattice %d (top_k %d, %d positions, p-min %.3f, chain %.3f, n_min %d): an invariant broke", t, top_k,
              n_pos, (double) p_min, chain, n_min);
        checked++;
        cut += why == trace_t::STOP_PMIN;
        chained += why == trace_t::STOP_CHAIN;
    }
    CHECK(cut > 200 && cut < checked - 200, "the generated lattices cut on some and not others (%d of %d)", cut, checked);
    CHECK(chained > 200, "the chain cut some of them (%d of %d)", chained, checked);
}

int main() {
    test_p_min_zero_reports_every_position();
    test_cut_lands_on_the_first_word_under_p_min();
    test_n_min_clears_the_draft();
    test_chain_cuts_at_the_first_product_under_the_threshold();
    test_the_earlier_of_p_min_and_the_chain_wins();
    test_generated_lattices();
    if (g_failures) {
        fprintf(stderr, "test-speculative-dflash-walk: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-speculative-dflash-walk: OK\n");
    return 0;
}

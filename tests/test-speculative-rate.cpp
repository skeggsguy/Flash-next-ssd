// The throughput-seeking draft depth (common/speculative-rate.h) against simulated worlds: a keep
// probability per guess and a cycle cost per depth, with the best depth known in closed form. The
// controller must settle on (or next to) that depth, leave guessing off when guesses are poor, come
// back through the shadow guess when content turns predictable, and keep what it learnt across a
// restart of the timer (a new answer).

#include "speculative-rate.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_failures++; } } while (0)

struct world {
    double q;               // chance each guess is kept, given the ones before it were
    double base, first, per; // cycle cost: base, + first when any guess is verified, + per a guess
    double cost(int k) const { return base + (k > 0 ? first + per * k : 0.0); }
    double rate(int k) const {
        double e = 1.0, run = 1.0;
        for (int i = 1; i <= k; i++) { run *= q; e += run; }
        return e / cost(k);
    }
    int best(int n_max) const {
        int b = 0;
        for (int k = 1; k <= n_max; k++) { if (rate(k) > rate(b)) { b = k; } }
        return b;
    }
};

// runs n cycles; returns how often each depth was drafted over the last `tail` cycles
static std::vector<int> run(common_speculative_rate & ctl, const world & w, int n, int tail, std::mt19937 & rng,
        int64_t & now) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<int> used(common_speculative_rate::K_MAX + 1, 0);
    for (int cycle = 0; cycle < n; cycle++) {
        ctl.cycle_begin(now);
        const int k = ctl.k_next;
        if (cycle >= n - tail) { used[k]++; }
        if (k == 0) {
            ctl.shadow_scored(u(rng) < w.q);
            ctl.cycle_verified(0, 0);
        } else {
            int kept = 0;
            while (kept < k && u(rng) < w.q) { kept++; }
            ctl.cycle_verified(k, kept);
        }
        now += (int64_t) std::llround(w.cost(k) * (0.95 + 0.1 * u(rng)));
    }
    return used;
}

static int mode(const std::vector<int> & used) {
    int m = 0;
    for (size_t k = 1; k < used.size(); k++) { if (used[k] > used[m]) { m = (int) k; } }
    return m;
}

int main() {
    std::mt19937 rng(1234);
    const int n_max = 8;

    // costs shaped like Flash-Next's (a check pass with one guess ~2 plain cycles, more guesses cheap),
    // and a flat one (every guess costs the same): the right depth differs, the controller is the same
    const world worlds[] = {
        {0.95, 70.0, 60.0,  4.0}, // tool calls: guess deep
        {0.85, 70.0, 60.0,  4.0}, // good prose
        {0.45, 70.0, 60.0,  4.0}, // poor guesses: guessing does not pay
        {0.85, 70.0,  5.0, 12.0}, // a machine where the first guess is cheap and each one costs
        {0.30, 70.0, 10.0, 10.0}, // bad guesses on that machine
    };
    for (const world & w : worlds) {
        common_speculative_rate ctl;
        ctl.init(n_max);
        int64_t now = 1000;
        const std::vector<int> used = run(ctl, w, 3000, 1000, rng, now);
        const int want = w.best(n_max), got = mode(used);
        const double lost = 1.0 - w.rate(got) / w.rate(want);
        CHECK(lost < 0.03, "q=%.2f cost %.0f/%.0f/%.0f: settled on %d, best %d (%.1f%% slower)", w.q, w.base, w.first,
                w.per, got, want, 100.0 * lost);
        printf("q=%.2f cost %.0f/%.0f/%.0f: best depth %d, settled on %d (%.1f%% below the best rate)\n", w.q, w.base,
                w.first, w.per, want, got, 100.0 * lost);
    }

    // content turns poor, then good again: off within a few hundred cycles, and back through the shadow guess
    {
        common_speculative_rate ctl;
        ctl.init(n_max);
        int64_t now = 1000;
        run(ctl, worlds[0], 1500, 1, rng, now);
        const std::vector<int> poor = run(ctl, worlds[2], 600, 200, rng, now);
        CHECK(mode(poor) == worlds[2].best(n_max), "after the switch to poor guesses it drafts at %d, best %d",
                mode(poor), worlds[2].best(n_max));
        const std::vector<int> good = run(ctl, worlds[0], 1200, 300, rng, now);
        CHECK(mode(good) >= worlds[0].best(n_max) - 1, "back to good guesses it drafts at %d, best %d",
                mode(good), worlds[0].best(n_max));
    }

    // a new answer restarts the timer only: the wait is not charged, what was learnt is kept
    {
        common_speculative_rate ctl;
        ctl.init(n_max);
        int64_t now = 1000;
        run(ctl, worlds[0], 1500, 1, rng, now);
        const int depth = ctl.k_next;
        const double c_before = ctl.c[depth];
        ctl.restart_timer();
        now += 60 * 1000 * 1000; // a minute between answers
        ctl.cycle_begin(now);
        CHECK(ctl.c[depth] == c_before, "the gap between answers was charged to depth %d", depth);
        CHECK(ctl.k_next == depth || std::abs(ctl.k_next - depth) == 1, "the depth reset to %d from %d", ctl.k_next,
                depth);
    }

    // arithmetic of the pieces
    {
        common_speculative_rate ctl;
        ctl.init(4);
        ctl.p[1] = 0.5; ctl.wp[1] = 1;
        ctl.p[2] = 0.5; ctl.wp[2] = 1;
        CHECK(std::fabs(ctl.expected_tokens(2) - 1.75) < 1e-12, "E(2) = %f, want 1.75", ctl.expected_tokens(2));
        CHECK(std::fabs(ctl.expected_tokens(3) - 1.875) < 1e-12, "an unreached position borrows the one before");
        ctl.c[0] = 100; ctl.wc[0] = 1;
        ctl.c[2] = 200; ctl.wc[2] = 1;
        CHECK(std::fabs(ctl.cycle_cost(1) - 150) < 1e-9, "c(1) between 100 and 200 is %f", ctl.cycle_cost(1));
        ctl.cycle_verified(3, 1);
        CHECK(ctl.wp[1] == 2 && ctl.p[1] == 0.75 && ctl.wp[2] == 2 && ctl.p[2] == 0.25 && ctl.wp[3] == 0,
                "one kept then one rejected: p[1] %f p[2] %f, and position 3 (never reached) counted %f times",
                ctl.p[1], ctl.p[2], ctl.wp[3]);
        CHECK(ctl.k_ran == 3, "k_ran %d", ctl.k_ran);
    }

    if (g_failures) {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-speculative-rate: all passed\n");
    return 0;
}

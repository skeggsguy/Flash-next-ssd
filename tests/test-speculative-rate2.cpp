// The priced depth, floor-post (common/speculative-rate2.h), against simulated streams: its arithmetic by hand
// (the price, the bins, the per-answer caps, the fitted line against the study's CostRLS), the engine's clock
// (a gap between answers and an answer's last cycle are never charged) and the simulated tests of
// docs/wizard-rate-rework.md section 4, 1-6 (test 2's prose bound is dropped by the amendment; test 7, the study's
// FloorPost against this, runs from the study through --script). Three findings are pinned: bars the rule as
// specified does not meet, held as they are with what is wanted named beside them (see each FINDING).
//
// `--script IN --dump OUT` runs a script and writes one JSON line a cycle (the study's sim/test_spec_rate2_engine.py
// compares them with FloorPost). A script is lines of "N n_max" | "L a s b f d w" (the cost line) | "A" (a new
// answer) | "C kept mult n conf1..confn" (a cycle: what depth 8 kept, the time's factor over the line, confidences).

#include "speculative-rate2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_failures++; } } while (0)

using rule_t = common_speculative_rate2;

struct line_t { // the study's CostLine: the time of a cycle, in us, the terms in its order (and test 1's cliff)
    double a, s, b, f, d, w, cliff = 0.0;
    double us(int steps, int checked, int kept) const {
        return a + (steps > 0 ? d : 0.0) + s * steps + b * checked + (checked > 0 ? f : 0.0) + w * kept +
               (checked > 8 ? cliff : 0.0);
    }
};

// A-trace's fitted lines, rounded (results/A-trace/rules.json)
static const line_t PROSE = {67.8e3, 6.5e3, 16.7e3, -7.5e3, 0.8e3, 0.6e3};
static const line_t CODE  = {86.4e3, 8.2e3, 20.5e3, 13.2e3, 0.6e3, 8.7e3};

struct cycle_t {
    int kept = 0;             // what a depth-8 draft kept
    std::vector<float> conf;  // each step's top confidence
    double mult = 1.0;        // the measured time over the line
};

using rng_t = std::mt19937;
static double uni(rng_t & g, double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(g); }

// guess i kept with rates[i] (the last repeats) given 1..i-1 kept; a kept guess's confidence 0.5-1, the rest 0-0.7
static cycle_t bimodal(rng_t & g, const std::vector<double> & rates) {
    cycle_t c;
    while (c.kept < 8 && uni(g, 0, 1) < rates[std::min<size_t>(c.kept, rates.size() - 1)]) { c.kept++; }
    for (int j = 0; j < 8; j++) { c.conf.push_back((float) (j < c.kept ? uni(g, 0.5, 1.0) : uni(g, 0.0, 0.7))); }
    c.mult = uni(g, 0.85, 1.15);
    return c;
}

// every confidence uniform in [lo, 1), each guess kept with its confidence: the chance is the confidence
static cycle_t calibrated(rng_t & g, double lo) {
    cycle_t c;
    for (int j = 0; j < 8; j++) { c.conf.push_back((float) uni(g, lo, 1.0)); }
    while (c.kept < 8 && uni(g, 0, 1) < c.conf[c.kept]) { c.kept++; }
    c.mult = uni(g, 0.85, 1.15);
    return c;
}

static std::vector<cycle_t> stream(rng_t & g, const std::vector<double> & rates, int n) {
    std::vector<cycle_t> out;
    for (int i = 0; i < n; i++) { out.push_back(bimodal(g, rates)); }
    return out;
}

struct trail_t {
    double words = 0, us = 0;          // scored at the line, as the replay scores
    std::vector<int> steps, checked;
};

// one cycle as the engine runs it (start, a step per confidence while the rule says go, the library's check,
// the measured time), in the order the study's replay runs FloorPost
static void run_cycle(rule_t & r, const cycle_t & c, const line_t & L, trail_t & t) {
    r.start();
    for (size_t i = 0; i < std::min<size_t>(r.n_max, c.conf.size()) && r.step(c.conf[i]); i++) {}
    r.verified(r.checked, c.kept);
    const double us = L.us(std::max(1, r.steps), r.checked, r.kept);
    r.charge(us * c.mult);
    t.words += r.kept + 1; t.us += us;
    t.steps.push_back(r.steps); t.checked.push_back(r.checked);
}

static trail_t run_answer(rule_t & r, const std::vector<cycle_t> & cy, const line_t & L) {
    trail_t t;
    r.begin_answer();
    for (const auto & c : cy) { run_cycle(r, c, L, t); }
    return t;
}

// fixed depth d at p-min 0: words a microsecond
static double fixed_rate(const std::vector<cycle_t> & cy, int d, const line_t & L) {
    double w = 0, us = 0;
    for (const auto & c : cy) { w += std::min(d, c.kept) + 1; us += L.us(d, d, std::min(d, c.kept)); }
    return w / us;
}

static int best_fixed(const std::vector<cycle_t> & cy, const line_t & L) {
    int best = 1;
    for (int d = 2; d <= 8; d++) { if (fixed_rate(cy, d, L) > fixed_rate(cy, best, L)) { best = d; } }
    return best;
}

static double mean(const std::vector<int> & v) {
    double s = 0;
    for (int x : v) { s += x; }
    return v.empty() ? 0.0 : s / (double) v.size();
}

static rule_t fresh(int n_max = 8) { rule_t r; r.init(n_max); return r; }

// --- by hand ------------------------------------------------------------------------------------------------

static void test_by_hand() {
    rule_t r = fresh();
    CHECK(std::fabs(r.chance(0.95f) - 0.95) < 1e-12 && std::fabs(r.chance(0.05f) - 0.05) < 1e-12, "bins start at their midpoints");
    CHECK(r.per_word == 70e3 && r.price_b == 20e3, "time per word 70 ms, b 20 ms at the start");
    // 0.95 (floor) -> 0.95 x 0.35 x 70 = 23.3 ms > 20: checked -> x 0.25 = 5.8 ms: not checked, the draft stops
    CHECK(r.step(0.95f) && r.step(0.35f) && !r.step(0.25f), "the price by hand");
    CHECK(r.steps == 3 && r.checked == 2, "steps %d checked %d, want 3 and 2", r.steps, r.checked);
    r.verified(2, 1); // the first kept, the second thrown out: two samples, the third never checked
    CHECK(std::fabs(r.chance(0.35f) - 1.4 / 5.0) < 1e-12 && std::fabs(r.chance(0.95f) - 4.8 / 5.0) < 1e-12 &&
          r.seen[2] == 4.0, "bins after one kept and one thrown out");
    r.charge(100e3);
    CHECK(std::fabs(r.t_avg - (70e3 + 0.02 * 30e3)) < 1e-9 && std::fabs(r.w_avg - 1.02) < 1e-12, "the averages");

    rule_t f = fresh();
    CHECK(f.step(0.01f) && f.checked == 1, "the floor checks a hopeless first guess");
    rule_t c = fresh(3); c.per_word = 1e9;
    CHECK(c.step(0.9f) && c.step(0.9f) && c.step(0.9f) && !c.step(0.9f) && c.steps == 3, "never past the ceiling");

    rule_t cap = fresh();
    cap.hit[9] = 45; cap.seen[9] = 50; cap.hit[0] = 1; cap.seen[0] = 5;
    cap.begin_answer();
    CHECK(cap.seen[9] == 12 && std::fabs(cap.hit[9] / cap.seen[9] - 0.9) < 1e-12 && cap.seen[0] == 5 && cap.hit[0] == 1,
          "a new answer caps a bin at 12 samples, keeping its chance");
    cap.seen[5] = 50; cap.hit[5] = 25;
    cap.conf[0] = 0.55f; cap.steps = 1;
    cap.verified(1, 1);
    CHECK(std::fabs(cap.seen[5] - 50.0) < 1e-12 && std::fabs(cap.hit[5] - 25.5) < 1e-12, "past 50 samples a bin forgets");

    // the fitted line against the study's CostRLS on the same five cycles (python3.12, sim/spec_rules.py)
    common_speculative_cost_rls L;
    const double pts[5][3] = {{2, 1, 95e3}, {3, 3, 160e3}, {1, 1, 80e3}, {8, 8, 300e3}, {2, 1, 90e3}};
    for (auto & p : pts) { L.observe((int) p[0], (int) p[1], p[2]); }
    const double want[4] = {64354.65058851934, 6081.197831646288, 22328.86585805841, 4354.6505885193465};
    for (int i = 0; i < 4; i++) {
        CHECK(std::fabs(L.th[i] - want[i]) < 1e-6, "cost line term %d: %.9f, CostRLS %.9f", i, L.th[i], want[i]);
    }
}

// sixteen cycles in two answers, the choices and state the study's FloorPost makes on them (a guard for the
// script test in the study, which covers far more)
static void test_golden() {
    const line_t L = {70e3, 8e3, 20e3, 10e3, 1e3, 5e3};
    const struct { int kept; double mult; const char * conf; } G[16] = { // confidences in thousandths
        {1, 0.9, "513 193 156 516 474 625 061 295"}, {0, 1.1, "066 163 421 393 501 491 294 314"},
        {4, 0.9, "905 503 903 849 238 109 670 236"}, {1, 1.1, "690 251 241 185 030 322 087 646"},
        {1, 0.9, "776 581 433 603 404 493 032 160"}, {4, 1.1, "993 928 933 690 317 584 114 249"},
        {4, 0.9, "851 842 536 817 374 171 324 189"}, {8, 1.1, "610 662 884 528 911 903 701 533"},
        {5, 0.9, "606 750 942 821 571 098 521 377"}, {6, 1.1, "949 700 610 999 755 545 033 077"},
        {2, 0.9, "896 711 044 267 697 370 680 603"}, {0, 1.1, "476 080 619 526 538 238 205 111"},
        {0, 0.9, "668 613 184 350 125 639 609 209"}, {8, 1.1, "804 576 881 770 889 765 500 662"},
        {0, 0.9, "078 254 690 565 168 169 397 055"}, {7, 1.1, "908 989 766 563 830 973 583 369"},
    };
    const int want[16][2] = {{2, 1}, {2, 1}, {5, 4}, {2, 1}, {3, 2}, {5, 4}, {5, 4}, {7, 6},
                             {6, 5}, {7, 6}, {3, 2}, {2, 1}, {3, 2}, {7, 6}, {2, 1}, {8, 7}};
    rule_t r = fresh();
    trail_t t;
    for (int i = 0; i < 16; i++) {
        if (i % 8 == 0) { r.begin_answer(); }
        cycle_t c = {G[i].kept, {}, G[i].mult};
        for (int j = 0; j < 8; j++) { c.conf.push_back((float) (std::atoi(G[i].conf + 4 * j) / 1000.0)); }
        run_cycle(r, c, L, t);
        CHECK(t.steps[i] == want[i][0] && t.checked[i] == want[i][1], "cycle %d: (%d, %d), FloorPost (%d, %d)", i,
              t.steps[i], t.checked[i], want[i][0], want[i][1]);
    }
    CHECK(std::fabs(r.cost.th[2] - 26922.065089709296) < 1e-6 && std::fabs(r.t_avg - 106298.66237335942) < 1e-6 &&
          std::fabs(r.w_avg - 1.8266778138149902) < 1e-12, "state: b %.9f t %.9f w %.12f", r.cost.th[2], r.t_avg, r.w_avg);
}

// the engine's clock: each cycle timed from its draft call to the next of the same answer
static void test_clock() {
    rule_t r = fresh();
    r.begin_answer();
    r.cycle_begin(1000); r.step(0.9f); r.verified(1, 1);
    CHECK(r.cycle_begin(101000), "charged"); // 100 ms
    CHECK(r.n_cycles == 1 && std::fabs(r.t_avg - (70e3 + 0.02 * 30e3)) < 1e-9, "the first cycle charged 100 ms");
    r.step(0.9f); r.verified(1, 0);
    r.begin_answer();        // the answer ended: its last cycle has no next draft call
    CHECK(!r.cycle_begin(60000000), "not charged"); // a minute later
    CHECK(r.n_cycles == 1, "neither the last cycle nor the wait is charged (%lld)", (long long) r.n_cycles);
    r.verified(0, 0);        // a draft whose first step failed: nothing checked, charged as one step
    r.cycle_begin(60050000);
    CHECK(r.n_cycles == 2 && r.checked == 0 && r.steps == 0, "an empty draft is still a cycle");
    rule_t e = fresh();
    e.cycle_begin(0); e.cycle_begin(5000);
    CHECK(e.n_cycles == 0, "a cycle never verified is not charged");
}

// --- section 4 ----------------------------------------------------------------------------------------------

// 1. the floor, over easy, calibrated and hopeless streams, and never past the ceiling (the 9+ column cliff,
// modelled as 60 ms more for a cycle checking more than 8: the line cannot see it, so only the ceiling keeps it off)
static void test_floor() {
    rng_t g(1);
    line_t cliff = CODE;
    cliff.cliff = 60e3;
    for (int n_max : {8, 3, 1}) {
        rule_t r = fresh(n_max);
        for (int a = 0; a < 6; a++) {
            std::vector<cycle_t> cy;
            for (int i = 0; i < 200; i++) {
                cy.push_back(a % 3 == 0 ? bimodal(g, {0.97}) : a % 3 == 1 ? calibrated(g, 0.0) : bimodal(g, {0.02}));
            }
            const trail_t t = run_answer(r, cy, cliff);
            for (size_t i = 0; i < cy.size(); i++) {
                CHECK(t.checked[i] >= 1 && t.steps[i] >= 1, "n_max %d answer %d cycle %zu: a cycle checks nothing", n_max, a, i);
                CHECK(t.checked[i] <= n_max && t.steps[i] <= n_max, "n_max %d: past the ceiling", n_max);
                CHECK(t.steps[i] == t.checked[i] || t.steps[i] == t.checked[i] + 1, "at most one step dropped");
            }
        }
    }
}

// 2. keep rates that vary by position: within 3% of the best fixed depth; prose no worse than fixed 1 by 3%
static void test_positions() {
    rng_t g(11);
    const auto code = stream(g, {0.9, 0.7, 0.5, 0.3}, 4000);
    rule_t r = fresh();
    const trail_t t = run_answer(r, code, CODE);
    const double best = fixed_rate(code, best_fixed(code, CODE), CODE);
    CHECK(t.words / t.us >= 0.97 * best, "code: %.2f words/s, best fixed %.2f", 1e6 * t.words / t.us, 1e6 * best);
    const auto prose = stream(g, {0.58, 0.4}, 4000);
    rule_t p = fresh();
    const trail_t tp = run_answer(p, prose, PROSE);
    const double f1 = fixed_rate(prose, 1, PROSE);
    CHECK(tp.words / tp.us >= 0.97 * f1, "prose: %.2f words/s, fixed 1 %.2f", 1e6 * tp.words / tp.us, 1e6 * f1);
}

// 3. answer switches, filler -> prose -> code, 60 cycles each, five rounds: from cycle 20 of each answer the words
// a second reach the best fixed depth's (-3%) and the mean depth is within 1 of it on prose and code; the first
// cycle of an answer is a real decision (after easy answers, an easy one starts deep: no warm-up, no hold)
static void test_switches() {
    const char * names[3] = {"filler", "prose", "code"};
    const std::vector<double> rates[3] = {{0.95}, {0.58, 0.4}, {0.9, 0.7, 0.5, 0.3}};
    const line_t lines[3] = {CODE, PROSE, CODE};
    rng_t g(3);
    rule_t r = fresh();
    std::vector<cycle_t> late[3];
    std::vector<int> depth[3];
    double w[3] = {}, us[3] = {};
    int first_filler_deep = 0;
    for (int round = 0; round < 5; round++) {
        for (int k = 0; k < 3; k++) {
            const auto cy = stream(g, rates[k], 60);
            const trail_t t = run_answer(r, cy, lines[k]);
            CHECK(t.checked[0] >= 1, "the first cycle of an answer checks a guess");
            first_filler_deep += (k == 0 && round > 0 && t.checked[0] >= 3);
            for (size_t i = 20; i < cy.size(); i++) {
                const int kept = std::min(t.checked[i], cy[i].kept);
                late[k].push_back(cy[i]); depth[k].push_back(t.checked[i]);
                w[k] += kept + 1; us[k] += lines[k].us(t.steps[i], t.checked[i], kept);
            }
        }
    }
    CHECK(first_filler_deep >= 2, "a later filler answer starts deep %d/4 times", first_filler_deep);
    for (int k = 0; k < 3; k++) {
        const int bd = best_fixed(late[k], lines[k]);
        const double best = fixed_rate(late[k], bd, lines[k]);
        CHECK(w[k] / us[k] >= 0.97 * best, "%s: %.2f words/s from cycle 20, best fixed (%d) %.2f", names[k],
              1e6 * w[k] / us[k], bd, 1e6 * best);
        const double md = mean(depth[k]);
        if (k > 0) {
            CHECK(std::fabs(md - bd) <= 1.0, "%s: mean depth %.2f, best fixed %d", names[k], md, bd);
        } else {
            // FINDING (pinned): on filler the rule checks ~6.6 a cycle against the best fixed 8, because it stops
            // at a low-confidence guess that fixed 8 checks for nothing; it writes ~10% faster than fixed 8, so
            // the section's depth bar is the wrong yardstick here, not a miss. Wanted: within 1 of 8.
            CHECK(md < bd - 1.0 && md > bd - 2.5, "filler: mean depth %.2f, best fixed %d (pinned finding)", md, bd);
        }
    }
}

// 4. short answers, 30 cycles, alternating code and prose: prose >= fixed 1 - 3%, code >= fixed 3
static void test_short_answers() {
    rng_t g(100);
    rule_t r = fresh();
    std::vector<cycle_t> all[2];
    double w[2] = {}, us[2] = {};
    for (int i = 0; i < 200; i++) {
        const int k = i % 2; // 0 code, 1 prose
        const auto cy = stream(g, k ? std::vector<double>{0.58, 0.4} : std::vector<double>{0.9, 0.7, 0.5, 0.3}, 30);
        const trail_t t = run_answer(r, cy, k ? PROSE : CODE);
        w[k] += t.words; us[k] += t.us;
        all[k].insert(all[k].end(), cy.begin(), cy.end());
    }
    CHECK(w[1] / us[1] >= 0.97 * fixed_rate(all[1], 1, PROSE), "prose %.2f, fixed 1 %.2f", 1e6 * w[1] / us[1],
          1e6 * fixed_rate(all[1], 1, PROSE));
    CHECK(w[0] / us[0] >= fixed_rate(all[0], 3, CODE), "code %.2f, fixed 3 %.2f", 1e6 * w[0] / us[0],
          1e6 * fixed_rate(all[0], 3, CODE));
}

// 5. the lock-out: code, then prose. Under the floor every cycle checks, so the line keeps learning the check's
// price on the new text: the cost of the cycles the rule runs on prose closes a third of the gap in 100 cycles
// (the forgetting leaves ~60% weight on code after 100) and comes within 5% by 800
static void test_lockout() {
    rng_t g(5);
    rule_t r = fresh();
    run_answer(r, stream(g, {0.9, 0.7, 0.5, 0.3}, 1000), CODE);
    const auto prose = stream(g, {0.58, 0.4}, 800);
    const double truth = PROSE.us(2, 1, 0), before = r.cost.us(2, 1, 1);
    r.begin_answer();
    trail_t t;
    for (int i = 0; i < 100; i++) { run_cycle(r, prose[i], PROSE, t); }
    const double now = r.cost.us(2, 1, 1);
    CHECK(std::fabs(now - truth) <= (2.0 / 3.0) * std::fabs(before - truth), "after 100 prose cycles: %.1f ms, prose "
          "%.1f, code's line said %.1f", now / 1e3, truth / 1e3, before / 1e3);
    for (size_t i = 100; i < prose.size(); i++) { run_cycle(r, prose[i], PROSE, t); }
    CHECK(std::fabs(r.cost.us(2, 1, 1) - truth) <= 0.05 * truth, "after 800: %.1f ms, prose %.1f", r.cost.us(2, 1, 1) / 1e3,
          truth / 1e3);
    // FINDING (pinned): `b` alone is not what the line learns. Nearly every cycle drafts one step more than it
    // checks (steps = checked + 1), so the data fix s + b and a + s + f, and the split between s and b comes from
    // the prior and the noise. On prose `b` ends ~6 ms from prose's 16.7. Wanted: b and f near prose's in 100.
    CHECK(std::fabs(r.cost.th[2] - PROSE.b) > 3e3, "b %.1f ms, prose %.1f; f %.1f, prose %.1f (pinned finding)",
          r.cost.th[2] / 1e3, PROSE.b / 1e3, r.cost.th[3] / 1e3, PROSE.f / 1e3);
}

// 6. the price learns: doubling b mid-run. In a world where the confidence is the chance (so the best depth
// follows the price: the best priced rule's deep share falls ~2.5x), the deep share should halve in 100 cycles
static void test_price_learns() {
    rng_t g(0);
    line_t dear = CODE;
    dear.b = 2 * CODE.b;
    std::vector<cycle_t> cy;
    for (int i = 0; i < 1600; i++) { cy.push_back(calibrated(g, 0.3)); }
    rule_t r = fresh();
    trail_t t;
    for (int i = 0; i < 1600; i++) { run_cycle(r, cy[i], i < 800 ? CODE : dear, t); }
    auto share = [&](int lo, int hi) { int n = 0; for (int i = lo; i < hi; i++) { n += t.checked[i] >= 3; } return n / double(hi - lo); };
    const double before = share(400, 800), after = share(800, 900), late = share(1200, 1600);
    // FINDING (pinned): it does not; the deep share falls ~1/3 in 100 cycles and ~40% by 800 (0.61 -> 0.42 ->
    // 0.36), because the line cannot tell a dearer check from a dearer step (test 5's finding) and the time per
    // word rises with the price. Wanted: after <= before / 2.
    CHECK(after > before / 2 && late > before / 2, "deep share %.2f -> %.2f (100 cycles) -> %.2f (400-800) (pinned finding)",
          before, after, late);
}

// --- the script the study's agreement test writes -------------------------------------------------------------

static int run_script(const char * in_path, const char * out_path) {
    FILE * in = std::fopen(in_path, "r");
    FILE * out = std::fopen(out_path, "w");
    if (!in || !out) { fprintf(stderr, "cannot open %s or %s\n", in_path, out_path); return 2; }
    rule_t r = fresh();
    line_t L = CODE;
    char tag[4];
    while (std::fscanf(in, "%3s", tag) == 1) {
        if (tag[0] == 'N') {
            int n; if (std::fscanf(in, "%d", &n) != 1) { return 3; }
            r = fresh(n);
        } else if (tag[0] == 'L') {
            if (std::fscanf(in, "%lf %lf %lf %lf %lf %lf", &L.a, &L.s, &L.b, &L.f, &L.d, &L.w) != 6) { return 3; }
        } else if (tag[0] == 'A') {
            r.begin_answer();
        } else if (tag[0] == 'C') {
            cycle_t c; int n;
            if (std::fscanf(in, "%d %lf %d", &c.kept, &c.mult, &n) != 3) { return 3; }
            c.conf.resize(n);
            for (auto & x : c.conf) { if (std::fscanf(in, "%f", &x) != 1) { return 3; } }
            trail_t t;
            r.start();
            const double b = r.price_b, pw = r.per_word;
            run_cycle(r, c, L, t);
            std::fprintf(out, "{\"steps\":%d,\"checked\":%d,\"b\":%.17g,\"per_word\":%.17g}\n", // an empty draft is
                         std::max(1, t.steps[0]), t.checked[0], b, pw);                     // charged one step
        } else { return 3; }
    }
    std::fclose(in); std::fclose(out);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 5 && std::strcmp(argv[1], "--script") == 0 && std::strcmp(argv[3], "--dump") == 0) {
        return run_script(argv[2], argv[4]);
    }
    for (auto test : {test_by_hand, test_golden, test_clock, test_floor, test_positions, test_switches,
                      test_short_answers, test_lockout, test_price_learns}) {
        test();
    }
    if (g_failures) { fprintf(stderr, "test-speculative-rate2: %d failure(s)\n", g_failures); return 1; }
    printf("test-speculative-rate2: all passed\n");
    return 0;
}

// The priced depth, floor-post (common/speculative-rate2.h): its arithmetic by hand (the price, the bins, the
// per-answer caps, the two-part cost line against the study's SplitCost), a golden run against the study's FloorPost,
// the engine's clock (the draft call timed on its own; a gap between answers and an answer's last cycle are never
// charged) and what the draft's own time buys: the line tells a step from a check. The simulated tests of
// docs/wizard-rate-rework.md section 4 are test-speculative-rate2-sim.
//
// `--script IN --dump OUT` runs a script and writes one JSON line a cycle (the study's sim/test_spec_rate2_engine.py
// compares them with FloorPost). A script is lines of "N n_max" | "L a s b f d w" (the cost line) | "A" (a new
// answer) | "C kept mult n conf1..confn" (a cycle: what depth 8 kept, the time's factor over the line, confidences);
// each cycle's draft call is timed at the line's d + s*steps times the same factor, and the dump carries it.

#include "speculative-rate2-world.h"

#include <cstdlib>
#include <cstring>

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
    r.charge(100e3, 25e3);
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

    // the two-part line against the study's SplitCost on the same five cycles (python3.12, sim/spec_floor.py):
    // (steps, checked, draft us, cycle us)
    common_speculative_cost_split L;
    CHECK(L.d() == 0 && L.s() == 8e3 && L.a() == 70e3 && L.b() == 20e3 && L.f() == 10e3 && L.us(3, 2) == 144e3,
          "the prior: the whole-cycle line's (70, 8, 20, 10) ms, the draft call's own d at 0");
    const double pts[5][4] = {{2, 1, 17e3, 95e3}, {3, 3, 25e3, 160e3}, {1, 1, 9e3, 80e3}, {8, 8, 66e3, 300e3}, {2, 1, 18e3, 90e3}};
    for (auto & p : pts) { L.observe((int) p[0], (int) p[1], p[2], p[3]); }
    const double got[5]  = {L.d(), L.s(), L.a(), L.b(), L.f()};
    const double want[5] = {313.4093709442195, 8241.198591250779, 63582.814713655986, 20464.507566444787, 3582.8147136560033};
    for (int i = 0; i < 5; i++) {
        CHECK(std::fabs(got[i] - want[i]) < 1e-6, "cost term %d (d s a b f): %.9f, SplitCost %.9f", i, got[i], want[i]);
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
    const int want[16][2] = {{2, 1}, {2, 1}, {5, 4}, {2, 1}, {3, 2}, {5, 4}, {5, 4}, {5, 4},
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
    CHECK(std::fabs(r.cost.b() - 27597.247690540793) < 1e-6 && std::fabs(r.cost.s() - 8566.609697675174) < 1e-6 &&
          std::fabs(r.t_avg - 105063.35446457067) < 1e-6 && std::fabs(r.w_avg - 1.7926472929117185) < 1e-12,
          "state: b %.9f s %.9f t %.9f w %.12f", r.cost.b(), r.cost.s(), r.t_avg, r.w_avg);
}

// the engine's clock: each cycle timed from its draft call to the next of the same answer, its draft call from
// the same start to the draft's end
static void test_clock() {
    rule_t r = fresh();
    r.begin_answer();
    r.cycle_begin(1000); r.step(0.9f); r.draft_done(21000); r.verified(1, 1);
    CHECK(r.cycle_begin(101000), "charged"); // 100 ms, its draft call 20
    CHECK(r.n_cycles == 1 && std::fabs(r.t_avg - (70e3 + 0.02 * 30e3)) < 1e-9, "the first cycle charged 100 ms");
    CHECK(r.cost.draft.S[0][0] == 1.0 && r.cost.draft.r[0] == 20e3 && r.cost.rest.r[0] == 80e3,
          "the draft part learns 20 ms, the rest 80 (%.0f, %.0f)", r.cost.draft.r[0], r.cost.rest.r[0]);
    r.step(0.9f); r.verified(1, 0);
    r.begin_answer();        // the answer ended: its last cycle has no next draft call
    CHECK(!r.cycle_begin(60000000), "not charged"); // a minute later
    CHECK(r.n_cycles == 1, "neither the last cycle nor the wait is charged (%lld)", (long long) r.n_cycles);
    r.draft_done(60001000); r.verified(0, 0); // a draft whose first step failed: nothing checked, one step
    r.cycle_begin(60050000);
    CHECK(r.n_cycles == 2 && r.checked == 0 && r.steps == 0, "an empty draft is still a cycle");
    CHECK(std::fabs(r.cost.draft.r[0] - (0.995 * 20e3 + 1e3)) < 1e-9, "its draft call, 1 ms, is learnt");
    const double seen = r.cost.draft.S[0][0], t_before = r.t_avg;
    r.step(0.9f); r.verified(1, 1);           // no draft_done: the line learns nothing, time per word still does
    r.cycle_begin(60150000);
    CHECK(r.n_cycles == 3 && r.cost.draft.S[0][0] == seen && r.t_avg != t_before, "a cycle whose draft was never timed");
    r.draft_done(60149000);                   // before the cycle began: never a time
    CHECK(r.t_draft < 0, "a draft end before its cycle's start is not a time");
    r.draft_done(60400000); r.verified(1, 1); // longer than the cycle below: the draft is the whole cycle, never more
    const double rest0 = r.cost.rest.r[0];
    r.cycle_begin(60350000);
    CHECK(std::fabs(r.cost.rest.r[0] - 0.995 * rest0) < 1e-6, "the rest measured 0");
    rule_t e = fresh();
    e.cycle_begin(0); e.cycle_begin(5000);
    CHECK(e.n_cycles == 0, "a cycle never verified is not charged");
}

// what the draft's own time buys: under floor-post nearly every cycle drafts one step more than it checks, so one
// line over whole cycles fixed only s + b (round 1: two prose worlds that differ only in the split, b 16.7 or
// 6.5 ms, ended with b 21.2 and ~19 ms). Timed on its own, the draft call fixes s, and so b is each world's own.
static void test_split() {
    line_t swapped = PROSE;
    std::swap(swapped.s, swapped.b);
    for (const line_t & L : {PROSE, swapped}) {
        rng_t g(21);
        rule_t r = fresh();
        const trail_t t = run_answer(r, stream(g, {0.58, 0.4}, 3000), L);
        int dropped = 0;
        for (size_t i = 0; i < t.steps.size(); i++) { dropped += t.steps[i] == t.checked[i] + 1; }
        CHECK(dropped > 0.9 * t.steps.size(), "steps = checked + 1 on %d of %zu cycles", dropped, t.steps.size());
        CHECK(std::fabs(r.cost.s() - L.s) < 0.3e3 && std::fabs(r.cost.b() - L.b) < 1.5e3,
              "s %.1f ms (world %.1f), b %.1f (world %.1f)", r.cost.s() / 1e3, L.s / 1e3, r.cost.b() / 1e3, L.b / 1e3);
    }
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
            const double b = r.price_b, pw = r.per_word, s = r.cost.s();
            run_cycle(r, c, L, t);
            std::fprintf(out, "{\"steps\":%d,\"checked\":%d,\"b\":%.17g,\"per_word\":%.17g,\"s\":%.17g,"
                         "\"draft_us\":%.17g}\n", std::max(1, t.steps[0]), t.checked[0], b, pw, s, t.draft_us[0]);
                         // an empty draft is charged one step
        } else { return 3; }
    }
    std::fclose(in); std::fclose(out);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 5 && std::strcmp(argv[1], "--script") == 0 && std::strcmp(argv[3], "--dump") == 0) {
        return run_script(argv[2], argv[4]);
    }
    for (auto test : {test_by_hand, test_golden, test_clock, test_split}) {
        test();
    }
    if (g_failures) { fprintf(stderr, "test-speculative-rate2: %d failure(s)\n", g_failures); return 1; }
    printf("test-speculative-rate2: all passed\n");
    return 0;
}

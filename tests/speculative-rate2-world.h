#pragma once

// The simulated world the priced depth's tests run floor-post in (tests/test-speculative-rate2.cpp and
// test-speculative-rate2-sim.cpp): cost lines, depth-8 cycles with what each step showed and what the library kept,
// and one cycle run as the engine runs it, in the order the study's replay (`sim/spec_rules.py` replay_rule) runs
// its FloorPost, so the two can be compared cycle by cycle.

#include "speculative-rate2.h"

#include <cstdio>
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
    std::vector<double> draft_us;      // what the rule was told its draft call took
};

// one cycle as the engine runs it (start, a step per confidence while the rule says go, the draft call's end, the
// library's check, the measured time): the cycle is its line times its factor, and the draft call the line's own
// part (d + s*steps) times the same factor, as the replay tells FloorPost
static void run_cycle(rule_t & r, const cycle_t & c, const line_t & L, trail_t & t) {
    r.start();
    for (size_t i = 0; i < std::min<size_t>(r.n_max, c.conf.size()) && r.step(c.conf[i]); i++) {}
    r.verified(r.checked, c.kept);
    const int    steps = std::max(1, r.steps); // a step that failed to decode is charged as one
    const double us    = L.us(steps, r.checked, r.kept);
    const double seen  = us * c.mult;
    const double draft = (L.d + L.s * steps) * (seen / us);
    r.charge(seen, draft);
    t.words += r.kept + 1; t.us += us;
    t.steps.push_back(r.steps); t.checked.push_back(r.checked); t.draft_us.push_back(draft);
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

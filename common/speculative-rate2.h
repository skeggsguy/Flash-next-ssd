#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// The apprentice's priced depth, `floor-post`: LLAMA_SPEC_ADAPTIVE_RATE=2 with --spec-type draft-mtp-adaptive
// ("1" is the measured depth of speculative-rate.h; unset or "0" the hand-tuned speculative-adaptive.h).
//
// Every checking cycle the apprentice drafts one guess and the library checks it, whatever it looks worth: the
// floor. There is no depth 0 and no shadow guess, so the check's price is measured on the current text every
// cycle. Each deeper step is drafted, and its guess is checked only if
//     (the chance guesses 1..i are kept) x (the calibrated chance for step i+1's confidence) x (time per word)
// is more than `b`, what a checked guess costs; otherwise the draft stops there, the step paid for and its guess
// dropped. The price replaces p-min (the startup line says "p-min: priced"); the ceiling is --spec-draft-n-max.
//
//   cost line   cycle = a + s*steps + b*checked + f*[any checked], least squares over this slot's own cycles with
//               forgetting (0.995) and a prior that never fades ((70, 8, 20, 10) ms with the weight of 5 cycles),
//               so a direction the rule never exercises returns to the prior instead of winding up. Under the
//               floor `f` is paid every cycle and enters no decision.
//   chance      10 bins of the step's top confidence, each a kept share with a prior of 4 samples at the bin's
//               midpoint, learnt from the guesses the library checked up to the first thrown out, capped at 50
//               samples (then an exponential average).
//   time/word   exponential averages (alpha 0.02) of the cycle's time and the words it wrote, from (70 ms, 1).
//   per answer  bins are capped at 12 samples (a capped bin keeps its chance); costs and time per word carry
//               over; no warm-up, no hold: the first cycle of an answer is a real decision.
//
// The study's replay holds the same rule (`sim/spec_floor.py` `FloorPost`), and its test
// `sim/test_spec_rate2_engine.py` runs tests/test-speculative-rate2 on a script and compares the choices cycle
// by cycle. State is per slot (one per sequence).

// the fitted line; the arithmetic follows the study's `CostRLS` and `_solve` (sim/spec_rules.py, spec_trace.py)
struct common_speculative_cost_rls {
    static constexpr int N = 4; // a, s, b, f

    double prior[N]     = {70e3, 8e3, 20e3, 10e3}; // us
    double prior_weight = 5.0;
    double forget       = 0.995;

    double S[N][N] = {};
    double r[N]    = {};
    double th[N]   = {70e3, 8e3, 20e3, 10e3};

    double us(double steps, double checked, double any_checked) const {
        return th[0] + th[1] * steps + th[2] * checked + th[3] * any_checked;
    }

    void observe(int steps, int checked, double y) {
        const double x[N] = {1.0, (double) steps, (double) checked, checked > 0 ? 1.0 : 0.0};
        const double g    = forget;
        for (int i = 0; i < N; i++) {
            r[i] = g * r[i] + x[i] * y;
            for (int j = 0; j < N; j++) {
                S[i][j] = g * S[i][j] + x[i] * x[j];
            }
        }
        double m[N][N + 1];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                m[i][j] = S[i][j] + (i == j ? prior_weight : 0.0);
            }
            m[i][N] = r[i] + prior_weight * prior[i];
        }
        solve(m, th); // the prior keeps the system positive definite, so it never fails; if it did, th stays
    }

    // Gauss-Jordan with partial pivoting (the first largest pivot, as Python's max); false if singular
    static bool solve(double a[N][N + 1], double out[N]) {
        for (int col = 0; col < N; col++) {
            int piv = col;
            for (int row = col + 1; row < N; row++) {
                if (std::fabs(a[row][col]) > std::fabs(a[piv][col])) {
                    piv = row;
                }
            }
            for (int j = 0; j <= N; j++) {
                std::swap(a[col][j], a[piv][j]);
            }
            if (std::fabs(a[col][col]) < 1e-12) {
                return false;
            }
            for (int row = 0; row < N; row++) {
                if (row != col) {
                    const double k = a[row][col] / a[col][col];
                    for (int j = 0; j <= N; j++) {
                        a[row][j] = a[row][j] - k * a[col][j];
                    }
                }
            }
        }
        for (int i = 0; i < N; i++) {
            out[i] = a[i][N] / a[i][i];
        }
        return true;
    }
};

struct common_speculative_rate2 {
    static constexpr int K_MAX = 16; // the deepest ceiling (the trace's MAX_STEPS)
    static constexpr int BINS  = 10;

    int    n_max       = 8;
    double bin_prior   = 4.0;  // samples at each bin's midpoint
    double bin_cap     = 50.0; // samples a bin holds (1 / alpha, alpha 0.02)
    double answer_cap  = 12.0; // samples a bin keeps into a new answer
    double word_alpha  = 0.02;
    bool   check_all   = false; // tests only: every step checked to the ceiling (fixed n-max at p-min 0)

    common_speculative_cost_rls cost;
    double hit[BINS]  = {};
    double seen[BINS] = {};
    double t_avg      = 70e3; // cycle time, us
    double w_avg      = 1.0;  // words a cycle

    // the running cycle
    int    steps    = 0;     // draft steps run
    int    checked  = 0;     // guesses sent to be checked
    double run      = 1.0;   // chance every guess so far is kept
    double price_b  = 0.0;   // this cycle's `b` and time per word, fixed at its start
    double per_word = 0.0;
    float  conf[K_MAX] = {};
    int    kept     = 0;     // guesses the library kept (verified)
    bool   pending  = false; // verified and not yet charged

    // timing, as speculative-rate.h: a cycle is timed from its draft call to the next one of the same answer
    int64_t t_start  = -1;
    int64_t n_cycles = 0;     // cycles charged
    double  checked_avg = 1.0; // exponential average of guesses checked (the status line)

    void init(int n_max_) {
        *this = common_speculative_rate2();
        n_max = std::max(1, std::min(n_max_, K_MAX));
        for (int b = 0; b < BINS; b++) {
            hit[b]  = bin_prior * (b + 0.5) / BINS;
            seen[b] = bin_prior;
        }
        start();
    }

    // in double, as the study's `Priced._bin` (a float product rounds 0.7f x 10 up to bin 7; the double is 6.99..)
    static int bin_of(float c) {
        return std::min(BINS - 1, std::max(0, (int) ((double) c * BINS)));
    }

    double chance(float c) const {
        const int b = bin_of(c);
        return hit[b] / seen[b];
    }

    double time_per_word() const {
        return t_avg / w_avg;
    }

    // a new answer: what the text taught fades (bins capped), the machine's costs stay, the wait is not work
    void begin_answer() {
        for (int b = 0; b < BINS; b++) {
            if (seen[b] > answer_cap) {
                hit[b] *= answer_cap / seen[b];
                seen[b] = answer_cap;
            }
        }
        t_start = -1;
        pending = false;
    }

    // a new cycle's decisions start from what is known now
    void start() {
        steps    = 0;
        checked  = 0;
        run      = 1.0;
        kept     = 0;
        pending  = false;
        price_b  = cost.th[2];
        per_word = time_per_word();
    }

    // the draft call: the previous cycle of this answer is charged its whole time, then a new cycle starts;
    // true if a cycle was charged
    bool cycle_begin(int64_t now_us) {
        const bool charged = t_start >= 0 && pending && now_us > t_start;
        if (charged) {
            charge((double) (now_us - t_start));
        }
        start();
        t_start = now_us;
        return charged;
    }

    // a draft step ran and showed its top confidence: check its guess (and draft on), or stop the draft here?
    bool step(float c) {
        if (steps >= n_max) {
            return false; // never past the ceiling
        }
        conf[steps++] = c;
        run *= chance(c);
        if (steps == 1 || check_all || run * per_word > price_b) { // the floor, then the price
            checked++;
            return true;
        }
        return false;
    }

    // the library checked n_checked guesses and kept n_accepted: the bins learn from the guesses up to the first
    // thrown out (each a sample of its confidence's chance), and the cycle waits for its time
    void verified(int n_checked, int n_accepted) {
        checked = std::max(0, std::min(n_checked, steps));
        kept    = std::max(0, std::min(n_accepted, checked));
        for (int i = 0; i < std::min(kept + 1, checked); i++) {
            const int b = bin_of(conf[i]);
            if (seen[b] >= bin_cap) { // forget slowly past the cap
                hit[b]  *= (bin_cap - 1.0) / bin_cap;
                seen[b] *= (bin_cap - 1.0) / bin_cap;
            }
            hit[b]  += i < kept ? 1.0 : 0.0;
            seen[b] += 1.0;
        }
        pending = true;
    }

    // the cycle's measured time: the cost line and time per word learn from it (a step that failed to decode
    // still counts as one step, as the replay does)
    void charge(double us) {
        cost.observe(std::max(1, steps), checked, us);
        t_avg += word_alpha * (us - t_avg);
        w_avg += word_alpha * ((double) (kept + 1) - w_avg);
        checked_avg += 0.01 * ((double) checked - checked_avg);
        n_cycles++;
        pending = false;
    }
};

// tests only (tests/test-speculative-rate2-mtp.cpp), for drafts set up after the call with LLAMA_SPEC_ADAPTIVE_RATE=2:
// every step checked to the ceiling (so its guesses can be held to fixed n-max at p-min 0), and a starting time per
// word (0: the rule's own 70 ms), so a fixture whose guesses are never kept still sees a range of depths
void common_speculative_rate2_check_all_for_tests(bool on);
void common_speculative_rate2_time_per_word_for_tests(double us);

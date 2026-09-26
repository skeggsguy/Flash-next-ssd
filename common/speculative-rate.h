#pragma once

#include <algorithm>
#include <cstdint>

// Throughput-seeking draft depth for MTP speculative decoding: LLAMA_SPEC_ADAPTIVE_RATE=1 with
// --spec-type draft-mtp-adaptive. Off (unset) keeps the hand-tuned controller in
// speculative-adaptive.h.
//
// The hand-tuned controller climbs and drops on fixed thresholds tuned on one model's prose, and it
// restarts from its floor at every answer. This one measures the two things the right depth depends
// on and picks the depth with the most tokens per second, so a different model or machine needs no
// retuning:
//   p[i]  how often the i-th guess of a draft is kept, given the guesses before it were (every draft
//         that reached position i counts, at any depth), weighted towards recent verifies so a switch
//         from code to prose shows within tens of cycles;
//   c[k]  the wall time of a whole cycle (draft, verify, accept) with k guesses verified, timed from
//         one draft call to the next, weighted the same way.
// A cycle at depth k writes E(k) = 1 + sum_{j=1..k} prod_{i=1..j} p[i] tokens on average, so the
// controller runs the k in [0, n_max] with the largest E(k) / c(k). Depth 0 verifies nothing: a
// one-token shadow guess is still drafted and scored against the token actually written, so p[1] keeps
// moving and the controller comes back when the content turns predictable. The first n_max + 1 cycles
// run every depth once, 0 first (a cycle with nothing to verify costs far less than any other, so it
// cannot be extrapolated from them); after that, every probe_every cycles the controller runs one step
// above its choice, one step below, or depth 0, in turn, so the numbers it chooses by stay current. What
// it has learnt carries over from one answer to the next; only the cycle timer restarts, so the wait
// between answers is never counted.
struct common_speculative_rate {
    static constexpr int K_MAX = 16;

    int    n_max       = 8;    // deepest depth allowed (--spec-draft-n-max)
    int    probe_every = 12;   // cycles between probes of a neighbouring depth
    double alpha_p     = 0.05; // weight of the newest observation once enough have been seen
    double alpha_c     = 0.10;

    double p[K_MAX + 1]  = {};  // p[i], i >= 1
    double wp[K_MAX + 1] = {};  // observations behind p[i], capped at 1/alpha_p
    double c[K_MAX + 1]  = {};  // cycle time in us at depth k
    double wc[K_MAX + 1] = {};  // observations behind c[k], capped at 1/alpha_c

    int     k_next   = 1;  // depth the coming cycle drafts at
    int     k_ran    = -1; // depth the running cycle verified (0: shadow only), -1: none running
    int64_t t_start  = -1; // when the running cycle's draft call began
    int64_t n_cycles = 0;  // cycles timed so far

    void init(int n_max_) {
        *this = common_speculative_rate();
        n_max  = std::max(0, std::min(n_max_, K_MAX));
        k_next = 0; // the warm-up runs 0, 1, ..., n_max
    }

    // a new answer: the gap since the last draft call is waiting, not work
    void restart_timer() {
        k_ran   = -1;
        t_start = -1;
    }

    // the start of a cycle: the previous cycle's wall time is charged to the depth it verified
    void cycle_begin(int64_t now_us) {
        if (t_start >= 0 && k_ran >= 0 && now_us > t_start) {
            observe(c[k_ran], wc[k_ran], (double) (now_us - t_start), alpha_c);
            n_cycles++;
            k_next = choose();
        }
        t_start = now_us;
        k_ran   = -1;
    }

    // the drafted guesses (n_draft, 0 for a shadow-only cycle) and how many the target kept
    void cycle_verified(int n_draft, int n_accepted) {
        k_ran = std::max(0, std::min(n_draft, K_MAX));
        for (int i = 1; i <= k_ran; i++) {
            if (i <= n_accepted) {
                observe(p[i], wp[i], 1.0, alpha_p);
            } else {
                observe(p[i], wp[i], 0.0, alpha_p); // the first rejected guess; later ones were never reached
                break;
            }
        }
    }

    // a depth-0 cycle: the one-token shadow guess against the token actually written
    void shadow_scored(bool kept) {
        observe(p[1], wp[1], kept ? 1.0 : 0.0, alpha_p);
    }

    // expected tokens written by one cycle at depth k
    double expected_tokens(int k) const {
        double e = 1.0, run = 1.0;
        for (int i = 1; i <= k; i++) {
            run *= keep(i);
            e   += run;
        }
        return e;
    }

    // p[i] where observed; a position never reached borrows the one before it
    double keep(int i) const {
        for (int j = i; j >= 1; j--) {
            if (wp[j] > 0.0) {
                return p[j];
            }
        }
        return 0.5;
    }

    // measured cycle time, else a straight line through the nearest measured depths
    double cycle_cost(int k) const {
        if (wc[k] > 0.0) {
            return c[k];
        }
        int lo = -1, hi = -1, lo2 = -1;
        for (int j = k - 1; j >= 0; j--) {
            if (wc[j] > 0.0) {
                if (lo < 0) { lo = j; } else { lo2 = j; break; }
            }
        }
        for (int j = k + 1; j <= K_MAX; j++) {
            if (wc[j] > 0.0) { hi = j; break; }
        }
        if (lo >= 0 && hi >= 0) {
            return c[lo] + (c[hi] - c[lo]) * (k - lo) / (double) (hi - lo);
        }
        if (lo >= 0 && lo2 >= 1) {
            return c[lo] + std::max(0.0, (c[lo] - c[lo2]) / (lo - lo2)) * (k - lo);
        }
        if (lo >= 0) {
            return c[lo] * (1.0 + 0.25 * (k - lo)); // one point only: assume each step costs a quarter more
        }
        if (hi >= 0) {
            return c[hi] * (1.0 - 0.1 * (hi - k));
        }
        return 1.0;
    }

    // the depth to run next: the warm-up sweep, then the best measured rate with a probe every
    // probe_every cycles (one step up, one step down, depth 0, in turn)
    int choose() const {
        if (n_cycles <= n_max) {
            return (int) n_cycles;
        }
        int deepest = 0;
        for (int k = 0; k <= n_max; k++) {
            if (wc[k] > 0.0) {
                deepest = k;
            }
        }
        const int limit = std::min(n_max, deepest + 1);
        int    best      = 0;
        double best_rate = -1.0;
        for (int k = 0; k <= limit; k++) {
            const double rate = expected_tokens(k) / std::max(1.0, cycle_cost(k));
            if (rate > best_rate) {
                best      = k;
                best_rate = rate;
            }
        }
        if (probe_every > 0 && n_cycles % probe_every == 0) {
            switch ((n_cycles / probe_every) % 3) {
                case 0:  return std::min(best + 1, n_max);
                case 1:  return std::max(best - 1, 0);
                default: return 0;
            }
        }
        return best;
    }

    // running mean until 1/alpha observations, then an exponential average
    static void observe(double & v, double & w, double x, double alpha) {
        w = std::min(w + 1.0, 1.0 / alpha);
        v += (x - v) / w;
    }
};

#pragma once

// The chain stop (CHAIN-PLAN step 2, PLAN #37 from Slipstream): LLAMA_SPEC_CHAIN_P=<threshold>, off unless set.
// A draft keeps drafting while the product of its words' confidences so far stays at or above the threshold; the
// word that takes the product under it ends the draft and is not drafted (it was walked, as a p-min cut's word is).
// One float, nothing learnt, so the study's replay prices it exactly (`sim/chain_replay.py` `chain_cut`, checked
// against this file by `sim/test_chain_engine.py` through `tests/test-speculative-chain --script`). The product
// runs in double over the float confidences, the threshold is parsed as a double (strtod), and p-min compares in
// float as it always has, so the replay reproduces every cut bit for bit.
//
// `common_speculative_cut` is the one cut every draft takes at its p-min check (draft-simple, EAGLE3, MTP outside
// LLAMA_SPEC_ADAPTIVE_RATE=2, DFlash 1, DSpark and the DFlash 2 walk): p-min first, then the chain, so when both
// fall on one word the stop is named p-min, and otherwise the earlier of the two wins.

#include "speculative-rate-trace.h"

struct common_speculative_chain {
    double threshold = 0.0; // 0: off
    double product   = 1.0;

    explicit common_speculative_chain(double threshold = 0.0) : threshold(threshold) {}

    bool on() const { return threshold > 0.0; }

    void start() { product = 1.0; }

    // false when the word at confidence p takes the product under the threshold
    bool keep(float p) {
        if (!on()) {
            return true;
        }
        product *= (double) p;
        return product >= threshold;
    }
};

// Where a draft stops: p-min (a word under it ends the draft) and the chain stop. `start` opens each draft.
struct common_speculative_cut {
    float                    p_min = 0.0f;
    common_speculative_chain chain;

    explicit common_speculative_cut(float p_min, double chain_p = 0.0) : p_min(p_min), chain(chain_p) {}

    bool needs_conf() const { return p_min > 0.0f || chain.on(); }

    void start() { chain.start(); }

    // false when the word at confidence p ends the draft; `why` says which cut ended it
    bool keep(float p, common_speculative_rate_trace::stop_reason & why) {
        if (p < p_min) {
            why = common_speculative_rate_trace::STOP_PMIN;
            return false;
        }
        if (!chain.keep(p)) {
            why = common_speculative_rate_trace::STOP_CHAIN;
            return false;
        }
        return true;
    }
};

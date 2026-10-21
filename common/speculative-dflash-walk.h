#pragma once

// The DFlash 2 selector walk (CHAIN-PLAN step 1): one pure function, so the cut can be tested on made-up
// lattices (no test model has a selector), and step 2's chain stop rides in the one cut it takes.
//
// One decode of the draft gives a lattice row for each drafted position. A row holds `top_k` candidate
// words (token ids stored as floats), then a top_k x top_k table of scores: row `j` of the table scores the
// candidates when the previous position picked its candidate j (the first position follows candidate 0).
// The walk picks the best-scoring candidate at each position, following the picks, and the draft keeps
// them until the cut drops one. A word's confidence is the softmax of its scores at the pick,
// 1 / sum(exp(s_k - s_max)), worked out only when something reads it: the cut (p-min > 0 or the chain
// stop, common/speculative-chain.h) or the trace. With the trace on, the walk goes on past the stop into
// `tail` and drafts nothing more, so the library's words never change with the trace on.

#include "llama.h"
#include "speculative-chain.h"
#include "speculative-rate-trace.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

struct common_dflash2_lattice {
    const float * rows   = nullptr; // the row of the first drafted position (block position 1)
    size_t        stride = 0;       // floats from one row to the next (the draft's n_embd)
    int32_t       n_pos  = 0;       // drafted positions (the block's tokens less the anchor)
    int32_t       top_k  = 0;       // candidates a position (the selector's top-k)
};

// What the walk saw, for the trace: each walked word's confidence up to and including the one the cut
// dropped (`conf`), the confidences past that (`tail`), and why it stopped.
struct common_dflash2_walk_seen {
    std::vector<float> conf;
    std::vector<float> tail;
    common_speculative_rate_trace::stop_reason stop = common_speculative_rate_trace::STOP_CAP;
};

// Appends the draft to `words` (which the caller hands in empty); a draft shorter than n_min is cleared.
// `seen` is nullptr when the trace is off. Returns why the draft stopped.
inline common_speculative_rate_trace::stop_reason common_dflash2_walk(const common_dflash2_lattice & lat,
        common_speculative_cut cut, int32_t n_min, std::vector<llama_token> & words,
        common_dflash2_walk_seen * seen) {
    const bool want_conf = seen != nullptr || cut.needs_conf();
    common_speculative_rate_trace::stop_reason stop = common_speculative_rate_trace::STOP_CAP;
    bool stopped = false;
    cut.start();
    int32_t predecessor = 0;
    for (int32_t i = 0; i < lat.n_pos; ++i) {
        const float * row    = lat.rows + (size_t) i * lat.stride;
        const float * scores = row + lat.top_k + (size_t) predecessor * lat.top_k;

        predecessor = (int32_t) std::distance(scores, std::max_element(scores, scores + lat.top_k));
        if (!want_conf) {
            words.push_back((llama_token) row[predecessor]);
            continue;
        }
        // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max)); float, as the walk always ran
        float sum = 0.0f;
        for (int32_t k = 0; k < lat.top_k; ++k) {
            sum += std::exp(scores[k] - scores[predecessor]);
        }
        const float p = 1.0f / sum;
        if (stopped) {
            seen->tail.push_back(p); // the trace only: past the stop, nothing is drafted
            continue;
        }
        if (seen != nullptr) {
            seen->conf.push_back(p);
        }
        if (!cut.keep(p, stop)) {
            stopped = true;
            if (seen == nullptr) {
                break;
            }
            continue;
        }
        words.push_back((llama_token) row[predecessor]);
    }
    if (words.size() < (size_t) n_min) {
        words.clear();
    }
    if (seen != nullptr) {
        seen->stop = stop;
    }
    return stop;
}

#pragma once

// The lenient check (the gamma dial): LLAMA_SPEC_LENIENT=<gamma>. FAIR-CHECK-PLAN L3 (#41, Tom's challenge "match on
// multiple instead of one word"). Off unless set; the daily library never sets it.
//
// Today the check keeps a drafted word w* only when the library's own draw lands on it (chance p*, the library's
// chance of w* after its whole chain). The dial keeps it with chance a = min(1, gamma * p*) instead, and on a miss
// writes the library's draw from p with w* taken out. It is built so that gamma 1 writes today's words byte for byte:
//   - the library draws x from p exactly as today (its own sampler, its own random stream, untouched);
//   - x == w*: kept, as today;
//   - x != w* and a > p*: w* is kept anyway with chance (a - p*) / (1 - p*), rolled on a separate random stream
//     seeded from the request's seed (`seed`), so P(keep) = a and a miss writes x, which follows p without w*;
//   - otherwise (gamma 1, p* = 0, p* = 1) nothing more is rolled and x is written, as today.
// The sampler is then told the word actually written (common_sampler_accept), w* when the dial kept it.
//
// Only where that is exact to do (otherwise today's check): temperature > 0, no mirostat or adaptive-p (the draw
// is the chain's `dist` over p), backend sampling off (`request`); and at each position, a draw not redone grammar
// first, a grammar that refuses none of the chain's candidates (a lazy grammar waiting for its trigger refuses
// none), and the reasoning budget not forcing (`keep`). At temperature 0 the chain leaves p* at 0 or 1, so nothing
// changes there either.
//
// Values: unset, "", "0" or "off": off. "1": on at gamma 1 (rolls nothing; a word test). A number > 1: that gamma.
// "inf": keep w* whenever the chain left it any chance (p* > 0; typical acceptance). Anything else (below 1, not a
// number) refuses the server's start: a typo must never run a measurement silently off.
// Priced in the study: results/S41-lenient-price/ (sim/lenient_price.py).

#include "llama.h"

#include <optional>
#include <random>
#include <string>
#include <vector>

struct common_sampler;
struct common_params_sampling;

struct common_spec_lenient {
    bool   on    = false;
    bool   inf   = false;
    double gamma = 1.0;

    // "gamma 1.3", "gamma inf" or "off"
    std::string describe() const;
};

// parses a value of LLAMA_SPEC_LENIENT (nullptr: unset); throws std::invalid_argument on a bad value
common_spec_lenient common_spec_lenient_parse(const char * value);

// the switch from the environment, parsed (throws as `parse`)
common_spec_lenient common_spec_lenient_from_env();

// the server's load line for it ("" when off)
std::string common_spec_lenient_load_line(const common_spec_lenient & lp);

// whether the dial runs for a request with these sampling settings
bool common_spec_lenient_request(const common_spec_lenient & lp, const common_params_sampling & sp, bool backend_sampling);

// seeds a request's own stream: from its seed, mixed so it never repeats the sampler's `dist` stream
void common_spec_lenient_seed(std::mt19937 & rng, uint32_t seed);

// after a miss (the library drew another word), the chance the dial keeps a drafted word the chain gave p_star:
// (a - p*) / (1 - p*) with a = min(1, gamma p*) (1 when inf and p* > 0); 0 when a <= p*, p* <= 0 or p* >= 1
double common_spec_lenient_extra(const common_spec_lenient & lp, double p_star);

// one checked position, right after the library's draw `drawn` (the sampler's candidates still those of that draw)
// and before its accept: true when the drafted word is written instead. `r0` is the sampler's resample count before
// the draw (common_sampler_get_draw_state). Rolls `rng` only when the extra chance is above 0 and the position allows
// the dial.
bool common_spec_lenient_keep(const common_spec_lenient & lp, common_sampler * smpl, llama_token drafted,
        llama_token drawn, uint64_t r0, std::mt19937 & rng);

// a check that restores and replays (the server's checkpoint route): the replay must roll the same chances as the check
// it replaces (so `rng` goes back to `before`), but the check's own rolls were seen, one of them by the miss that ended it,
// so after the replay the stream moves on to where the check left it (`after`), never back over rolls already used
// (a later position would otherwise reuse a roll known to be above its chance, and under-keep)
void common_spec_lenient_restore(std::mt19937 & rng, const std::mt19937 & before, std::optional<std::mt19937> & after);
void common_spec_lenient_replayed(std::mt19937 & rng, std::optional<std::mt19937> & after);

// a request's dial, as the check loops take it (the p/q record's accept_n too)
struct common_spec_lenient_draw {
    const common_spec_lenient * lp;
    std::mt19937 *              rng;
};

// the library's accept loop (common_sampler_sample_and_accept_n, grammar_first false) with the dial; `kept_by_dial`
// (optional) gets the indices of the drafted words the dial wrote over the library's draw
std::vector<llama_token> common_spec_lenient_accept_n(const common_spec_lenient_draw & len, common_sampler * smpl,
        llama_context * ctx, const std::vector<int32_t> & idxs, const std::vector<llama_token> & draft,
        std::vector<int> * kept_by_dial = nullptr);

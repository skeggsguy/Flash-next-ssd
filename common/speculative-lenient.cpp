#include "speculative-lenient.h"

#include "common.h"
#include "reasoning-budget.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static const char * const LENIENT_ENV = "LLAMA_SPEC_LENIENT";

std::string common_spec_lenient::describe() const {
    if (!on) {
        return "off";
    }
    return inf ? std::string("gamma inf") : string_format("gamma %g", gamma);
}

common_spec_lenient common_spec_lenient_parse(const char * value) {
    common_spec_lenient lp;
    if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0) {
        return lp;
    }
    if (std::strcmp(value, "inf") == 0) {
        lp.on  = true;
        lp.inf = true;
        return lp;
    }
    char * end = nullptr;
    const double g = std::strtod(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(g) || g < 1.0) {
        throw std::invalid_argument(string_format("%s=%s is not a dial: a number >= 1, \"inf\", or unset / \"0\" "
                "for off", LENIENT_ENV, value));
    }
    lp.on    = true;
    lp.gamma = g;
    return lp;
}

common_spec_lenient common_spec_lenient_from_env() {
    return common_spec_lenient_parse(std::getenv(LENIENT_ENV));
}

std::string common_spec_lenient_load_line(const common_spec_lenient & lp) {
    if (!lp.on) {
        return "";
    }
    return string_format("lenient check: %s, a drafted word kept with chance %s at temperature > 0 (%s=%s; not the "
            "library's own odds)", lp.describe().c_str(), lp.inf ? "1 whenever the library gives it any" :
            string_format("min(1, %g p)", lp.gamma).c_str(), LENIENT_ENV, std::getenv(LENIENT_ENV));
}

bool common_spec_lenient_request(const common_spec_lenient & lp, const common_params_sampling & sp, bool backend_sampling) {
    if (!lp.on || backend_sampling || !(sp.temp > 0.0f) || sp.mirostat != 0) {
        return false;
    }
    // adaptive-p picks the word itself (in place of `dist`), so the chain's p is not what was drawn from
    return std::find(sp.samplers.begin(), sp.samplers.end(), COMMON_SAMPLER_TYPE_ADAPTIVE_P) == sp.samplers.end();
}

void common_spec_lenient_seed(std::mt19937 & rng, uint32_t seed) {
    std::seed_seq seq = { seed, 0x6c656e69u /* "leni" */ };
    rng.seed(seq);
}

void common_spec_lenient_restore(std::mt19937 & rng, const std::mt19937 & before, std::optional<std::mt19937> & after) {
    if (!after) {
        after = rng; // the first restore only: a replay that restored again must still end where the check ended
    }
    rng = before;
}

void common_spec_lenient_replayed(std::mt19937 & rng, std::optional<std::mt19937> & after) {
    if (after) {
        rng = *after;
        after.reset();
    }
}

double common_spec_lenient_extra(const common_spec_lenient & lp, double p_star) {
    if (!lp.on || !(p_star > 0.0) || p_star >= 1.0) {
        return 0.0;
    }
    const double a = lp.inf ? 1.0 : std::min(1.0, lp.gamma * p_star);
    if (a <= p_star) {
        return 0.0;
    }
    return std::min(1.0, (a - p_star) / (1.0 - p_star));
}

bool common_spec_lenient_keep(const common_spec_lenient & lp, common_sampler * smpl, llama_token drafted,
        llama_token drawn, uint64_t r0, std::mt19937 & rng) {
    if (!lp.on || drawn == drafted || smpl == nullptr) {
        return false;
    }
    const common_sampler_draw_state st = common_sampler_get_draw_state(smpl);
    if (st.n_resample != r0) {
        return false; // the draw was redone grammar first: not a draw from the chain's p
    }
    if (st.budget == REASONING_BUDGET_FORCING || st.budget == REASONING_BUDGET_WAITING_UTF8) {
        return false; // the reasoning budget is closing the scratch work
    }
    const llama_token_data_array * cur = common_sampler_get_candidates(smpl, false); // reads only, no sort
    if (cur == nullptr || cur->data == nullptr || cur->selected < 0 || (size_t) cur->selected >= cur->size) {
        return false; // a backend-sampled word leaves no chain to read
    }
    double p_star = 0.0;
    for (size_t i = 0; i < cur->size; i++) {
        if (cur->data[i].id == drafted) {
            p_star = cur->data[i].p;
            break;
        }
    }
    const double extra = common_spec_lenient_extra(lp, p_star);
    if (extra <= 0.0) {
        return false; // gamma 1, or a word the chain left out: nothing rolled
    }
    if (st.grammar) {
        // a grammar that would refuse any word the draw could have taken shapes the draw (a refused draw is redone):
        // the dial needs the plain p. A lazy grammar awaiting its trigger refuses none.
        std::vector<llama_token> ids;
        ids.reserve(cur->size);
        for (size_t i = 0; i < cur->size; i++) {
            if (cur->data[i].p > 0.0f) {
                ids.push_back(cur->data[i].id);
            }
        }
        if (common_sampler_grammar_refuses(smpl, ids.data(), ids.size()) > 0) {
            return false;
        }
    }
    std::uniform_real_distribution<double> u(0.0, 1.0);
    return u(rng) < extra;
}

std::vector<llama_token> common_spec_lenient_accept_n(const common_spec_lenient_draw & len, common_sampler * smpl,
        llama_context * ctx, const std::vector<int32_t> & idxs, const std::vector<llama_token> & draft,
        std::vector<int> * kept_by_dial) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // common_sampler_sample_and_accept_n, call for call (grammar_first false); the dial only reads in between
    size_t i = 0;
    for (; i < draft.size(); i++) {
        const uint64_t r0 = common_sampler_get_draw_state(smpl).n_resample;
        const llama_token drawn = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool dial = common_spec_lenient_keep(*len.lp, smpl, draft[i], drawn, r0, *len.rng);
        const llama_token id = dial ? draft[i] : drawn;
        if (dial && kept_by_dial != nullptr) {
            kept_by_dial->push_back((int) i);
        }

        common_sampler_accept(smpl, id, true); // the word written, so the sampler's history follows the text

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);

        common_sampler_accept(smpl, id, true);

        result.push_back(id);
    }

    return result;
}

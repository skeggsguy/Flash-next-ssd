// LLAMA_SPEC_ADAPTIVE_RATE=2 (the priced depth, floor-post: common/speculative-rate2.h) on the MTP fixture
// (qwen4exp-moe-e64mtp.gguf), driven as examples/speculative-simple drives speculation, the library greedy:
//   all     with the price forced to "check every step" (the tests-only switch), draft-mtp-adaptive under
//           RATE=2 drafts, checks and writes byte-identically to fixed draft-mtp at n-max 8, p-min 0
//   priced  the real rule, from its own start and from a tests-only 1000 s a word (so a fixture that keeps no
//           guess still drafts deep): every cycle checks at least one guess; each cycle's guesses are the first
//           ones fixed 8 drafts from the same place (a second fixed-8 run, its drafts cut to the rule's lengths,
//           walks the same path), so a step the rule paid for and dropped leaves no trace on later guesses; and
//           the text written is the second run's
// The rule's own arithmetic and the section 4 tests are test-speculative-rate2 (no model).

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "speculative-rate2.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

std::string g_path;
int         g_ngl      = 99;
int         g_failures = 0;

constexpr int N_MAX     = 8;
constexpr int N_PREDICT = 160;

void check(bool ok, const std::string & what) {
    printf("  %-90s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

struct cycle_rec {
    llama_tokens full;  // what the drafter handed back
    llama_tokens draft; // what the library checked (cut, in the cut run)
    int          accepted = 0;
};

struct run_out {
    bool                   ok = false;
    std::vector<cycle_rec> cycles;
    llama_tokens           text;
};

// one greedy generation from `prompt`; `cut` (optional) shortens each cycle's draft to that cycle's length
run_out run(common_speculative_type type, const char * rate, bool check_all, const llama_tokens & prompt,
        const std::vector<size_t> * cut, double per_word = 0.0) {
    run_out out;
    if (rate) {
        setenv("LLAMA_SPEC_ADAPTIVE_RATE", rate, 1);
    } else {
        unsetenv("LLAMA_SPEC_ADAPTIVE_RATE");
    }
    common_speculative_rate2_check_all_for_tests(check_all);
    common_speculative_rate2_time_per_word_for_tests(per_word);

    common_params params;
    params.model.path   = g_path;
    params.n_gpu_layers = g_ngl;
    params.fit_params   = false;
    postprocess_cpu_params(params.cpuparams, nullptr); // what the argument parser does
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    params.warmup       = false;
    params.n_ctx        = 512;
    params.n_batch      = 512;
    params.n_ubatch     = 64;
    params.sampling.temp = 0.0f; // greedy: the text cannot depend on the drafts
    params.speculative.types       = { type };
    params.speculative.draft.n_max = N_MAX;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0.0f;

    const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel,
            common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (model == nullptr || ctx == nullptr) {
        return out;
    }
    common_params params_dft = common_base_params_to_speculative(params);
    auto spec_init = common_speculative_init_from_params(params_dft, model, ctx);
    params.speculative.draft.ctx_tgt = ctx;
    params.speculative.draft.ctx_dft = spec_init->context();
    llama_context * ctx_dft = params.speculative.draft.ctx_dft;
    if (ctx_dft == nullptr || common_context_can_seq_rm(ctx) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        fprintf(stderr, "no MTP context, or the library needs checkpoints (this test does not drive them)\n");
        return out;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    common_sampler_ptr smpl(common_sampler_init(model, params.sampling));
    common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (spec == nullptr) {
        return out;
    }

    const llama_seq_id seq = 0;
    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);
    for (size_t i = 0; i + 1 < prompt.size(); i++) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { seq }, false);
    }
    bool ok = llama_decode(ctx, batch) == 0 && common_speculative_process(spec, batch);

    llama_token  id_last = prompt.back();
    llama_tokens prompt_tgt(prompt.begin(), prompt.end() - 1);
    int          n_past  = (int) prompt.size() - 1;
    common_speculative_begin(spec, seq, prompt_tgt);

    llama_tokens draft;
    while (ok && (int) out.text.size() < N_PREDICT) {
        const int n_draft_max = std::max(0, std::min((int) llama_n_ctx(ctx) - n_past - 2,
                N_PREDICT - (int) out.text.size() - 1));
        common_speculative_get_draft_params(spec, seq) = { true, n_draft_max, n_past, id_last, &prompt_tgt, &draft };
        common_speculative_draft(spec);
        cycle_rec rec;
        rec.full = draft;
        if (cut != nullptr && out.cycles.size() < cut->size() && draft.size() > (*cut)[out.cycles.size()]) {
            draft.resize((*cut)[out.cycles.size()]);
        }
        rec.draft = draft;
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, n_past, -1); // the drafter's guesses go, as in speculative-simple

        common_batch_clear(batch);
        common_batch_add(batch, id_last, n_past++, { seq }, true);
        for (size_t i = 0; i < draft.size(); i++) {
            common_batch_add(batch, draft[i], n_past + (llama_pos) i, { seq }, true);
        }
        ok = llama_decode(ctx, batch) == 0 && common_speculative_process(spec, batch);
        if (!ok) {
            break;
        }
        const auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx, draft);
        common_speculative_accept(spec, seq, (uint16_t) (ids.size() - 1));
        rec.accepted = (int) ids.size() - 1;
        out.cycles.push_back(rec);
        n_past += (int) ids.size() - 1;

        bool eog = false;
        for (llama_token id : ids) {
            prompt_tgt.push_back(id_last);
            id_last = id;
            out.text.push_back(id);
            if ((eog = llama_vocab_is_eog(vocab, id))) {
                break;
            }
        }
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(ctx), seq, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, n_past, -1);
        if (eog) {
            break;
        }
    }
    out.ok = ok;
    llama_batch_free(batch);
    common_speculative_free(spec);
    common_speculative_rate2_check_all_for_tests(false);
    common_speculative_rate2_time_per_word_for_tests(0.0);
    unsetenv("LLAMA_SPEC_ADAPTIVE_RATE");
    return out;
}

bool same_cycles(const run_out & a, const run_out & b) {
    if (a.cycles.size() != b.cycles.size()) {
        return false;
    }
    for (size_t i = 0; i < a.cycles.size(); i++) {
        if (a.cycles[i].draft != b.cycles[i].draft || a.cycles[i].accepted != b.cycles[i].accepted) {
            return false;
        }
    }
    return true;
}

void test_prompt(const llama_tokens & prompt) {
    printf("prompt of %zu tokens, ngl %d\n", prompt.size(), g_ngl);
    const run_out fixed = run(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, nullptr, false, prompt, nullptr);
    const run_out all   = run(COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, "2", true, prompt, nullptr);
    check(fixed.ok && all.ok && fixed.cycles.size() > 10, "fixed 8 and RATE=2 checking every step both ran (" +
            std::to_string(fixed.cycles.size()) + " cycles)");
    size_t full = 0;
    for (const auto & c : fixed.cycles) { full += c.draft.size() == (size_t) N_MAX; }
    check(full > 0, "fixed 8 drafted 8 on " + std::to_string(full) + " cycles");
    check(same_cycles(fixed, all) && fixed.text == all.text, "all: every guess, keep and word byte-identical to fixed 8 at p-min 0");

    // the rule from its own start, then from a time per word of 1000 s: the fixture keeps no guess, so from its
    // own start the price stops every draft at the dropped second step; from 1000 s it drafts deep and comes down
    // as its confidence bins learn that nothing is kept
    for (double per_word : {0.0, 1e9}) {
        const std::string name = per_word > 0 ? "priced from 1000 s a word: " : "priced: ";
        const run_out priced = run(COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, "2", false, prompt, nullptr, per_word);
        std::vector<size_t> lengths;
        bool floor = priced.ok;
        int  shorter = 0, deeper = 0;
        for (const auto & c : priced.cycles) {
            lengths.push_back(c.draft.size());
            floor = floor && !c.draft.empty() && c.draft == c.full;
            deeper  += c.draft.size() > 1;
        }
        check(floor && priced.cycles.size() > 10, name + "every cycle checks at least one guess, and all it hands back");
        const run_out cut = run(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, nullptr, false, prompt, &lengths);
        bool prefix = cut.ok && cut.cycles.size() == priced.cycles.size();
        for (size_t i = 0; prefix && i < priced.cycles.size(); i++) {
            const auto & p = priced.cycles[i].draft;
            const auto & f = cut.cycles[i].full;
            prefix = p.size() <= f.size() && std::equal(p.begin(), p.end(), f.begin()) && cut.cycles[i].draft == p &&
                     cut.cycles[i].accepted == priced.cycles[i].accepted;
            shorter += p.size() < f.size(); // the price stopped it where fixed 8 drafts on
        }
        check(prefix, name + "each cycle's guesses are fixed 8's first ones from the same place (" +
                std::to_string(shorter) + " stopped before fixed 8 did, " + std::to_string(deeper) + " past 1)");
        check(cut.text == priced.text, name + "the same text as fixed 8 cut to its lengths");
        check(shorter > 0 && (per_word == 0.0 || deeper > 0), name + (per_word > 0 ? "the price stopped drafts that "
                "fixed 8 drafted on, some past 1 (not vacuous)" : "the price stopped drafts that fixed 8 drafted on (not vacuous)"));
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) {
            g_path = argv[++i];
        } else if (a == "-ngl" && i + 1 < argc) {
            g_ngl = std::atoi(argv[++i]);
        }
    }
    if (g_path.empty()) {
        fprintf(stderr, "usage: %s -m qwen4exp-moe-e64mtp.gguf [-ngl N]\n", argv[0]);
        return 1;
    }
    common_init();
    llama_backend_init();

    std::mt19937 rng(42);
    for (int n : {37, 120}) {
        llama_tokens prompt;
        for (int i = 0; i < n; i++) {
            prompt.push_back((llama_token) (1 + rng() % 100)); // the fixture has 128 words
        }
        test_prompt(prompt);
    }

    llama_backend_free();
    printf("%s\n", g_failures ? "test-speculative-rate2-mtp: FAILED" : "test-speculative-rate2-mtp: OK");
    return g_failures ? 1 : 0;
}

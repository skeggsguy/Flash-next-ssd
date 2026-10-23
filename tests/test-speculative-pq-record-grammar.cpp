// The p/q record's view of a grammar (common_sampler_get_draw_state, common_sampler_grammar_refuses), on a real
// vocabulary: the MTP fixture has none, so its record test cannot run a grammar. Vocab only, no logits, so no draw:
//   - no grammar: no flag, nothing refused;
//   - a user grammar `root ::= "ab"`: a grammar applies (not lazy); of {a, b, c} the first step refuses b and c, asking
//     twice answers the same (the call reads the grammar's stacks, it never moves them), and after the sampler
//     accepts a, it refuses a and c and lets b through;
//   - a lazy grammar awaiting its trigger: flagged lazy, and it refuses nothing yet;
//   - the draw state starts at no resamples and no reasoning budget (-1).
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <cstdio>
#include <string>

static int g_failures = 0;

static void check(bool ok, const std::string & what) {
    printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s ggml-vocab-gpt-2.gguf\n", argv[0]);
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token a = common_tokenize(vocab, "a", false, false)[0];
    const llama_token b = common_tokenize(vocab, "b", false, false)[0];
    const llama_token c = common_tokenize(vocab, "c", false, false)[0];
    const llama_token abc[3] = { a, b, c };

    {
        common_params_sampling sp;
        common_sampler_ptr smpl(common_sampler_init(model, sp));
        const auto st = common_sampler_get_draw_state(smpl.get());
        check(!st.grammar && !st.lazy && st.budget == -1 && st.n_resample == 0, "no grammar: no flag, no budget, no resamples");
        check(common_sampler_grammar_refuses(smpl.get(), abc, 3) == 0, "no grammar: nothing refused");
    }
    {
        common_params_sampling sp;
        sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, "root ::= \"ab\"");
        common_sampler_ptr smpl(common_sampler_init(model, sp));
        const auto st = common_sampler_get_draw_state(smpl.get());
        check(st.grammar && !st.lazy, "a user grammar applies and is not lazy");
        check(common_sampler_grammar_refuses(smpl.get(), abc, 3) == 2, "at its start it refuses b and c, not a");
        check(common_sampler_grammar_refuses(smpl.get(), abc, 3) == 2, "asking again answers the same (the stacks do not move)");
        common_sampler_accept(smpl.get(), a, true);
        check(common_sampler_grammar_refuses(smpl.get(), abc, 3) == 2, "after a it refuses a and c, not b");
        const llama_token only_b = b;
        check(common_sampler_grammar_refuses(smpl.get(), &only_b, 1) == 0, "b alone is let through");
        check(common_sampler_grammar_refuses(smpl.get(), abc, 0) == 0, "an empty list refuses nothing");
    }
    {
        common_params_sampling sp;
        sp.grammar      = common_grammar(COMMON_GRAMMAR_TYPE_USER, "root ::= \"ab\"");
        sp.grammar_lazy = true;
        sp.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "zzz", LLAMA_TOKEN_NULL });
        common_sampler_ptr smpl(common_sampler_init(model, sp));
        const auto st = common_sampler_get_draw_state(smpl.get());
        check(st.grammar && st.lazy, "a lazy grammar is flagged grammar and lazy");
        check(common_sampler_grammar_refuses(smpl.get(), abc, 3) == 0, "awaiting its trigger it refuses nothing");
    }
    llama_model_free(model);
    llama_backend_free();
    printf("%s\n", g_failures ? "test-speculative-pq-record-grammar: FAILED" : "test-speculative-pq-record-grammar: OK");
    return g_failures ? 1 : 0;
}

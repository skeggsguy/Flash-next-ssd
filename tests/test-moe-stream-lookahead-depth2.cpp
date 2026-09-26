// LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2 (src/llama-moe-stream-lookahead.cpp) on the four-floor 64-book fixture:
// floor L also fetches floor L+2's most likely books, predicted by L+2's router on L's input. A fetch only
// changes when a book arrives, never which book a token reads, so the check is byte for byte: a short
// prompt, then 40 single written tokens and 20 checks of 4 tokens, on two fresh streamed models (the
// switch is read at load), give identical logits, the same as no streaming; and with the switch on the
// fetches two floors ahead happen and some are read by their floor ("used"), or it would pass by doing
// nothing. The desk is 40 slots for 64 books a floor, so tokens miss often.

#include "test-moe-room-model.h"

namespace {

const char * g_depth2_env = "LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2";

bool run_writing(llama_model * model, std::vector<float> & logits) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 1024;
    cp.n_batch    = 512;
    cp.n_ubatch   = 512;
    cp.op_offload = false;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        return false;
    }
    std::mt19937 rng(7);
    const std::vector<llama_token> toks = random_tokens(rng, 20 + 40 + 20*4, n_vocab);
    llama_batch batch = llama_batch_init(512, 0, 1);
    bool ok = true;
    size_t pos = 0;
    for (int b = 0; ok && b <= 60; b++) {
        const size_t n = b == 0 ? 20 : (b <= 40 ? 1 : 4); // the prompt, single tokens, then checks
        common_batch_clear(batch);
        for (size_t i = 0; i < n; i++, pos++) {
            common_batch_add(batch, toks[pos], (llama_pos) pos, { 0 }, true);
        }
        ok = llama_decode(ctx, batch) == 0;
        if (ok) {
            const float * l = llama_get_logits(ctx);
            logits.insert(logits.end(), l, l + n*n_vocab);
        }
    }
    llama_batch_free(batch);
    llama_free(ctx);
    return ok;
}

struct la2_counts {
    int64_t fetched = 0;
    int64_t used    = 0;
};

la2_counts counts(llama_model * model) {
    llama_moe_stream * ms = model->moe_stream();
    std::unique_lock<std::mutex> lk(ms->mtx);
    return { ms->stats.n_la2_issued, ms->stats.n_la2_used };
}

} // namespace

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD"); // the lookahead at its default width, as in every rung
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_ALL");

    std::vector<float> ref;
    check(run_writing(ref_model, ref), "no streaming: the tokens ran");
    llama_model_free(ref_model);

    config c; // a 40-slot desk with the room
    std::vector<float> off, on;
    la2_counts c_off, c_on;

    unsetenv(g_depth2_env);
    llama_model * m_off = load(c);
    check(m_off != nullptr, "streamed model (one floor ahead) loaded");
    if (m_off) {
        check(n_streamed_floors(m_off) >= 3, "the fixture streams at least three floors");
        check(run_writing(m_off, off), "one floor ahead: the tokens ran");
        c_off = counts(m_off);
        llama_model_free(m_off);
    }

    setenv(g_depth2_env, "4", 1);
    llama_model * m_on = load(c);
    check(m_on != nullptr, "streamed model (two floors ahead) loaded");
    if (m_on) {
        check(run_writing(m_on, on), "two floors ahead: the tokens ran");
        c_on = counts(m_on);
        llama_model_free(m_on);
    }
    unsetenv(g_depth2_env);

    check(nonzero(ref), "the reference logits are not all zero");
    check(same(off, ref), "one floor ahead: the same logits as no streaming");
    check(same(on, ref), "two floors ahead: the same logits as no streaming");
    check(c_off.fetched == 0 && c_off.used == 0, "switch off: nothing fetched two floors ahead");
    check(c_on.fetched > 0, "switch on: books fetched two floors ahead");
    check(c_on.used > 0 && c_on.used <= c_on.fetched, "switch on: some were read by their floor (" +
            std::to_string(c_on.used) + " of " + std::to_string(c_on.fetched) + ")");
    printf("two floors ahead: fetched %lld, used %lld\n", (long long) c_on.fetched, (long long) c_on.used);
    return finish();
}

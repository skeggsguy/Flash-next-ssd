// LLAMA_MOE_STREAM_LOOKAHEAD_ALL (src/llama-moe-stream-lookahead.cpp) on the 64-book fixtures: a small
// batch, the apprentice's check of a few guessed words, prefetches the next floor's predicted books for
// every token instead of only the last. A prefetch only changes when a book arrives, never which book a
// token reads, so the check is byte for byte: 40 batches of 4 tokens after a short prompt, written on
// two fresh streamed models (the switch is read at load), give identical logits for every token, the
// same as no streaming, and the switched-on model issued more prefetches, or it would pass by doing
// nothing. The desk is 40 slots for 64 books a floor, so a 4-token check misses often.

#include "test-moe-room-model.h"

namespace {

const char * g_all_env = "LLAMA_MOE_STREAM_LOOKAHEAD_ALL";

// a 20-token prompt (under the room's threshold, so waves), then 40 checks of 4 tokens each
bool run_checks(llama_model * model, std::vector<float> & logits) {
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
    std::mt19937 rng(42);
    const std::vector<llama_token> toks = random_tokens(rng, 20 + 40*4, n_vocab);
    llama_batch batch = llama_batch_init(512, 0, 1);
    bool ok = true;
    size_t pos = 0;
    for (int b = 0; ok && b <= 40; b++) {
        const size_t n = b == 0 ? 20 : 4;
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

int64_t preloads(llama_model * model) {
    llama_moe_stream * ms = model->moe_stream();
    std::unique_lock<std::mutex> lk(ms->mtx);
    return ms->stats.n_preload_issued;
}

} // namespace

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD"); // the lookahead at its default width, as in every rung

    std::vector<float> ref;
    check(run_checks(ref_model, ref), "no streaming: the checks ran");
    llama_model_free(ref_model);

    config c; // a 40-slot desk with the room
    std::vector<float> off, on;
    int64_t pre_off = 0, pre_on = 0;

    unsetenv(g_all_env);
    llama_model * m_off = load(c);
    check(m_off != nullptr, "streamed model (last token only) loaded");
    if (m_off) {
        check(run_checks(m_off, off), "last token only: the checks ran");
        pre_off = preloads(m_off);
        llama_model_free(m_off);
    }

    setenv(g_all_env, "1", 1);
    llama_model * m_on = load(c);
    check(m_on != nullptr, "streamed model (every token) loaded");
    if (m_on) {
        check(run_checks(m_on, on), "every token: the checks ran");
        pre_on = preloads(m_on);
        llama_model_free(m_on);
    }
    std::vector<float> narrow;
    int64_t pre_narrow = 0;
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_ALL_RANKS", "3", 1);
    llama_model * m_narrow = load(c);
    check(m_narrow != nullptr, "streamed model (every token, 3 books each before the last) loaded");
    if (m_narrow) {
        check(run_checks(m_narrow, narrow), "every token, 3 ranks: the checks ran");
        pre_narrow = preloads(m_narrow);
        llama_model_free(m_narrow);
    }
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_ALL_RANKS");
    unsetenv(g_all_env);

    check(same(narrow, ref), "every token, 3 ranks: the same logits as no streaming");
    check(pre_off < pre_narrow && pre_narrow < pre_on, "3 ranks sit between last token only and every rank (" +
            std::to_string(pre_off) + " < " + std::to_string(pre_narrow) + " < " + std::to_string(pre_on) + ")");

    check(nonzero(ref), "the reference logits are not all zero");
    check(same(off, ref), "last token only: the same logits as no streaming");
    check(same(on, ref), "every token: the same logits as no streaming");
    check(same(on, off), "every token: the same logits as last token only");
    check(pre_on > pre_off, "every token issued more prefetches (" + std::to_string(pre_on) + " against " +
            std::to_string(pre_off) + ")");
    printf("prefetches issued: last token only %lld, every token 3 ranks %lld, every token %lld\n",
            (long long) pre_off, (long long) pre_narrow, (long long) pre_on);
    return finish();
}

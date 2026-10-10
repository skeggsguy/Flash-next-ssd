// The ceiling tax (LLAMA_RS_SKIP_SLOTS, src/llama-rs-skip.h, #ceiling-tax): with the switch on, a round
// writes only the undo copies (rollback snapshots) of the fixed summary that a rollback after it can reach,
// slots 0..min(n, K)-1, and the memory refuses any rollback deeper than the last round's n - 1.
//
//   exact:    a round of n tokens, then a rollback of r = 0..min(n-1, n_rs_seq), then a replay: the summary
//             read back (the rolled-back slot) and the replay's logits and states are byte-equal, switch on
//             against off. The prefill round writes every slot first, so a slot the switch failed to write
//             still holds the prefill's copy and shows up as a byte difference.
//   refused:  on, a rollback of n or n + 1 is refused, moves nothing (state bytes and position unchanged),
//             and n - 1 is then taken; off, n is taken (on qwen4exp it reads the round's start exactly).
//   restored: on, a state read back (a bookmark copy) carries no undo copies: rollback 1 is refused until
//             the next round.
//   copies:   the graph's copies into the summaries for one round, counted through cb_eval: with the
//             switch on they no longer depend on the ceiling (n_rs_seq 5 against 10), off they do, and a
//             round of K or more tokens writes the same with both.
//
// The generated models' logits cannot see a stale conv history (qwen35-dense replays byte-equal logits
// from a wrong one), so every check compares the states' bytes too.

#include "arg.h"
#include "common.h"
#include "llama-cpp.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int n_vocab = 0;
static int n_fail  = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++n_fail; } } while (0)

struct copy_count {
    bool counting = false;
    int  copies   = 0;
};

// a copy into the recurrent memory's rows: cache_r_l*, cache_s_l*, cache_ple_r_l*
static bool count_copies(ggml_tensor * t, bool ask, void * user_data) {
    auto * c = (copy_count *) user_data;
    if (!ask || !c->counting) {
        return false;
    }
    const ggml_tensor * dst = t->src[1];
    if (t->op == GGML_OP_CPY && dst != nullptr && dst->view_src != nullptr) {
        const char * name = dst->view_src->name;
        if (strncmp(name, "cache_r_l", 9) == 0 || strncmp(name, "cache_s_l", 9) == 0 ||
            strncmp(name, "cache_ple_r_l", 13) == 0) {
            ++c->copies;
        }
    }
    return false;
}

static llama_context_ptr make_ctx(llama_model * model, bool skip, uint32_t n_rs_seq, copy_count * cc = nullptr) {
    // the switch is read once, when the context is made
    setenv("LLAMA_RS_SKIP_SLOTS", skip ? "1" : "0", 1);
    auto cparams = llama_context_default_params();
    cparams.n_ctx     = 256;
    cparams.n_batch   = 64;
    cparams.n_ubatch  = 64;
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = n_rs_seq;
    if (cc != nullptr) {
        cparams.cb_eval           = count_copies;
        cparams.cb_eval_user_data = cc;
    }
    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    unsetenv("LLAMA_RS_SKIP_SLOTS");
    return ctx;
}

static llama_token tok(llama_pos pos, int salt) {
    return (llama_token) ((7*(uint32_t) pos + 13*(uint32_t) salt + 1) % (uint32_t) n_vocab);
}

// one batch [p0, p1) on seq 0, every logit kept
static bool decode(llama_context * ctx, llama_pos p0, llama_pos p1, int salt) {
    llama_batch batch = llama_batch_init(p1 - p0, 0, 1);
    for (llama_pos pos = p0; pos < p1; ++pos) {
        common_batch_add(batch, tok(pos, salt), pos, { 0 }, true);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// the seq's saved state: the recurrent rows only (the slot a pending rollback reads), or everything
static std::vector<uint8_t> state(llama_context * ctx, bool recurrent_only) {
    const llama_state_seq_flags flags = recurrent_only ? LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY : 0;
    std::vector<uint8_t> buf(llama_state_seq_get_size_ext(ctx, 0, flags));
    buf.resize(llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, flags));
    return buf;
}

static std::vector<float> logits(llama_context * ctx, int n_out) {
    std::vector<float> out;
    for (int i = 0; i < n_out; ++i) {
        const float * l = llama_get_logits_ith(ctx, i);
        out.insert(out.end(), l, l + n_vocab);
    }
    return out;
}

static bool rollback(llama_context * ctx, llama_pos to) {
    return llama_memory_seq_rm(llama_get_memory(ctx), 0, to, -1);
}

static llama_pos pos_max(llama_context * ctx) {
    return llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
}

constexpr uint32_t  N_RS_SEQ = 5;  // the everyday ceiling
constexpr llama_pos P        = 8;  // the prefill round: more than K tokens, so it writes every slot

// prefill, then a round of n: both contexts end at P + n - 1
static bool prefill_and_round(llama_context * ctx, uint32_t n) {
    return decode(ctx, 0, P, 0) && decode(ctx, P, P + (llama_pos) n, 1);
}

static void test_exact(llama_model * model) {
    int n_cases = 0;
    for (uint32_t n : { 1u, 2u, 3u, 5u, 6u, 7u }) {
        const uint32_t r_max = std::min(n - 1, N_RS_SEQ);
        for (uint32_t r = 0; r <= r_max; ++r) {
            auto off = make_ctx(model, false, N_RS_SEQ);
            auto on  = make_ctx(model, true,  N_RS_SEQ);
            if (!off || !on || !prefill_and_round(off.get(), n) || !prefill_and_round(on.get(), n)) {
                CHECK(false, "n %u r %u: setup failed", n, r);
                return;
            }
            const llama_pos to = P + (llama_pos) (n - r);
            if (r > 0) {
                CHECK(rollback(off.get(), to), "n %u: off refused a rollback of %u", n, r);
                CHECK(rollback(on.get(),  to), "n %u: on refused a rollback of %u", n, r);
            }
            CHECK(state(off.get(), true) == state(on.get(), true),
                    "n %u r %u: the summary read back differs, on against off", n, r);

            // the replay continues with other words than the ones rolled back
            const int n_replay = (int) r + 1;
            const bool ok = decode(off.get(), to, to + n_replay, 2) && decode(on.get(), to, to + n_replay, 2);
            CHECK(ok, "n %u r %u: replay failed", n, r);
            if (ok) {
                CHECK(logits(off.get(), n_replay) == logits(on.get(), n_replay), "n %u r %u: replay logits differ", n, r);
                CHECK(state(off.get(), false) == state(on.get(), false), "n %u r %u: states after the replay differ", n, r);
            }
            ++n_cases;
        }
    }
    fprintf(stderr, "%s: %d round/rollback cases, on byte-equal to off\n", __func__, n_cases);
}

static void test_refused(llama_model * model, bool is_qwen4exp) {
    for (uint32_t n : { 1u, 3u, 4u }) {
        auto on = make_ctx(model, true, N_RS_SEQ);
        if (!on || !prefill_and_round(on.get(), n)) {
            CHECK(false, "n %u: setup failed", n);
            return;
        }
        const llama_pos last = P + (llama_pos) n - 1;
        const auto before = state(on.get(), false);
        CHECK(!rollback(on.get(), P),     "n %u: a rollback of the whole round was taken", n);
        CHECK(!rollback(on.get(), P - 1), "n %u: a rollback of n + 1 was taken", n);
        CHECK(pos_max(on.get()) == last, "n %u: a refused rollback moved the position", n);
        CHECK(state(on.get(), false) == before, "n %u: a refused rollback changed the memory", n);
        if (n >= 2) {
            CHECK(rollback(on.get(), P + 1), "n %u: the rollback of n - 1 after a refusal was refused", n);
            CHECK(pos_max(on.get()) == P, "n %u: the rollback of n - 1 left the wrong position", n);
        }

        // off, today's best-effort rollback of the whole round is taken; on qwen4exp it is exact (the round's
        // start, from the clamp and qwen4exp's own conv slots). The shared conv path never writes slot n.
        auto off = make_ctx(model, false, N_RS_SEQ);
        auto ref = make_ctx(model, false, N_RS_SEQ);
        if (!off || !ref || !prefill_and_round(off.get(), n) || !decode(ref.get(), 0, P, 0)) {
            CHECK(false, "n %u: off setup failed", n);
            return;
        }
        CHECK(rollback(off.get(), P), "n %u: off refused a rollback of the whole round", n);
        if (is_qwen4exp) {
            CHECK(state(off.get(), true) == state(ref.get(), true), "n %u: off's rollback of n is not the round's start", n);
        }
    }
    fprintf(stderr, "%s: rollbacks past the round refused on, taken off\n", __func__);
}

// a bookmark copy of one seq (whole = false) or of the whole context (whole = true)
static std::vector<uint8_t> save(llama_context * ctx, bool whole) {
    std::vector<uint8_t> buf(whole ? llama_state_get_size(ctx) : llama_state_seq_get_size(ctx, 0));
    buf.resize(whole ? llama_state_get_data(ctx, buf.data(), buf.size())
                     : llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0));
    return buf;
}

static bool load(llama_context * ctx, const std::vector<uint8_t> & buf, bool whole) {
    return (whole ? llama_state_set_data(ctx, buf.data(), buf.size())
                  : llama_state_seq_set_data(ctx, buf.data(), buf.size(), 0)) == buf.size();
}

static void test_restored(llama_model * model) {
    for (bool whole : { false, true }) {
        auto src = make_ctx(model, true, N_RS_SEQ);
        auto dst = make_ctx(model, true, N_RS_SEQ);
        if (!src || !dst || !prefill_and_round(src.get(), 3)) {
            CHECK(false, "setup failed");
            return;
        }
        const auto saved = save(src.get(), whole);
        CHECK(load(dst.get(), saved, whole), "whole %d: restore failed", whole);
        CHECK(load(src.get(), saved, whole), "whole %d: restore over itself failed", whole);

        const llama_pos last = P + 2;
        for (llama_context * ctx : { src.get(), dst.get() }) {
            CHECK(pos_max(ctx) == last, "whole %d: restored at the wrong position", whole);
            CHECK(!rollback(ctx, last), "whole %d: a rollback of 1 right after a restore was taken", whole);
            // the next round can be rolled back again
            CHECK(decode(ctx, last + 1, last + 3, 3), "whole %d: decode after a restore failed", whole);
            CHECK(rollback(ctx, last + 2), "whole %d: a rollback of 1 after the next round was refused", whole);
        }
    }
    fprintf(stderr, "%s: a restored state (one seq or the whole context) refuses rollbacks until its next round\n", __func__);
}

// the copies into the summaries one round makes, after a prefill
static int round_copies(llama_model * model, bool skip, uint32_t n_rs_seq, uint32_t n) {
    copy_count cc;
    auto ctx = make_ctx(model, skip, n_rs_seq, &cc);
    if (!ctx || !decode(ctx.get(), 0, 16, 0)) {
        return -1;
    }
    cc.counting = true;
    const bool ok = decode(ctx.get(), 16, 16 + (llama_pos) n, 1);
    return ok ? cc.copies : -1;
}

static void test_copies(llama_model * model) {
    for (uint32_t n : { 1u, 3u }) {
        const int off5 = round_copies(model, false, 5, n), off10 = round_copies(model, false, 10, n);
        const int on5  = round_copies(model, true,  5, n), on10  = round_copies(model, true,  10, n);
        fprintf(stderr, "%s: a round of %u: copies into the summaries off %d / %d, on %d / %d (ceiling 5 / 10)\n",
                __func__, n, off5, off10, on5, on10);
        CHECK(off5 > 0 && on5 > 0, "round of %u: no copies counted", n);
        CHECK(on5 == on10, "round of %u: with the switch on the copies still depend on the ceiling", n);
        CHECK(off10 > off5, "round of %u: off, the ceiling's copies did not show", n);
        CHECK(on5 < off5, "round of %u: the switch skipped nothing", n);
    }
    // a round of K or more writes every slot either way
    const int off_k = round_copies(model, false, 5, 6), on_k = round_copies(model, true, 5, 6);
    CHECK(off_k == on_k && off_k > 0, "a round of K: off %d, on %d copies", off_k, on_k);
    CHECK(round_copies(model, true, 5, 1) < round_copies(model, true, 5, 3), "on, a longer round wrote no more slots");
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();

    llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params)));
    if (!model) {
        fprintf(stderr, "failed to load %s\n", params.model.path.c_str());
        return 1;
    }
    n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

    char arch[64] = {};
    llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch));
    {
        auto probe = make_ctx(model.get(), true, N_RS_SEQ);
        if (!probe || llama_n_rs_seq(probe.get()) != N_RS_SEQ) {
            fprintf(stderr, "%s has no recurrent rollback: nothing to test\n", arch);
            return 1;
        }
    }

    test_exact(model.get());
    test_refused(model.get(), strcmp(arch, "qwen4exp") == 0);
    test_restored(model.get());
    test_copies(model.get());

    fprintf(stderr, "%s: %s, %d failure(s)\n", arch, n_fail == 0 ? "OK" : "FAILED", n_fail);
    return n_fail == 0 ? 0 : 1;
}

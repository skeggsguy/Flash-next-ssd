// The keep rule (LLAMA_MOE_STREAM_LOOKAHEAD_KEEP, src/llama-moe-stream-keep.cpp) on the four-floor 64-book
// fixture: a book the lookahead guessed for a floor, one floor ahead or two, stays on that floor's desk until the
// floor stops. The rule changes which slot a book sits in and when a trip is made, never which book a token
// reads, so the check is byte for byte: a 20-token prompt (waves), 40 single written tokens, 20 checks of 4
// tokens, a 40-token read-in (the reading room) and 10 more single tokens, with every guess on (two floors ahead,
// every token of a check), give the same logits and the same MoE output at every floor as no streaming, with the
// rule off and on (the switch is read at load). And the rule must be seen working, or this would pass by doing
// nothing: off, guessed books are put back unread before their floor stops and nothing is forced; on, the only
// guessed books put back before their floor stops are the ones a floor's own need took rather than wait
// (n_keep_forced), and every floor's kept slots were released at its stop. At the default width (8 books a word)
// the guesses never crowd the default desk (50 slots on this fixture, with the room) enough to force a take, so a
// last arm guesses 24 books a word on a 42-slot desk (a budget of 49): kept books fill the desk, and a floor's
// need must take them rather than wait (forced > 0), still byte for byte, and without a hang.

#include "test-moe-room-model.h"

namespace {

bool run_writing(llama_model * model, outputs & out) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 1024;
    cp.n_batch    = 512;
    cp.n_ubatch   = 512;
    cp.op_offload = false;
    cp.cb_eval           = capture_moe_out;
    cp.cb_eval_user_data = &out;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        return false;
    }
    std::mt19937 rng(11);
    std::vector<size_t> sizes = { 20 };
    sizes.insert(sizes.end(), 40, 1);
    sizes.insert(sizes.end(), 20, 4);
    sizes.push_back(40);
    sizes.insert(sizes.end(), 10, 1);
    size_t total = 0;
    for (const size_t n : sizes) {
        total += n;
    }
    const std::vector<llama_token> toks = random_tokens(rng, total, n_vocab);
    llama_batch batch = llama_batch_init(512, 0, 1);
    bool ok = true;
    size_t pos = 0;
    for (size_t b = 0; ok && b < sizes.size(); b++) {
        common_batch_clear(batch);
        for (size_t i = 0; i < sizes[b]; i++, pos++) {
            common_batch_add(batch, toks[pos], (llama_pos) pos, { 0 }, true);
        }
        ok = llama_decode(ctx, batch) == 0;
        if (ok) {
            const float * l = llama_get_logits(ctx);
            out.logits.insert(out.logits.end(), l, l + sizes[b]*n_vocab);
        }
    }
    llama_batch_free(batch);
    llama_free(ctx);
    return ok;
}

struct keep_counts {
    bool    rule = false;
    int64_t la1_issued = 0, la1_used = 0, la1_put_back = 0;
    int64_t la2_issued = 0, la2_used = 0, la2_put_back = 0;
    int64_t forced = 0;
    int64_t kept_left = 0;  // slots still kept after the run, over every floor (every floor has stopped since)
    int64_t room_groups = 0;
    uint32_t desk = 0;      // slots a floor
};

keep_counts counts(llama_model * model) {
    llama_moe_stream * ms = model->moe_stream();
    std::unique_lock<std::mutex> lk(ms->mtx);
    keep_counts c;
    c.rule = ms->la_keep;
    c.la1_issued = ms->stats.n_la1_issued; c.la1_used = ms->stats.n_la1_used; c.la1_put_back = ms->stats.n_la1_put_back;
    c.la2_issued = ms->stats.n_la2_issued; c.la2_used = ms->stats.n_la2_used; c.la2_put_back = ms->stats.n_la2_put_back;
    c.forced = ms->stats.n_keep_forced;
    for (const auto & sl : ms->layers) {
        if (sl) {
            c.kept_left += sl->n_slot_kept;
            for (const uint8_t k : sl->slot_kept) {
                c.kept_left += k != 0;
            }
        }
    }
    c.room_groups = ms->room ? ms->room->stats.n_groups : 0;
    c.desk = ms->n_slots;
    return c;
}

std::string str(const keep_counts & c) {
    char buf[256];
    snprintf(buf, sizeof(buf), "desk %u: 1 floor %lld/%lld used, %lld put back; 2 floors %lld/%lld used, %lld put back; forced %lld",
            c.desk, (long long) c.la1_used, (long long) c.la1_issued, (long long) c.la1_put_back,
            (long long) c.la2_used, (long long) c.la2_issued, (long long) c.la2_put_back, (long long) c.forced);
    return buf;
}

bool arm(const char * keep, outputs & out, keep_counts & c, const char * width = nullptr, uint32_t slots = 0) {
    if (width) {
        setenv("LLAMA_MOE_STREAM_LOOKAHEAD", width, 1);
    } else {
        unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD"); // the lookahead at its default width, as in every rung
    }
    if (keep) {
        setenv("LLAMA_MOE_STREAM_LOOKAHEAD_KEEP", keep, 1);
    } else {
        unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_KEEP");
    }
    config cfg; // the room, and a desk of 50 slots on this fixture
    if (slots > 0) {
        cfg.slots = slots;
    }
    llama_model * m = load(cfg);
    if (m == nullptr) {
        return false;
    }
    const bool ok = n_streamed_floors(m) >= 3 && run_writing(m, out);
    c = counts(m);
    llama_model_free(m);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", "4", 1);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD_ALL", "1", 1);
    // the put-back rule this was proven on (hotness): under LRU, the default since 2026-10-15, this fixture's desk
    // puts back no guess unread with the rule off, so "the rule must be seen working" would have nothing to see
    setenv("LLAMA_MOE_STREAM_LRU", "0", 1);

    outputs ref;
    check(run_writing(ref_model, ref), "no streaming: the tokens ran");
    llama_model_free(ref_model);

    outputs off, on, zero, wide;
    keep_counts c_off, c_on, c_zero, c_wide;
    check(arm(nullptr, off, c_off), "keep rule unset: the tokens ran");
    check(arm("0", zero, c_zero), "keep rule \"0\": the tokens ran");
    check(arm("1", on, c_on), "keep rule on: the tokens ran");
    check(arm("1", wide, c_wide, "24", 49), "keep rule on, 24 books a word guessed, desk 42: the tokens ran");
    check(c_on.desk == 50 && c_wide.desk == 42, "the desks are 50 and 42 slots (" + std::to_string(c_on.desk) + ", " +
            std::to_string(c_wide.desk) + ")");
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD");
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_KEEP");
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_ALL");
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2");

    check(nonzero(ref.logits) && nonzero(ref.moe), "the reference logits and MoE outputs are not all zero");
    check(same(off.logits, ref.logits) && same(off.moe, ref.moe), "rule off: logits and every ffn_moe_out byte-identical to no streaming");
    check(same(zero.logits, ref.logits) && same(zero.moe, ref.moe), "rule \"0\": logits and every ffn_moe_out byte-identical to no streaming");
    check(same(on.logits, ref.logits) && same(on.moe, ref.moe), "rule on: logits and every ffn_moe_out byte-identical to no streaming");
    check(same(wide.logits, ref.logits) && same(wide.moe, ref.moe), "rule on, wide: logits and every ffn_moe_out byte-identical");

    printf("  off:  %s\n  on:   %s\n  wide: %s\n", str(c_off).c_str(), str(c_on).c_str(), str(c_wide).c_str());
    check(!c_off.rule && !c_zero.rule && c_on.rule, "the switch: unset and \"0\" are off, \"1\" is on");
    check(c_on.room_groups > 0 && c_off.room_groups > 0, "the 40-token read-in went through the reading room");
    check(c_off.la1_issued > 0 && c_off.la2_issued > 0 && c_on.la1_issued > 0 && c_on.la2_issued > 0,
            "guesses one and two floors ahead were fetched, rule off and on");
    check(c_off.la1_used > 0 && c_off.la2_used > 0 && c_on.la1_used > 0 && c_on.la2_used > 0,
            "some of each were read by their floor, rule off and on");
    check(c_off.forced == 0 && c_zero.forced == 0, "rule off: nothing forced");
    check(c_off.la1_put_back + c_off.la2_put_back > 0, "rule off: guessed books put back unread before their floor stopped");
    check(c_on.la1_put_back + c_on.la2_put_back == c_on.forced,
            "rule on: the only books put back before their floor stopped are the forced ones");
    check(c_wide.forced > 0, "rule on, wide: a floor's own need took a kept slot rather than wait");
    check(c_wide.la1_put_back + c_wide.la2_put_back == c_wide.forced,
            "rule on, wide: the only books put back before their floor stopped are the forced ones");
    check(c_on.kept_left == 0 && c_off.kept_left == 0 && c_wide.kept_left == 0,
            "every floor's kept slots were released at its stop");
    return finish();
}

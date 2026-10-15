// Desk books first (study #39) on a read-in that asks for the last word's logits only, as a server without the
// apprentice reads a prompt in: llama.cpp then runs the last floor on the output rows alone, so a ubatch before the
// last has no rows there at all, and the last ubatch's last floor has one. Found when desk books first became the
// engine's default (2026-10-16): the floor with no rows took desk books first, the CPU skipped its empty stop, Metal
// still called its gate fn, and the gate aborted ("gate ran with no stop before it"). Every earlier test asked for
// every row's logits, so no floor was ever empty. Checked, on the 64-book fixtures:
//   ran    a 60-token prompt in ubatches of 8 (last logits only), then 6 written words, with the switch unset (on,
//          the default) and at "1": no abort, every decode returns 0
//   same   the logits, prompt and written, byte-identical to no streaming at all
//   rows   on Metal, the last ubatch's one-row last floor stopped desk-first too (stops > written words x floors),
//          and no floor stopped desk-first with the switch at "0"
// Mutation (by hand): drop `n_tokens < 1 ||` from moe_stream_desk_first_on and the process aborts.

#include "test-moe-room-model.h"

namespace {

const char * const ENV_DF = "LLAMA_MOE_STREAM_DESK_FIRST";
const int N_PROMPT = 60, N_UBATCH = 8, N_WRITE = 6;

struct arm {
    bool ran = false;
    outputs out;
    int64_t stops  = 0;
    int64_t floors = 0;
};

arm run_arm(const config & c, const std::vector<segment> & segs, const char * df) {
    arm a;
    df != nullptr ? setenv(ENV_DF, df, 1) : unsetenv(ENV_DF);
    llama_model * m = load(c);
    a.ran = m != nullptr && run(m, N_UBATCH, segs, a.out, false); // the context reads the switch when it opens
    unsetenv(ENV_DF);
    if (m != nullptr) {
        a.stops  = m->moe_stream()->stats.n_df_stops;
        a.floors = n_streamed_floors(m);
    }
    llama_model_free(m);
    return a;
}

} // namespace

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    const bool gpu = g_ngl > 0 && ggml_backend_reg_by_name("MTL") != nullptr;
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD", "0", 1);
    std::mt19937 rng(7);
    segment s;
    s.prompt  = random_tokens(rng, N_PROMPT, llama_vocab_n_tokens(llama_model_get_vocab(ref_model)));
    s.n_write = N_WRITE;
    s.all     = false; // the last word's logits only: the earlier ubatches' last floor has no rows
    const std::vector<segment> segs = { s };

    outputs ref;
    check(run(ref_model, N_UBATCH, segs, ref, false), "no streaming: ran");
    llama_model_free(ref_model);

    config c; // a desk of 40 of 64 books a floor: 8-token ubatches take waves, single rows one wave
    c.room_mode = LLAMA_MOE_ROOM_OFF;
    const arm dflt = run_arm(c, segs, nullptr);
    const arm on   = run_arm(c, segs, "1");
    const arm off  = run_arm(c, segs, "0");
    printf("  desk-first stops: unset %lld, \"1\" %lld, \"0\" %lld (streamed floors %lld)\n", (long long) dflt.stops,
            (long long) on.stops, (long long) off.stops, (long long) on.floors);
    check(dflt.ran && on.ran && off.ran, "unset, \"1\" and \"0\": the read-in and every written word ran");
    for (const arm * a : { &dflt, &on, &off }) {
        check(same(ref.logits, a->out.logits), "logits byte-identical to no streaming");
    }
    check(off.stops == 0, "\"0\": no desk-first stop");
    if (gpu) {
        for (const arm * a : { &dflt, &on }) {
            check(a->floors > 0 && a->stops > (int64_t) N_WRITE*a->floors,
                    "the read-in's one-row last floor stopped desk-first beside the written words");
        }
    } else {
        check(dflt.stops == 0 && on.stops == 0, "CPU desk: desk books first stays off");
    }
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD");
    printf("%s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}

// The reading room (--moe-stream-room) on the 64-book fixtures, byte for byte against the same model with
// every book resident (the harness and the promise: test-moe-room-model.h; the study's I/O path and the
// apprentice: test-moe-room-model-paths.cpp).
//
// Scenarios, each with the room's groups counted so a test that silently fell back to waves fails. They
// run at the old threshold of 35 tokens (the harness sets LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS; setup says why)
// except "default", which unsets it:
//   long      300 tokens read in at -ub 128 (128, 128, 44) then 8 written, parts 4 and parts 1
//   t_min     105 tokens at -ub 35, the smallest ubatch the room takes
//   default   the fixtures' own threshold, 160 (20 slips per book): 320 tokens at -ub 160 take the room,
//             159 tokens take waves
//   on by default  llama_model_default_params() takes auto's room where it fits (B through it, byte for
//             byte), and where the room is refused (40 slots; floors on two devices; a batch under the
//             threshold) loads with it off and a warning, while an asked-for auto stops the load; a load
//             without streaming says nothing about the room and makes none
//   last row  B asking for its last row only, as a server does: the last floor takes waves beside the room
//   empty     a desk of 62 of 64 books, so two of the four parts hold nothing (all-skip links)
//   desks     prompt B after A and after C on two fresh models: different desks, the same bytes
//   short     20 tokens, under the threshold: waves, not the room (printed against the reference)
//   poison    LLAMA_MOE_ROOM_POISON=1 fills handed-back parts with 0xFF; run WITHOUT the eval callback,
//             which would make every split synchronous and hide a part handed back too early
// Waves against the reference is printed, not asserted: their exactness is not the room's promise.

#include "test-moe-room-model.h"

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(ref_model));
    std::mt19937 rng(1234);
    const segment A = { random_tokens(rng, 300, n_vocab), 8, false };
    const segment B = { random_tokens(rng, 300, n_vocab), 8, true };
    const segment C = { random_tokens(rng, 300, n_vocab), 8, false };
    const segment D = { random_tokens(rng, 20, n_vocab), 0, false };
    const segment E = { random_tokens(rng, 20, n_vocab), 0, false };
    const segment S = { random_tokens(rng, 20, n_vocab), 4, true };
    const segment T = { random_tokens(rng, 105, n_vocab), 4, true };
    const segment G = { random_tokens(rng, 320, n_vocab), 4, true };
    const segment H = { random_tokens(rng, 159, n_vocab), 0, true };
    segment L = B; // B asking for its last row only: a floor that keeps only output rows (qwen3moe's last) takes waves
    L.all = false;

    outputs ref_b, ref_s, ref_t, ref_l, ref_g, ref_h;
    bool ok = run(ref_model, 128, { B }, ref_b, true) && run(ref_model, 128, { S }, ref_s, true) &&
              run(ref_model, 35, { T }, ref_t, true) && run(ref_model, 128, { L }, ref_l, true) &&
              run(ref_model, 160, { G }, ref_g, true) && run(ref_model, 160, { H }, ref_h, true);
    llama_model_free(ref_model);
    if (!ok) {
        fprintf(stderr, "the reference run failed\n");
        return 1;
    }
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);

    config room4, room1 = room4, waves = room4, empty = room4;
    room1.parts     = 1;
    waves.room_mode = LLAMA_MOE_ROOM_OFF;
    waves.slots     = 40;
    empty.slots      = 63;                  // 2.25 MiB of room leaves 62 slots: 2 books a floor on the belt
    empty.room_mode  = LLAMA_MOE_ROOM_GIB;
    empty.room_value = 2.25f/1024.0f;

    // 300 tokens at -ub 128 is three ubatches, all over the 35-token threshold
    scenario("long, parts 4", room4, 128, { B }, ref_b, 3);
    scenario("long, parts 1", room1, 128, { B }, ref_b, 3);
    scenario("long, waves",   waves, 128, { B }, ref_b, 0);
    scenario("t_min: -ub 35", room4, 35, { T }, ref_t, 3);
    {
        // the default threshold, 20 slips per book: 160 tokens take the room, 159 do not
        unsetenv("LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS");
        llama_model * model = load(room4);
        check(model != nullptr && model->moe_stream()->room_layout.sweep_min_tokens == 160,
                "default: the fixtures' threshold is 160 tokens");
        llama_model_free(model);
        scenario("default: 320 tokens at -ub 160", room4, 160, { G }, ref_g, 2);
        scenario("default: 159 tokens take waves", room4, 160, { H }, ref_h, 0);
        setenv("LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS", G_SWEEP_MIN_TOKENS, 1);
    }
    {
        // The room is on by default (Tom, 2026-09-26): a load that asks for nothing gets auto's room where
        // it can be made. It is a mode of its own, not auto, because where the room cannot be made the
        // default steps back to off and loads, while a room asked for stops the load.
        config dflt = room4;
        dflt.room_mode = llama_model_default_params().moe_stream_room_mode;
        check(dflt.room_mode == LLAMA_MOE_ROOM_DEFAULT, "on by default: the default params leave the room to the default");
        llama_model * m_auto = load(room4);
        g_room_log.clear();
        llama_model * m_dflt = load(dflt);
        const bool both = m_auto != nullptr && m_dflt != nullptr && room_of(m_auto) && room_of(m_dflt);
        check(both && room_of(m_dflt)->lay.room_bytes == room_of(m_auto)->lay.room_bytes &&
              m_dflt->moe_stream()->n_slots == m_auto->moe_stream()->n_slots,
              "on by default: auto's room and desk (" + std::to_string(both ? m_dflt->moe_stream()->n_slots : 0) + " slots)");
        check(g_room_log.find("reading room: ") != std::string::npos && g_room_log.find("floors of look-ahead") != std::string::npos &&
              g_room_log.find("on by default but") == std::string::npos, "on by default: the startup line says the room, no warning");
        printf("  | %s", g_room_log.substr(0, g_room_log.find('\n') + 1).c_str());
        llama_model_free(m_auto);
        llama_model_free(m_dflt);
        scenario("on by default: B through the room", dflt, 128, { B }, ref_b, 3);

        // where it cannot be made it steps back: 1.25 floors do not fit a 40-slot budget
        config small = dflt, small_auto = room4;
        small.slots = small_auto.slots = 40;
        g_room_log.clear();
        llama_model * m = load(small);
        check(m != nullptr && room_of(m) == nullptr && m->moe_stream()->n_slots == 40,
                "on by default, 40 slots: loads with the room off and the whole desk");
        const size_t off_at = g_room_log.find("reading room: off;");
        const size_t why_at = g_room_log.find("on by default but cannot be made here");
        check(off_at < why_at && why_at != std::string::npos &&
              g_room_log.find("takes the whole desk budget") != std::string::npos,
              "on by default, 40 slots: the off line, then a warning that says why");
        llama_model_free(m);
        g_room_log.clear();
        check(load(small_auto) == nullptr && g_room_log.find("takes the whole desk budget") != std::string::npos,
                "auto asked for, 40 slots: the load stops, saying why");

        // A batch that can never reach the threshold (35 here, 1,024 on the library against llama.cpp's own
        // -ub 512): told so, the default room stays off and says why, a room asked for stops the load, and
        // a batch that reaches it keeps the room. Not told (0, every other scenario), the room is made.
        config low = dflt, low_auto = room4, told = dflt;
        low.ubatch = low_auto.ubatch = 34;
        told.ubatch = 35;
        g_room_log.clear();
        m = load(low);
        check(m != nullptr && room_of(m) == nullptr && m->moe_stream()->n_slots == room4.slots &&
              g_room_log.find("reading room: off;") < g_room_log.find("on by default but cannot be made here") &&
              g_room_log.find("batches of 34 tokens (-ub), fewer than the 35") != std::string::npos,
              "on by default, -ub 34: loads with the room off and the whole desk, saying why");
        llama_model_free(m);
        g_room_log.clear();
        check(load(low_auto) == nullptr && g_room_log.find("the room would never be used") != std::string::npos,
                "auto asked for, -ub 34: the load stops, saying why");
        m = load(told);
        check(m != nullptr && room_of(m) != nullptr, "on by default, -ub 35: the room is made");
        llama_model_free(m);

        // floors on two devices: at -ngl 2 the fixture's floor 0 is on the CPU and floor 1 on the GPU
        if (llama_supports_gpu_offload()) {
            const int ngl = g_ngl;
            g_ngl = 2;
            g_room_log.clear();
            m = load(dflt);
            check(m != nullptr && room_of(m) == nullptr && g_room_log.find("on by default but") != std::string::npos &&
                  g_room_log.find("different devices") != std::string::npos,
                  "on by default, floors on two devices: loads with the room off, saying why");
            llama_model_free(m);
            check(load(room4) == nullptr, "auto asked for, floors on two devices: the load stops");
            g_ngl = ngl;
        }

        // without streaming the room is never sized: no manager, no room, not a word about it
        config plain = dflt;
        plain.stream = false;
        g_room_log.clear();
        m = load(plain);
        check(m != nullptr && m->moe_stream() == nullptr && g_room_log.empty(),
                "on by default, no streaming: no room made and nothing said about one");
        llama_model_free(m);
    }
    {
        // the last row only: the last floor works on the output rows alone, under the threshold, so it
        // takes waves in the same ubatch where every other floor takes the room
        llama_model * model = load(room4);
        outputs got;
        check(model != nullptr && run(model, 128, { L }, got, true), "last row only: ran");
        const int64_t groups = room_of(model)->stats.n_groups;
        const int64_t want   = 3*(n_streamed_floors(model) - 1)*(1 + room4.parts);
        check(groups == want, "last row only: room groups ran " + std::to_string(groups) + " of " + std::to_string(want));
        check(same(ref_l.logits, got.logits) && same(ref_l.moe, got.moe), "last row only: byte-identical");
        llama_model_free(model);
    }
    scenario("short: 20 tokens take waves", room4, 128, { S }, ref_s, 0);
    {
        llama_model * model = load(empty);
        outputs got;
        check(model != nullptr && run(model, 128, { B }, got, true), "empty parts: ran");
        size_t n_empty = 0;
        for (const auto & F : room_of(model)->floors) {
            for (const auto & p : F.parts) {
                n_empty += F.sl && p.empty();
            }
        }
        check(n_empty == 2*(size_t) n_streamed_floors(model), "empty parts: two of four parts empty on every floor");
        check(same(ref_b.logits, got.logits) && same(ref_b.moe, got.moe), "empty parts: byte-identical");
        llama_model_free(model);
    }
    {
        // the strongest check: the same prompt after different histories, so different desks
        llama_model * m1 = load(room4);
        llama_model * m2 = load(room4);
        outputs o1, o2, none;
        const bool ran = m1 && m2 && run(m1, 128, { A, D }, none, false) && run(m2, 128, { C, E }, none, false);
        const std::vector<int32_t> d1 = ran ? desk_of(m1) : std::vector<int32_t>();
        const std::vector<int32_t> d2 = ran ? desk_of(m2) : std::vector<int32_t>();
        size_t n_diff = 0;
        for (size_t i = 0; i < d1.size() && i < d2.size(); i++) {
            n_diff += d1[i] != d2[i];
        }
        check(ran && n_diff > 0, "desks: after A and after C the desks differ (" + std::to_string(n_diff) + " slots)");
        check(ran && run(m1, 128, { B }, o1, true) && run(m2, 128, { B }, o2, true), "desks: B ran on both");
        check(same(o1.logits, o2.logits) && same(o1.moe, o2.moe), "desks: B after A == B after C, byte for byte");
        check(same(ref_b.logits, o1.logits) && same(ref_b.moe, o1.moe), "desks: and == no streaming");
        // A or C then B read in through the room, three ubatches each; D, E and the written words take waves
        const int64_t want = ran ? 6*n_streamed_floors(m1)*(1 + room4.parts) : -1;
        check(ran && room_of(m1)->stats.n_groups == want && room_of(m2)->stats.n_groups == want,
                "desks: room groups ran " + std::to_string(want) + " on both");
        llama_model_free(m1);
        llama_model_free(m2);
    }
    {
        // poison: a part handed back while a GEMM still reads it would feed that GEMM 0xFF books
        setenv("LLAMA_MOE_ROOM_POISON", "1", 1);
        scenario("poison, no eval callback, parts 4", room4, 128, { B }, ref_b, 3, false);
        scenario("poison, no eval callback, parts 1", room1, 128, { B }, ref_b, 3, false);
        unsetenv("LLAMA_MOE_ROOM_POISON");
    }

    return finish();
}

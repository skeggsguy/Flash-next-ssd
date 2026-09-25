// The reading room (--moe-stream-room) on the 64-book fixtures, byte for byte against the same model with
// every book resident (the harness and the promise: test-moe-room-model.h; the study's I/O path and the
// apprentice: test-moe-room-model-paths.cpp).
//
// Scenarios, each with the room's groups counted so a test that silently fell back to waves fails:
//   long      300 tokens read in at -ub 128 (128, 128, 44) then 8 written, parts 4 and parts 1
//   t_min     105 tokens at -ub 35, the smallest ubatch the room takes
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
    segment L = B; // B asking for its last row only: a floor that keeps only output rows (qwen3moe's last) takes waves
    L.all = false;

    outputs ref_b, ref_s, ref_t, ref_l;
    bool ok = run(ref_model, 128, { B }, ref_b, true) && run(ref_model, 128, { S }, ref_s, true) &&
              run(ref_model, 35, { T }, ref_t, true) && run(ref_model, 128, { L }, ref_l, true);
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

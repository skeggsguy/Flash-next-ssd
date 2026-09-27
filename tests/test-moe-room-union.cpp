// The reading room under union attention (LLAMA_QSA_UNION=1, the engine's default since the study's
// WRITING-PLAN.md step 1), byte for byte against the same model with every book resident, through the
// harness of test-moe-room-model.h. That harness holds its own scenarios at "bias" (the exact GPU-bias
// route): on the e64 fixtures union attention has no Metal kernel at heads of 128 and runs on the CPU
// beside a Metal graph, where the resident and streamed models place the neighbouring ops differently.
// This test runs on the h256 fixture (heads of 256, the shape the Metal kernel takes) and asks for "1"
// itself, so the room's promise is checked on the route every rung and gate now runs:
//   long    300 tokens read in at -ub 128 (128 and 128 through union attention, 44 through the small host
//           inputs) then 8 written, parts 4: logits and every ffn_moe_out byte-identical to no streaming,
//           the room's groups counted, union floors built in the reference and in the room run alike
//   desks   prompt B after A and after C on two fresh models: different desks, the same bytes, and the
//           same as no streaming
// Each context's startup line must say "qsa union: on,", or the run proved nothing about union attention.

#include "test-moe-room-model.h"

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    // the harness pins "bias" for its own scenarios (setup says why); this test is about "1"
    setenv("LLAMA_QSA_UNION", "1", 1);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(ref_model));
    std::mt19937 rng(1234);
    const segment A = { random_tokens(rng, 300, n_vocab), 8, false };
    const segment B = { random_tokens(rng, 300, n_vocab), 8, true };
    const segment C = { random_tokens(rng, 300, n_vocab), 8, false };
    const segment D = { random_tokens(rng, 20, n_vocab), 0, false };
    const segment E = { random_tokens(rng, 20, n_vocab), 0, false };

    outputs ref_b;
    const bool ok = run(ref_model, 128, { B }, ref_b, true);
    llama_model_free(ref_model);
    if (!ok) {
        fprintf(stderr, "the reference run failed\n");
        return 1;
    }
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);
    const int64_t ref_builds = g_union_attn_builds;
    check(ref_builds > 0, "reference: union attention floors built (" + std::to_string(ref_builds) + ")");

    config room4;
    g_union_attn_builds = 0;
    scenario("union: long, parts 4", room4, 128, { B }, ref_b, 3);
    check(g_union_attn_builds == ref_builds, "union: long, parts 4: the room run built the same union floors ("
            + std::to_string(g_union_attn_builds) + ")");

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
        check(ran && n_diff > 0, "union: desks: after A and after C the desks differ (" + std::to_string(n_diff) + " slots)");
        check(ran && run(m1, 128, { B }, o1, true) && run(m2, 128, { B }, o2, true), "union: desks: B ran on both");
        check(same(o1.logits, o2.logits) && same(o1.moe, o2.moe), "union: desks: B after A == B after C, byte for byte");
        check(same(ref_b.logits, o1.logits) && same(ref_b.moe, o1.moe) && nonzero(o1.moe), "union: desks: and == no streaming, nonzero");
        const int64_t want = ran ? 6*n_streamed_floors(m1)*(1 + room4.parts) : -1;
        check(ran && room_of(m1)->stats.n_groups == want && room_of(m2)->stats.n_groups == want,
                "union: desks: room groups ran " + std::to_string(want) + " on both");
        llama_model_free(m1);
        llama_model_free(m2);
    }

    // every context (the reference, the room runs) took union attention, and none fell back to "bias"
    size_t n_on = 0;
    for (size_t at = g_union_log.find("qsa union: on,"); at != std::string::npos; at = g_union_log.find("qsa union: on,", at + 1)) {
        n_on++;
    }
    check(n_on == 6 && g_union_log.find("qsa union: bias") == std::string::npos,
            "every context's startup line says \"qsa union: on,\" (" + std::to_string(n_on) + " of 6)");

    unsetenv("LLAMA_QSA_UNION");
    return finish();
}

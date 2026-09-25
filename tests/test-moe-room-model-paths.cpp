// The reading room on the study's own path and beside the apprentice, on the 64-book fixtures (the
// harness and the promise: test-moe-room-model.h):
//   direct     --moe-stream-direct (a staging read, then tensor_set onto the belt) with two runners on a
//              byte-identical copy of the fixture, both asserted on; every group ran; byte-identical
//   apprentice the MTP context shares the manager and builds its graph, room disallowed, between two of
//              the target's batches; a same-shape batch then reuses the target's graph, whose desk ops
//              must still find the floors that begin the ubatch (graph reuse asserted via n_reused)

#include "test-moe-room-model.h"

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(ref_model));
    std::mt19937 rng(1234);
    random_tokens(rng, 300, n_vocab); // A, as test-moe-room-model draws it, so B is the same prompt there and here
    const segment B = { random_tokens(rng, 300, n_vocab), 8, true };

    outputs ref_b;
    const bool ok = run(ref_model, 128, { B }, ref_b, true);
    llama_model_free(ref_model);
    if (!ok) {
        fprintf(stderr, "the reference run failed\n");
        return 1;
    }
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);

    const config room4;
    {
        // the study's path: direct I/O (a staging read, then tensor_set onto the belt) and two runners on a
        // byte-identical copy; both are asserted on, or this would test the zero-copy path a second time
        config prod = room4;
        prod.direct = true;
        prod.alt    = g_path + ".twin-ngl" + std::to_string(g_ngl) + ".gguf";
        std::filesystem::copy_file(g_path, prod.alt, std::filesystem::copy_options::overwrite_existing);
        scenario("direct I/O, two runners", prod, 128, { B }, ref_b, 3, true, [](const llama_model * model) {
            const llama_moe_stream * ms = model->moe_stream();
            check(ms->use_direct_io && !ms->files_alt.empty(), "direct I/O, two runners: direct reads and the alt copy are on");
            check(ms->stats.n_bytes_file > 0 && ms->stats.n_bytes_alt > 0, "direct I/O, two runners: both runners read");
        });
        std::filesystem::remove(prod.alt);
    }
    {
        // the apprentice's (MTP) context shares the manager and builds its graph, room disallowed, between
        // two of the target's batches; a same-shape batch then reuses the target's graph, whose desk ops
        // must still find the floors that begin the ubatch
        llama_model * model = load(room4);
        const segment F = { std::vector<llama_token>(B.prompt.begin(), B.prompt.begin() + 128), 0, true };
        const auto apprentice_build = [&] {
            for (const auto & fl : room_of(model)->floors) {
                if (fl.sl) { room_of(model)->take(fl.sl->il, 128, false); }
            }
        };
        outputs got;
        check(model != nullptr && run(model, 128, { F, F }, got, true, apprentice_build), "apprentice between batches: ran");
        check(g_n_reused >= 1, "apprentice between batches: the second batch reused the graph (" + std::to_string(g_n_reused) + ")");
        // both batches read in through the room, or the reused graph's desk ops were never tested
        const int64_t want = model ? 2*n_streamed_floors(model)*(1 + room4.parts) : -1;
        check(model != nullptr && room_of(model)->stats.n_groups == want,
                "apprentice between batches: room groups ran " + std::to_string(want));
        // each batch is B's first ubatch: its logits are ref_b's first 128 rows, its MoE outputs the other batch's
        const size_t n1 = (size_t) 128*n_vocab, m1 = got.moe.size()/2;
        check(got.logits.size() == 2*n1 && m1 > 0 &&
              memcmp(got.logits.data(), ref_b.logits.data(), n1*sizeof(float)) == 0 &&
              memcmp(got.logits.data() + n1, ref_b.logits.data(), n1*sizeof(float)) == 0 &&
              memcmp(got.moe.data(), got.moe.data() + m1, m1*sizeof(float)) == 0,
              "apprentice between batches: both batches byte-identical to no streaming");
        llama_model_free(model);
    }
    return finish();
}

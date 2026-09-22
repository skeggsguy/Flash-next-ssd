// Two runners (--moe-stream-alt-path / --moe-stream-alt-split): the two things that decide which
// bytes come off which drive, and which would both fail silently.
//
//   - the alt copy's sibling shard names must be the ones the split loader would derive from -m,
//     because the two sets share one file_idx per weight. Getting them wrong reads an expert from
//     the wrong shard, which does not crash - it changes the words.
//   - the id split must put alt_split percent of a layer's expert ids on the model's own path.
//     Getting it wrong still runs, still reads, and quietly measures a different stripe than the
//     one the rung config claims.

#include "testing.h"

#include "llama.h"

#include "../src/llama-moe-stream.h"

#include <memory>
#include <string>
#include <vector>

// use_alt only splits when there IS an alt set, so a test of the arithmetic needs one open file.
// Any readable file will do: nothing here reads a byte from it.
static std::unique_ptr<llama_file> some_file() {
    const char * d = std::getenv("TMPDIR");
    const std::string path = (d && *d ? std::string(d) : std::string("/tmp/")) + "/moe-split-probe";
    FILE * f = fopen(path.c_str(), "wb");
    GGML_ASSERT(f != nullptr);
    fputc('x', f);
    fclose(f);
    return std::make_unique<llama_file>(path.c_str(), "rb", false);
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }
    if (!t.verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("sibling_shards", [&](testing & t) {
        const std::string first = "/Volumes/annex/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
        const auto got = llama_moe_stream_alt_paths(first, 4);

        t.assert_equal("one path per shard", (size_t) 4, got.size());
        for (int i = 0; i < 4; i++) {
            char want[512];
            snprintf(want, sizeof(want),
                    "/Volumes/annex/Qwen3.8-Flash-Next-UD-Q4_K_XL-%05d-of-%05d.gguf", i + 1, 4);
            t.assert_equal(std::string(want), got[i]);
        }
        // and it is the same naming the public split helper produces, which is what the loader uses
        char direct[512];
        llama_split_path(direct, sizeof(direct), "/Volumes/annex/Qwen3.8-Flash-Next-UD-Q4_K_XL", 2, 4);
        t.assert_equal("agrees with llama_split_path", std::string(direct), got[2]);
    });

    t.test("a_single_file_model_has_no_siblings", [&](testing & t) {
        const auto got = llama_moe_stream_alt_paths("/Volumes/annex/model.gguf", 1);
        t.assert_equal((size_t) 1, got.size());
        t.assert_equal(std::string("/Volumes/annex/model.gguf"), got[0]);
    });

    t.test("not_the_first_shard_is_refused", [&](testing & t) {
        // pointing at shard 2, or at a differently-split copy, would silently produce a list that
        // does not exist - better to say so at load than to open four missing files
        for (const char * bad : {
                "/Volumes/annex/model-00002-of-00004.gguf",
                "/Volumes/annex/model-00001-of-00006.gguf",
                "/Volumes/annex/model.gguf" }) {
            bool threw = false;
            try {
                llama_moe_stream_alt_paths(bad, 4);
            } catch (const std::exception &) {
                threw = true;
            }
            t.assert_true(std::string("refuses ") + bad, threw);
        }
    });

    t.test("id_split", [&](testing & t) {
        llama_moe_stream mgr(/*n_layer =*/ 1, /*n_slots =*/ 8, /*n_io_threads =*/ 1, /*direct =*/ false);

        // one runner: every expert comes off the model's own path, whatever alt_split says
        mgr.alt_split = 53;
        t.assert_true("no alt set, no split", !mgr.use_alt(0, 512));
        t.assert_true("no alt set, no split", !mgr.use_alt(511, 512));

        mgr.files_alt.emplace_back(some_file().release());

        // 53% of 512 ids is 271.36, so 272 is the first id on the alt runner
        t.assert_true("id 0 stays home",    !mgr.use_alt(0, 512));
        t.assert_true("id 271 stays home",  !mgr.use_alt(271, 512));
        t.assert_true("id 272 goes alt",     mgr.use_alt(272, 512));
        t.assert_true("id 511 goes alt",     mgr.use_alt(511, 512));

        int home = 0;
        for (int e = 0; e < 512; e++) {
            home += mgr.use_alt(e, 512) ? 0 : 1;
        }
        t.assert_equal("272 of 512 ids, i.e. 53.1%", 272, home);

        // the whole point of splitting by id: the share is the same on a smaller expert pool
        mgr.alt_split = 50;
        int half = 0;
        for (int e = 0; e < 128; e++) {
            half += mgr.use_alt(e, 128) ? 0 : 1;
        }
        t.assert_equal("half of 128", 64, half);
    });

    return t.summary();
}

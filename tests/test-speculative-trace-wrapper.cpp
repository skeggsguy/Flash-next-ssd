// The wrapper's rate trace (CHAIN-PLAN step 1, common_speculative_draft / _accept in common/speculative.cpp),
// driven through the public API as the server drives it.
//   no model   ngram-simple (a draft that reports no confidences) with LLAMA_SPEC_RATE_TRACE set: a header naming
//              the draft, an answer line per begin, a line per draft call opened before the draft and closed
//              after the wrapper's trim (checked = what went to the check, stop "trim" when the server's
//              ceiling cut it), kept from the accept, the server's replay counted one word less, a failed draft
//              still a line; unset, no file
//   -m MTP     the guard: today's apprentice writes its own trace, so the file has one header and one closing
//              line (one writer), and its own hook mirrors the replay too
// Usage: test-speculative-trace-wrapper [-m qwen4exp-moe-e64mtp.gguf [-ngl N]]

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const std::string & what) {
    printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

static std::string tmp_path(const char * tag) {
    const char * dir = std::getenv("TMPDIR");
    return std::string(dir && *dir ? dir : "/tmp") + "/test-spec-trace-wrapper-" + tag + "-" +
           std::to_string((long) getpid()) + ".srtr";
}

static std::vector<std::string> read_lines(const std::string & path) {
    std::vector<std::string> out;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        out.push_back(line);
    }
    return out;
}

static bool has(const std::string & line, const std::string & part) {
    return line.find(part) != std::string::npos;
}

static int count_with(const std::vector<std::string> & lines, const std::string & part) {
    int n = 0;
    for (const auto & l : lines) {
        n += has(l, part);
    }
    return n;
}

// history 10 1 2 3 4 5 6 7 8 20 21 1: after the last word 1, the word 2 makes the pair (1, 2) seen at position 1,
// so ngram-simple (n 2, m 4) drafts the four words after it, 3 4 5 6; after 99 nothing matches
static void test_ngram(bool traced) {
    const std::string path = tmp_path(traced ? "ngram" : "ngram-off");
    std::remove(path.c_str());
    if (traced) {
        setenv("LLAMA_SPEC_RATE_TRACE", path.c_str(), 1);
    } else {
        unsetenv("LLAMA_SPEC_RATE_TRACE");
    }
    common_params_speculative params;
    params.types = { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE };
    params.ngram_simple.size_n = 2;
    params.ngram_simple.size_m = 4;
    common_speculative * spec = common_speculative_init(params, 1);
    check(spec != nullptr, "ngram-simple sets up");
    if (spec == nullptr) {
        return;
    }
    llama_tokens hist = { 10, 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 1 };
    llama_tokens draft;
    auto cycle = [&](int n_max, llama_token id_last) {
        draft.clear();
        common_speculative_get_draft_params(spec, 0) = { true, n_max, (llama_pos) hist.size(), id_last, &hist, &draft };
        common_speculative_draft(spec);
        return draft;
    };
    common_speculative_begin(spec, 0, hist);
    const llama_tokens d0 = cycle(-1, 2);
    common_speculative_accept(spec, 0, 3);
    const llama_tokens d1 = cycle(2, 2);
    common_speculative_accept(spec, 0, 2, /*replay*/ true);
    const llama_tokens d2 = cycle(-1, 99);
    common_speculative_free(spec);
    unsetenv("LLAMA_SPEC_RATE_TRACE");

    const std::string tag = traced ? "trace on: " : "trace off: ";
    check(d0 == llama_tokens({ 3, 4, 5, 6 }) && d1 == llama_tokens({ 3, 4 }) && d2.empty(),
          tag + "the drafts (4 words, trimmed to the server's 2, none)");
    const auto l = read_lines(path);
    if (!traced) {
        check(l.empty() && access(path.c_str(), F_OK) != 0, tag + "no file");
        return;
    }
    check(l.size() == 6, "header, answer, 3 cycles, closing line (" + std::to_string(l.size()) + " lines)");
    if (l.size() != 6) {
        std::remove(path.c_str());
        return;
    }
    check(has(l[0], "\"version\":2,") && has(l[0], "\"n_max\":4,") && has(l[0], "\"draft\":\"ngram-simple\",\"block\":0,"),
          "the header names the draft: " + l[0]);
    check(has(l[1], "{\"answer\":0,\"seq\":0,\"n_prompt\":12,"), "an answer line at begin: " + l[1]);
    check(has(l[2], "\"pos\":12,\"cap\":4,\"steps\":0,\"drafted\":4,\"checked\":4,\"kept\":3,\"stop\":\"cap\",") &&
          has(l[2], "\"replay\":0,") && !has(l[2], "\"cycle_us\":-1}"),
          "cycle 0: checked what went to the check, kept from the accept, timed to the next draft: " + l[2]);
    check(has(l[3], "\"cap\":2,\"steps\":0,\"drafted\":4,\"checked\":2,\"kept\":1,\"stop\":\"trim\",") && has(l[3], "\"replay\":1,"),
          "cycle 1: the trim is the stop, a replay of 2 keeps 1: " + l[3]);
    check(has(l[4], "\"drafted\":0,\"checked\":0,\"kept\":0,\"stop\":\"error\",") && has(l[4], "\"accept_us\":-1,") &&
          has(l[4], "\"cycle_us\":-1}"), "cycle 2: a failed draft is still a line: " + l[4]);
    check(has(l[5], "{\"end\":\"spec-rate\",\"cycles\":3,"), "the closing line: " + l[5]);
    std::remove(path.c_str());
}

// the MTP fixture through the wrapper with the trace on: one writer (MTP's own hooks), its replay mirrored
static void test_mtp_guard(const std::string & model_path, int ngl) {
    const std::string path = tmp_path("mtp");
    std::remove(path.c_str());
    setenv("LLAMA_SPEC_RATE_TRACE", path.c_str(), 1);

    common_params params;
    params.model.path   = model_path;
    params.n_gpu_layers = ngl;
    params.fit_params   = false;
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    params.warmup        = false;
    params.n_ctx         = 512;
    params.n_batch       = 512;
    params.n_ubatch      = 64;
    params.sampling.temp = 0.0f;
    params.speculative.types       = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = 3;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0.0f;
    const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel,
            common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;

    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    check(init->model() != nullptr && ctx != nullptr, "the MTP fixture loads");
    if (ctx == nullptr) {
        return;
    }
    common_params params_dft = common_base_params_to_speculative(params);
    auto spec_init = common_speculative_init_from_params(params_dft, init->model(), ctx);
    params.speculative.draft.ctx_tgt = ctx;
    params.speculative.draft.ctx_dft = spec_init->context();
    llama_context * ctx_dft = params.speculative.draft.ctx_dft;
    common_sampler_ptr smpl(common_sampler_init(init->model(), params.sampling));
    common_speculative * spec = ctx_dft ? common_speculative_init(params.speculative, 1) : nullptr;
    check(spec != nullptr, "draft-mtp sets up");
    if (spec == nullptr) {
        return;
    }

    std::mt19937 rng(42);
    llama_tokens prompt;
    for (int i = 0; i < 40; i++) {
        prompt.push_back((llama_token) (1 + rng() % 100));
    }
    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);
    for (size_t i = 0; i + 1 < prompt.size(); i++) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, false);
    }
    bool ok = llama_decode(ctx, batch) == 0 && common_speculative_process(spec, batch);
    llama_token  id_last = prompt.back();
    llama_tokens hist(prompt.begin(), prompt.end() - 1);
    int          n_past  = (int) hist.size();
    common_speculative_begin(spec, 0, hist);
    llama_tokens draft;
    int n_cycles = 0, n_replay = -1, replay_cycle = -1;
    for (; ok && n_cycles < 12; n_cycles++) {
        common_speculative_get_draft_params(spec, 0) = { true, 3, n_past, id_last, &hist, &draft };
        common_speculative_draft(spec);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_past, -1);
        common_batch_clear(batch);
        common_batch_add(batch, id_last, n_past++, { 0 }, true);
        for (size_t i = 0; i < draft.size(); i++) {
            common_batch_add(batch, draft[i], n_past + (llama_pos) i, { 0 }, true);
        }
        ok = llama_decode(ctx, batch) == 0 && common_speculative_process(spec, batch);
        if (!ok) {
            break;
        }
        const auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx, draft);
        // the last cycle is reported as the server reports a re-check after a checkpoint restore, with its whole
        // draft kept: the fixture's random apprentice keeps nothing, and a replay of 0 reads 0 with or without the
        // mirror; being last, nothing after it reads the apprentice's state
        const bool replay = n_cycles == 11 && !draft.empty();
        const int  n_report = replay ? (int) draft.size() : (int) ids.size() - 1;
        common_speculative_accept(spec, 0, (uint16_t) n_report, replay);
        if (replay) {
            n_replay     = n_report;
            replay_cycle = n_cycles;
        }
        n_past += (int) ids.size() - 1;
        for (llama_token id : ids) {
            hist.push_back(id_last);
            id_last = id;
        }
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n_past, -1);
    }
    llama_batch_free(batch);
    common_speculative_free(spec);
    unsetenv("LLAMA_SPEC_RATE_TRACE");

    const auto l = read_lines(path);
    check(ok && n_cycles == 12, "12 cycles ran");
    check(count_with(l, "{\"trace\":") == 1 && count_with(l, "{\"end\":") == 1 && count_with(l, "{\"answer\":") == 1,
          "one header, one answer line, one closing line: one writer");
    check(!l.empty() && has(l[0], "\"draft\":\"draft-mtp\",\"block\":0,"), "the header is MTP's own");
    check(count_with(l, "{\"cycle\":") == 12, "one line a cycle (" + std::to_string(count_with(l, "{\"cycle\":")) + ")");
    check(n_replay > 0, "the replayed cycle drafted, so the replay has a word to take back");
    const std::string cycle = "{\"cycle\":" + std::to_string(replay_cycle) + ",";
    const std::string kept  = "\"kept\":" + std::to_string(n_replay - 1) + ",";
    bool mirrored = false;
    for (const auto & line : l) {
        mirrored = mirrored || (has(line, cycle) && has(line, "\"replay\":1,") && has(line, kept));
    }
    check(mirrored && count_with(l, "\"replay\":1,") == 1, "MTP's own hook mirrors the replay (cycle " +
          std::to_string(replay_cycle) + ", " + kept + " of " + std::to_string(n_replay) + ")");
    std::remove(path.c_str());
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string model_path;
    int ngl = 99;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "-ngl" && i + 1 < argc) {
            ngl = std::atoi(argv[++i]);
        }
    }
    common_init();
    printf("ngram-simple through the wrapper\n");
    test_ngram(false);
    test_ngram(true);
    if (!model_path.empty()) {
        llama_backend_init();
        printf("draft-mtp through the wrapper, ngl %d\n", ngl);
        test_mtp_guard(model_path, ngl);
        llama_backend_free();
    }
    printf("%s\n", g_failures ? "test-speculative-trace-wrapper: FAILED" : "test-speculative-trace-wrapper: OK");
    return g_failures ? 1 : 0;
}

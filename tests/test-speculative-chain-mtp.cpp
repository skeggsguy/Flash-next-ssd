// The chain stop (LLAMA_SPEC_CHAIN_P, common/speculative-chain.h, CHAIN-PLAN step 2) on the MTP fixture
// (qwen4exp-moe-e64mtp.gguf), driven as examples/speculative-simple drives speculation, the library greedy, with the
// apprentice's own rate trace on for the confidences it drafted on:
//   threshold  the product of the first cycle's first two confidences at p-min 0 (chain off), so the first cycle
//              sits exactly on the edge: with the chain on it keeps both words and drops the third
//   cut        every cycle with the chain on: the product of its drafted words' confidences is at or above the
//              threshold, and a cycle that stopped short ("chain") has the next word's product under it; each
//              cycle's draft is the first words of what fixed 8 drafts from the same place (a second chain-off run,
//              its drafts cut to the chain's lengths, walks the same path); the text is the same with the chain on
//              or off; and some cycles stopped on the chain (not vacuous)
//   rate2      LLAMA_SPEC_ADAPTIVE_RATE=2 ignores the switch: checking every step it drafts as fixed 8 does
// The cut's own arithmetic is test-speculative-chain (no model), the DFlash 2 walk's test-speculative-dflash-walk.

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "speculative-rate2.h" // the tests-only "check every step"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

std::string g_path;
int         g_ngl      = 99;
int         g_failures = 0;

constexpr int N_MAX     = 8;
constexpr int N_PREDICT = 160;

void check(bool ok, const std::string & what) {
    printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

struct cycle_rec {
    llama_tokens full;  // what the drafter handed back
    llama_tokens draft; // what the library checked (cut, in the cut run)
    int          accepted = 0;
};

// one line of the apprentice's own trace
struct traced {
    std::vector<float> conf;
    std::string        stop;
    int                drafted = 0;
};

struct run_out {
    bool                   ok = false;
    std::vector<cycle_rec> cycles;
    std::vector<traced>    trace;
    llama_tokens           text;
};

std::string tmp_path(const std::string & tag) {
    const char * dir = std::getenv("TMPDIR");
    return std::string(dir && *dir ? dir : "/tmp") + "/test-spec-chain-mtp-" + tag + "-" +
           std::to_string((long) getpid()) + ".srtr";
}

std::string field(const std::string & line, const std::string & key) {
    const std::string k = "\"" + key + "\":";
    const size_t at = line.find(k);
    if (at == std::string::npos) {
        return "";
    }
    size_t beg = at + k.size(), end = beg;
    if (line[beg] == '[') {
        end = line.find(']', beg);
        return line.substr(beg + 1, end - beg - 1);
    }
    if (line[beg] == '"') {
        end = line.find('"', beg + 1);
        return line.substr(beg + 1, end - beg - 1);
    }
    end = line.find_first_of(",}", beg);
    return line.substr(beg, end - beg);
}

std::vector<traced> read_trace(const std::string & path) {
    std::vector<traced> out;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        if (line.rfind("{\"cycle\":", 0) != 0) {
            continue;
        }
        traced t;
        t.stop    = field(line, "stop");
        t.drafted = std::atoi(field(line, "drafted").c_str());
        const std::string conf = field(line, "conf");
        for (size_t at = 0; at < conf.size();) {
            const size_t comma = conf.find(',', at);
            t.conf.push_back(std::strtof(conf.substr(at, comma - at).c_str(), nullptr)); // %.9g: the float exactly
            at = comma == std::string::npos ? conf.size() : comma + 1;
        }
        out.push_back(t);
    }
    return out;
}

double product(const std::vector<float> & conf, size_t n) {
    double prod = 1.0;
    for (size_t i = 0; i < n && i < conf.size(); i++) {
        prod *= (double) conf[i];
    }
    return prod;
}

void set_env(const char * name, const char * value) {
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
}

// one greedy generation from `prompt`; `cut` (optional) shortens each cycle's draft to that cycle's length
run_out run(common_speculative_type type, const char * rate, const char * chain, bool check_all,
        const llama_tokens & prompt, const std::vector<size_t> * cut, const std::string & trace_tag) {
    run_out out;
    set_env("LLAMA_SPEC_ADAPTIVE_RATE", rate);
    set_env("LLAMA_SPEC_CHAIN_P", chain);
    const std::string trace_path = tmp_path(trace_tag);
    std::remove(trace_path.c_str());
    set_env("LLAMA_SPEC_RATE_TRACE", trace_tag.empty() ? nullptr : trace_path.c_str());
    common_speculative_rate2_check_all_for_tests(check_all);

    common_params params;
    params.model.path   = g_path;
    params.n_gpu_layers = g_ngl;
    params.fit_params   = false;
    postprocess_cpu_params(params.cpuparams, nullptr); // what the argument parser does
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    params.warmup        = false;
    params.n_ctx         = 512;
    params.n_batch       = 512;
    params.n_ubatch      = 64;
    params.sampling.temp = 0.0f; // greedy: the text cannot depend on the drafts
    params.speculative.types       = { type };
    params.speculative.draft.n_max = N_MAX;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0.0f;

    const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel,
            common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (model == nullptr || ctx == nullptr) {
        return out;
    }
    common_params params_dft = common_base_params_to_speculative(params);
    auto spec_init = common_speculative_init_from_params(params_dft, model, ctx);
    params.speculative.draft.ctx_tgt = ctx;
    params.speculative.draft.ctx_dft = spec_init->context();
    llama_context * ctx_dft = params.speculative.draft.ctx_dft;
    if (ctx_dft == nullptr || common_context_can_seq_rm(ctx) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        fprintf(stderr, "no MTP context, or the library needs checkpoints (this test does not drive them)\n");
        return out;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    common_sampler_ptr smpl(common_sampler_init(model, params.sampling));
    common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (spec == nullptr) {
        return out;
    }

    const llama_seq_id seq = 0;
    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);
    for (size_t i = 0; i + 1 < prompt.size(); i++) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { seq }, false);
    }
    bool ok = llama_decode(ctx, batch) == 0 && common_speculative_process(spec, batch);

    llama_token  id_last = prompt.back();
    llama_tokens prompt_tgt(prompt.begin(), prompt.end() - 1);
    int          n_past  = (int) prompt.size() - 1;
    common_speculative_begin(spec, seq, prompt_tgt);

    llama_tokens draft;
    while (ok && (int) out.text.size() < N_PREDICT) {
        const int n_draft_max = std::max(0, std::min((int) llama_n_ctx(ctx) - n_past - 2,
                N_PREDICT - (int) out.text.size() - 1));
        common_speculative_get_draft_params(spec, seq) = { true, n_draft_max, n_past, id_last, &prompt_tgt, &draft };
        common_speculative_draft(spec);
        cycle_rec rec;
        rec.full = draft;
        if (cut != nullptr && out.cycles.size() < cut->size() && draft.size() > (*cut)[out.cycles.size()]) {
            draft.resize((*cut)[out.cycles.size()]);
        }
        rec.draft = draft;
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, n_past, -1); // the drafter's guesses go

        common_batch_clear(batch);
        common_batch_add(batch, id_last, n_past++, { seq }, true);
        for (size_t i = 0; i < draft.size(); i++) {
            common_batch_add(batch, draft[i], n_past + (llama_pos) i, { seq }, true);
        }
        ok = llama_decode(ctx, batch) == 0 && common_speculative_process(spec, batch);
        if (!ok) {
            break;
        }
        const auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx, draft);
        common_speculative_accept(spec, seq, (uint16_t) (ids.size() - 1));
        rec.accepted = (int) ids.size() - 1;
        out.cycles.push_back(rec);
        n_past += (int) ids.size() - 1;

        bool eog = false;
        for (llama_token id : ids) {
            prompt_tgt.push_back(id_last);
            id_last = id;
            out.text.push_back(id);
            if ((eog = llama_vocab_is_eog(vocab, id))) {
                break;
            }
        }
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(ctx), seq, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, n_past, -1);
        if (eog) {
            break;
        }
    }
    out.ok = ok;
    llama_batch_free(batch);
    common_speculative_free(spec); // closes the trace
    common_speculative_rate2_check_all_for_tests(false);
    unsetenv("LLAMA_SPEC_ADAPTIVE_RATE");
    unsetenv("LLAMA_SPEC_CHAIN_P");
    unsetenv("LLAMA_SPEC_RATE_TRACE");
    if (!trace_tag.empty()) {
        out.trace = read_trace(trace_path);
        std::remove(trace_path.c_str());
    }
    return out;
}

bool same_cycles(const run_out & a, const run_out & b) {
    if (a.cycles.size() != b.cycles.size()) {
        return false;
    }
    for (size_t i = 0; i < a.cycles.size(); i++) {
        if (a.cycles[i].draft != b.cycles[i].draft || a.cycles[i].accepted != b.cycles[i].accepted) {
            return false;
        }
    }
    return true;
}

void test_prompt(const llama_tokens & prompt) {
    printf("prompt of %zu tokens, ngl %d\n", prompt.size(), g_ngl);
    const run_out off = run(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, nullptr, nullptr, false, prompt, nullptr, "off");
    check(off.ok && off.cycles.size() > 10 && off.trace.size() == off.cycles.size() && off.trace[0].conf.size() >= 3,
          "the chain off ran, one trace line a cycle (" + std::to_string(off.cycles.size()) + " cycles)");
    // the switch unset: the MTP head keeps the chain off (only draft-dflash defaults it on, flashnext.h)
    check(std::none_of(off.trace.begin(), off.trace.end(), [](const traced & l) { return l.stop == "chain"; }),
          "LLAMA_SPEC_CHAIN_P unset: no cycle stopped on the chain (the MTP head's default is off)");
    if (off.trace.empty() || off.trace[0].conf.size() < 3) {
        return;
    }

    // the edge: the first cycle's two-word product exactly, as the shortest text that reads back as that double
    const double threshold = product(off.trace[0].conf, 2);
    char text[32];
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(text, sizeof(text), "%.*g", prec, threshold);
        if (std::strtod(text, nullptr) == threshold) {
            break;
        }
    }
    printf("  threshold %s (cycle 0's confidences %.4f %.4f %.4f)\n", text, (double) off.trace[0].conf[0],
           (double) off.trace[0].conf[1], (double) off.trace[0].conf[2]);

    const run_out on = run(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, nullptr, text, false, prompt, nullptr, "on");
    check(on.ok && on.trace.size() == on.cycles.size() && !on.cycles.empty(), "the chain on ran, one trace line a cycle");
    if (on.cycles.empty() || on.trace.size() != on.cycles.size()) {
        return;
    }
    check(on.cycles[0].draft.size() == 2 && on.trace[0].stop == "chain",
          "cycle 0 sits on the edge: its two words' product equals the threshold, kept, and the third dropped (" +
          std::to_string(on.cycles[0].draft.size()) + " drafted, stop " + on.trace[0].stop + ")");

    bool held = true, fell = true, named = true;
    int  chained = 0, deeper = 0;
    std::vector<int> at_depth(N_MAX + 1, 0);
    std::vector<size_t> lengths;
    for (size_t i = 0; i < on.cycles.size(); i++) {
        const auto & c = on.cycles[i];
        const auto & t = on.trace[i];
        const size_t d = c.draft.size();
        lengths.push_back(d);
        at_depth[std::min<size_t>(d, N_MAX)]++;
        held = held && (int) d == t.drafted && product(t.conf, d) >= threshold;
        if (t.stop == "chain") {
            fell = fell && t.conf.size() == d + 1 && product(t.conf, d + 1) < threshold;
            chained++;
            deeper += d > 0;
        } else {
            named = named && t.stop == "cap" && t.conf.size() == d; // p-min 0: the ceiling is the only other stop
        }
    }
    check(held, "every checked draft's product is at or above the threshold");
    check(fell, "every chain stop's next word takes the product under it");
    check(named, "every other cycle stopped at the ceiling");
    std::string depths;
    int n_depths = 0;
    for (int d = 0; d <= N_MAX; d++) {
        if (at_depth[d] > 0) {
            depths += (depths.empty() ? "" : ", ") + std::to_string(at_depth[d]) + " at " + std::to_string(d);
            n_depths++;
        }
    }
    check(chained > 0 && deeper > 0 && n_depths > 1, "the chain stopped " + std::to_string(chained) + " of " +
          std::to_string(on.cycles.size()) + " cycles, " + std::to_string(deeper) + " after a word; drafts " + depths +
          " (not vacuous)");

    const run_out cut = run(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, nullptr, nullptr, false, prompt, &lengths, "");
    bool prefix = cut.ok && cut.cycles.size() == on.cycles.size();
    for (size_t i = 0; prefix && i < on.cycles.size(); i++) {
        const auto & p = on.cycles[i].draft;
        const auto & f = cut.cycles[i].full;
        prefix = p.size() <= f.size() && std::equal(p.begin(), p.end(), f.begin()) && cut.cycles[i].draft == p &&
                 cut.cycles[i].accepted == on.cycles[i].accepted;
    }
    check(prefix, "each cycle's draft is fixed 8's first words from the same place");
    check(on.text == off.text && on.text == cut.text, "the same text with the chain on and off");

    const run_out all       = run(COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, "2", nullptr, true, prompt, nullptr, "");
    const run_out all_chain = run(COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, "2", text, true, prompt, nullptr, "");
    check(all.ok && all_chain.ok && same_cycles(all, all_chain) && same_cycles(all, off),
          "RATE=2 ignores the chain: checking every step, it drafts as fixed 8 with the switch set");
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) {
            g_path = argv[++i];
        } else if (a == "-ngl" && i + 1 < argc) {
            g_ngl = std::atoi(argv[++i]);
        }
    }
    if (g_path.empty()) {
        fprintf(stderr, "usage: %s -m qwen4exp-moe-e64mtp.gguf [-ngl N]\n", argv[0]);
        return 1;
    }
    common_init();
    llama_backend_init();

    std::mt19937 rng(42);
    for (int n : {37, 120}) {
        llama_tokens prompt;
        for (int i = 0; i < n; i++) {
            prompt.push_back((llama_token) (1 + rng() % 100)); // the fixture has 128 words
        }
        test_prompt(prompt);
    }

    llama_backend_free();
    printf("%s\n", g_failures ? "test-speculative-chain-mtp: FAILED" : "test-speculative-chain-mtp: OK");
    return g_failures ? 1 : 0;
}

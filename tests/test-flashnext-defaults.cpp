// The fork's proven defaults (the Flash-Next study's standard roll-out, 2026-10-15): src/llama-flashnext.h and
// common/flashnext.h. No model; ggml-metal's two are pinned by test-flashnext-metal.
//
// - the switches: each one unset is the everyday value, "0" (or the old value) the old way, and the guesser's tuning
//   takes its everyday values only with the guesser's file named;
// - LLAMA_SPEC_ADAPTIVE_RATE: unset is the measured depth, "" and "0" the acceptance controller, "2" the priced depth;
// - LLAMA_SPEC_CHAIN_P: unset, "" and "0" off, a number in (0, 1] the threshold (as a double), anything else throws;
// - the start-up line: its first word, the defaults in force and the user's values marked, the preset on and off;
// - `--flashnext` through common_params_parse, as llama-server parses: the everyday values where the user passed
//   nothing; a flag passed on the command line (in any spelling, before or after it), as its LLAMA_ARG_* or in a
//   config file is left as given; the draft settings only with a draft model; and without it upstream's defaults.

#include "testing.h"

#include "arg.h"
#include "common.h"
#include "flashnext.h"

#include "../src/llama-flashnext.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

void test_switches(testing & t) {
    t.assert_true("LRU unset: on", llama_flashnext_lru(nullptr));
    t.assert_true("LRU \"1\": on", llama_flashnext_lru("1"));
    t.assert_true("LRU \"0\": hotness", !llama_flashnext_lru("0"));
    t.assert_true("LRU \"\": hotness", !llama_flashnext_lru(""));
    t.assert_true("LRU a word: hotness", !llama_flashnext_lru("x"));

    t.assert_true("RS skip unset: on", llama_flashnext_rs_skip_slots(nullptr));
    t.assert_true("RS skip \"1\": on", llama_flashnext_rs_skip_slots("1"));
    t.assert_true("RS skip \"0\": off", !llama_flashnext_rs_skip_slots("0"));
    t.assert_true("RS skip \"\": off", !llama_flashnext_rs_skip_slots(""));
    t.assert_true("RS skip \"2\": off", !llama_flashnext_rs_skip_slots("2"));

    t.assert_equal("chunk unset: 4096", (int64_t) 4096, llama_flashnext_alloc_chunk_mib(nullptr));
    t.assert_equal("chunk \"0\": one go", (int64_t) 0, llama_flashnext_alloc_chunk_mib("0"));
    t.assert_equal("chunk \"-3\": one go", (int64_t) 0, llama_flashnext_alloc_chunk_mib("-3"));
    t.assert_equal("chunk \"1024\"", (int64_t) 1024, llama_flashnext_alloc_chunk_mib("1024"));

    t.assert_true("desk first unset: on", llama_flashnext_desk_first(nullptr));
    t.assert_true("desk first \"1\": on", llama_flashnext_desk_first("1"));
    t.assert_true("desk first \"0\": the stop waits for every trip", !llama_flashnext_desk_first("0"));
    t.assert_true("desk first \"\": off", !llama_flashnext_desk_first(""));

    t.assert_true("guesser unset: not named", !llama_flashnext_guesser_named(nullptr));
    t.assert_true("guesser \"\": not named", !llama_flashnext_guesser_named(""));
    t.assert_true("guesser a path: named", llama_flashnext_guesser_named("/x/guesser.bin"));
}

void test_guesser_tuning(testing & t) {
    // with the guesser named: K-final's values unset
    t.assert_true("ALL unset, guesser: on", llama_flashnext_lookahead_all(nullptr, true));
    t.assert_equal("DEPTH2 unset, guesser: 4", 4u, llama_flashnext_lookahead_depth2(nullptr, true));
    t.assert_true("KEEP unset, guesser: on", llama_flashnext_lookahead_keep(nullptr, true));
    t.assert_equal("CONF unset, guesser: 0.2", 0.2f, llama_flashnext_lookahead_conf(nullptr, true));
    // without it: the old way, which is all the study proved there
    t.assert_true("ALL unset, no guesser: off", !llama_flashnext_lookahead_all(nullptr, false));
    t.assert_equal("DEPTH2 unset, no guesser: 0", 0u, llama_flashnext_lookahead_depth2(nullptr, false));
    t.assert_true("KEEP unset, no guesser: off", !llama_flashnext_lookahead_keep(nullptr, false));
    t.assert_equal("CONF unset, no guesser: 0", 0.0f, llama_flashnext_lookahead_conf(nullptr, false));
    // "0" restores the old way beside the guesser; a value given is the value, guesser or not
    t.assert_true("ALL \"0\", guesser: off", !llama_flashnext_lookahead_all("0", true));
    t.assert_true("ALL \"\", guesser: off", !llama_flashnext_lookahead_all("", true));
    t.assert_true("ALL \"1\", no guesser: on", llama_flashnext_lookahead_all("1", false));
    t.assert_equal("DEPTH2 \"0\", guesser: 0", 0u, llama_flashnext_lookahead_depth2("0", true));
    t.assert_equal("DEPTH2 \"6\", no guesser: 6", 6u, llama_flashnext_lookahead_depth2("6", false));
    t.assert_equal("DEPTH2 \"-2\": 0", 0u, llama_flashnext_lookahead_depth2("-2", true));
    t.assert_true("KEEP \"0\", guesser: off", !llama_flashnext_lookahead_keep("0", true));
    t.assert_true("KEEP \"1\", no guesser: on", llama_flashnext_lookahead_keep("1", false));
    t.assert_equal("CONF \"0\", guesser: the fixed pick", 0.0f, llama_flashnext_lookahead_conf("0", true));
    t.assert_equal("CONF \"\", guesser: the fixed pick", 0.0f, llama_flashnext_lookahead_conf("", true));
    t.assert_equal("CONF \"0.5\"", 0.5f, llama_flashnext_lookahead_conf("0.5", true));
    for (const char * bad : { "abc", "1.5", "-0.1", "0.2x" }) {
        bool threw = false;
        try {
            llama_flashnext_lookahead_conf(bad, true);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        t.assert_true(std::string("CONF \"") + bad + "\" throws", threw);
    }
}

void test_adaptive_rate(testing & t) {
    t.assert_equal("unset: the measured depth", 1, common_flashnext_adaptive_rate(nullptr));
    t.assert_equal("\"1\": the measured depth", 1, common_flashnext_adaptive_rate("1"));
    t.assert_equal("\"0\": the acceptance controller", 0, common_flashnext_adaptive_rate("0"));
    t.assert_equal("\"\": the acceptance controller", 0, common_flashnext_adaptive_rate(""));
    t.assert_equal("\"2\": the priced depth", 2, common_flashnext_adaptive_rate("2"));
    t.assert_equal("a word: the measured depth", 1, common_flashnext_adaptive_rate("yes"));
}

void test_chain_p(testing & t) {
    t.assert_equal("unset: off", 0.0, common_flashnext_chain_p(nullptr));
    t.assert_equal("\"\": off", 0.0, common_flashnext_chain_p(""));
    t.assert_equal("\"0\": off", 0.0, common_flashnext_chain_p("0"));
    t.assert_equal("\"0.3\": the double 0.3, not the float", 0.3, common_flashnext_chain_p("0.3"));
    t.assert_equal("\"1\": the top of the range", 1.0, common_flashnext_chain_p("1"));
    t.assert_equal("\"1e-3\"", 1e-3, common_flashnext_chain_p("1e-3"));
    for (const char * bad : { "abc", "1.5", "1.0000001", "-0.1", "0.2x", "0.0", "nan", "off", "on" }) {
        bool threw = false;
        try {
            common_flashnext_chain_p(bad);
        } catch (const std::runtime_error & e) {
            threw = std::string(e.what()).find("LLAMA_SPEC_CHAIN_P") != std::string::npos;
        }
        t.assert_true(std::string("CHAIN_P \"") + bad + "\" throws, naming the switch", threw);
    }
}

// a fake environment for the start-up line
struct fake_env {
    std::map<std::string, std::string> vars;
    const char * operator()(const char * name) const {
        auto it = vars.find(name);
        return it == vars.end() ? nullptr : it->second.c_str();
    }
};

const std::string ENGINE_DEFAULTS = "flashnext: engine defaults: LLAMA_MOE_STREAM_LRU=1, LLAMA_RS_SKIP_SLOTS=1, "
    "LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB=4096, LLAMA_MOE_STREAM_DESK_FIRST=1, LLAMA_SPEC_ADAPTIVE_RATE=1, "
    "LLAMA_SPEC_CHAIN_P=off, GGML_METAL_STAFF_MV=0, GGML_METAL_RESIDENCY_KEEP_ALIVE_S=10000000";

void test_line(testing & t) {
    common_params params;
    fake_env env;
    auto line = [&]() { return common_flashnext_load_line(params, std::cref(env)); };

    t.assert_equal("unset, no preset", ENGINE_DEFAULTS +
        "; the guesser's tuning: off, no guesser named; preset --flashnext: off", line());

    env.vars["LLAMA_MOE_STREAM_LRU"] = "0";
    env.vars["GGML_METAL_STAFF_MV"]  = "1";
    env.vars["LLAMA_MOE_STREAM_DESK_FIRST"] = "0";
    env.vars["LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2"] = "4";
    env.vars["LLAMA_SPEC_CHAIN_P"] = "0.3";
    const std::string l1 = line();
    t.assert_true("the user's chain stop marked", l1.find(", LLAMA_SPEC_CHAIN_P=0.3 (yours), ") != std::string::npos);
    t.assert_true("the user's LRU marked", l1.find("LLAMA_MOE_STREAM_LRU=0 (yours)") != std::string::npos);
    t.assert_true("the user's staff marked", l1.find("GGML_METAL_STAFF_MV=1 (yours)") != std::string::npos);
    t.assert_true("the user's desk first marked", l1.find("LLAMA_MOE_STREAM_DESK_FIRST=0 (yours)") != std::string::npos);
    t.assert_true("a tuning set without the guesser",
        l1.find("off, no guesser named, but LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2=4 (yours);") != std::string::npos);

    env.vars.clear();
    env.vars["LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER"] = "/m/guesser.bin";
    env.vars["LLAMA_MOE_STREAM_LOOKAHEAD_KEEP"]    = "0";
    t.assert_equal("the guesser named", ENGINE_DEFAULTS + "; the guesser's tuning: "
        "LLAMA_MOE_STREAM_LOOKAHEAD_ALL=1, LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2=4, "
        "LLAMA_MOE_STREAM_LOOKAHEAD_KEEP=0 (yours), LLAMA_MOE_STREAM_LOOKAHEAD_CONF=0.2; preset --flashnext: off",
        line());

    env.vars.clear();
    params.flashnext       = true;
    params.flashnext_set   = { "-ngl 99", "--moe-stream" };
    params.flashnext_yours = { "-c" };
    const std::string l2 = line();
    t.assert_true("the preset on, what it set and what it kept", l2.find(
        "; preset --flashnext: on, set -ngl 99 --moe-stream, yours kept: -c, the draft settings wait for -md")
        != std::string::npos);
    params.speculative.draft.mparams.path = "/m/draft.gguf";
    t.assert_true("with a draft model nothing waits", line().find("wait for -md") == std::string::npos);

    // the study's log readers take a line by its first words (runner/roomstats.py): this one is none of theirs
    for (const char * word : { "moe stream:", "reading room", "lent belt", "drives busy", "undo copies:",
                               "staff mat-vec", "put-back rule", "draft depth:" }) {
        t.assert_true(std::string("no \"") + word + "\"", l2.find(word) == std::string::npos);
    }
    t.assert_true("the line reads the real environment", common_flashnext_load_line(params).rfind("flashnext: ", 0) == 0);
}

// common_params_parse as llama-server does it, from argv, the environment and $XDG_CONFIG_HOME's config.ini
common_params parse(testing & t, std::vector<std::string> args, llama_example ex = LLAMA_EXAMPLE_SERVER) {
    args.insert(args.begin(), "llama-server");
    std::vector<char *> argv;
    for (auto & a : args) {
        argv.push_back(a.data());
    }
    common_params params;
    t.assert_true("parses", common_params_parse((int) argv.size(), argv.data(), params, ex));
    return params;
}

bool has(const std::vector<std::string> & v, const std::string & s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

void test_preset(testing & t) {
    {
        // upstream's defaults, untouched
        const common_params p = parse(t, {});
        t.assert_true("off", !p.flashnext);
        t.assert_equal("n_ctx", 0, p.n_ctx);
        t.assert_equal("n_batch", 2048, p.n_batch);
        t.assert_equal("n_ubatch", 512, p.n_ubatch);
        t.assert_equal("n_gpu_layers", -1, p.n_gpu_layers);
        t.assert_equal("n_parallel: auto", -1, p.n_parallel);
        t.assert_equal("n_ctx_checkpoints", 32, p.n_ctx_checkpoints);
        t.assert_equal("cache_ram_mib", 8192, p.cache_ram_mib);
        t.assert_true("flash attention auto", p.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO);
        t.assert_true("no moe stream", !p.moe_stream && !p.moe_stream_direct);
        t.assert_equal("io threads", 0, p.moe_stream_io_threads);
        t.assert_true("nothing set", p.flashnext_set.empty() && p.flashnext_yours.empty());
    }
    {
        const common_params p = parse(t, { "--flashnext" });
        t.assert_true("on", p.flashnext);
        t.assert_equal("-ngl 99", 99, p.n_gpu_layers);
        t.assert_true("--moe-stream", p.moe_stream);
        t.assert_true("--moe-stream-direct", p.moe_stream_direct);
        t.assert_equal("--moe-stream-io-threads 8", 8, p.moe_stream_io_threads);
        t.assert_equal("-c 200000", 200000, p.n_ctx);
        t.assert_equal("-b 4096", 4096, p.n_batch);
        t.assert_equal("-ub 4096", 4096, p.n_ubatch);
        t.assert_equal("-cms 8192", 8192, p.checkpoint_min_step);
        t.assert_equal("-ctxcp 3", 3, p.n_ctx_checkpoints);
        t.assert_equal("-np 1", 1, p.n_parallel);
        t.assert_true("-fa on", p.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED);
        t.assert_equal("--cache-reuse 0", 0, p.n_cache_reuse);
        t.assert_equal("--cache-ram 400", 400, p.cache_ram_mib);
        t.assert_true("--cache-ram-opening", p.cache_ram_opening);
        t.assert_equal("--chat-window 90", 90, p.chat_window);
        t.assert_true("--jinja", p.use_jinja);
        t.assert_true("--reasoning-format deepseek", p.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK);
        t.assert_true("no draft settings without a draft model",
            p.speculative.types == std::vector<common_speculative_type>{ COMMON_SPECULATIVE_TYPE_NONE });
        t.assert_true("n_max upstream's", p.speculative.draft.n_max == common_params().speculative.draft.n_max);
        t.assert_equal("everything else set", (size_t) 17, p.flashnext_set.size());
        t.assert_true("-c 200000 recorded", has(p.flashnext_set, "-c 200000"));
        t.assert_true("--moe-stream recorded", has(p.flashnext_set, "--moe-stream"));
        t.assert_true("nothing yours", p.flashnext_yours.empty());
    }
    {
        // the user's flags, any spelling, before or after --flashnext
        const common_params p = parse(t, { "-c", "98304", "--flashnext", "--batch-size", "2048", "-fa", "off",
                                           "--no-jinja", "--cache-ram", "400", "--cache-ram-opening" });
        t.assert_equal("-c kept", 98304, p.n_ctx);
        t.assert_equal("--batch-size kept", 2048, p.n_batch);
        t.assert_equal("-ub set", 4096, p.n_ubatch);
        t.assert_true("-fa off kept", p.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED);
        t.assert_true("--no-jinja kept", !p.use_jinja);
        t.assert_equal("--cache-ram 400 kept", 400, p.cache_ram_mib);
        t.assert_true("the opening copy kept", p.cache_ram_opening);
        for (const char * a : { "-c", "-b", "-fa", "--jinja", "--cache-ram", "--cache-ram-opening" }) {
            t.assert_true(std::string("yours: ") + a, has(p.flashnext_yours, a));
        }
        t.assert_true("not recorded as set", !has(p.flashnext_set, "-c 200000"));
    }
    {
        const common_params p = parse(t, { "--flashnext", "-md", "/m/draft.gguf" });
        t.assert_true("--spec-type draft-mtp-adaptive",
            p.speculative.types == std::vector<common_speculative_type>{ COMMON_SPECULATIVE_TYPE_NONE,
                                                                         COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE });
        t.assert_equal("--spec-draft-n-max 5", 5, p.speculative.draft.n_max);
        t.assert_equal("--spec-draft-p-min 0.3", 0.3f, p.speculative.draft.p_min);
        t.assert_equal("--spec-draft-ngl 99", 99, p.speculative.draft.n_gpu_layers);
        t.assert_equal("--spec-max-prompt 0", 0, (int) p.speculative.n_prompt_max);
        t.assert_equal("all twenty-two set", (size_t) 22, p.flashnext_set.size());
    }
    {
        // --spec-type adds to the list: the user's type alone, not the preset's beside it
        const common_params p = parse(t, { "--flashnext", "-md", "/m/draft.gguf", "--spec-type", "draft-mtp",
                                           "--spec-draft-n-max", "3" });
        t.assert_true("the user's type alone",
            p.speculative.types == std::vector<common_speculative_type>{ COMMON_SPECULATIVE_TYPE_NONE,
                                                                         COMMON_SPECULATIVE_TYPE_DRAFT_MTP });
        t.assert_equal("the user's n-max", 3, p.speculative.draft.n_max);
        t.assert_equal("p-min set", 0.3f, p.speculative.draft.p_min);
    }
    {
        // a flag set as its LLAMA_ARG_*, and the preset itself from the environment
        setenv("LLAMA_ARG_CTX_SIZE", "65536", 1);
        setenv("LLAMA_ARG_FLASHNEXT", "1", 1);
        const common_params p = parse(t, {});
        unsetenv("LLAMA_ARG_CTX_SIZE");
        unsetenv("LLAMA_ARG_FLASHNEXT");
        t.assert_true("LLAMA_ARG_FLASHNEXT=1: on", p.flashnext);
        t.assert_equal("LLAMA_ARG_CTX_SIZE kept", 65536, p.n_ctx);
        t.assert_equal("-b set", 4096, p.n_batch);
        t.assert_true("yours: -c", has(p.flashnext_yours, "-c"));
    }
    {
        // a flag set in the user's config file
        const char * home = getenv("XDG_CONFIG_HOME");
        const std::string dir = std::string(home) + "/llama.cpp";
        std::filesystem::create_directories(dir);
        // its two sections that apply to every program: the default (no header) and "*"
        std::ofstream(dir + "/config.ini") << "ubatch-size = 1024\n[*]\nctx-checkpoints = 5\n";
        const common_params p = parse(t, { "--flashnext" });
        std::filesystem::remove(dir + "/config.ini");
        t.assert_equal("config's -ub kept", 1024, p.n_ubatch);
        t.assert_equal("config's [*] -ctxcp kept", 5, p.n_ctx_checkpoints);
        t.assert_equal("-b set", 4096, p.n_batch);
        t.assert_true("yours: -ub", has(p.flashnext_yours, "-ub"));
        t.assert_true("yours: -ctxcp", has(p.flashnext_yours, "-ctxcp"));
    }
    {
        // a program without the server's flags (one that needs no model to parse): those are skipped, the rest set
        const common_params p = parse(t, { "--flashnext" }, LLAMA_EXAMPLE_EXPORT_GRAPH_OPS);
        t.assert_equal("-c set", 200000, p.n_ctx);
        t.assert_equal("--cache-ram skipped: upstream's", 8192, p.cache_ram_mib);
        t.assert_true("--cache-ram not recorded", !has(p.flashnext_set, "--cache-ram 400"));
    }
}

} // namespace

int main(int argc, char ** argv) {
    testing t(std::cout);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    // a config directory of our own: the user's config.ini must not move these results
    const auto dir = std::filesystem::temp_directory_path() / ("flashnext-defaults-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    setenv("XDG_CONFIG_HOME", dir.c_str(), 1);
    // the switches the line names, unset as in a bare start
    for (const char * e : { "LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_FLASHNEXT", "LLAMA_ARG_BATCH", "LLAMA_ARG_UBATCH" }) {
        unsetenv(e);
    }

    t.test("the switches", test_switches);
    t.test("the guesser's tuning", test_guesser_tuning);
    t.test("the draft depth", test_adaptive_rate);
    t.test("the chain stop", test_chain_p);
    t.test("the start-up line", test_line);
    t.test("the preset", test_preset);

    std::filesystem::remove_all(dir);
    return t.summary();
}

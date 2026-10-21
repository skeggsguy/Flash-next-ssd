// DFlash's defaults (C37's winner, 2026-10-21: +10.8% chat, +8.7% code): common/flashnext.h. No model.
//
// - LLAMA_SPEC_CHAIN_P unset is 0.5 for draft-dflash and off for every other draft (the MTP head, draft-simple,
//   EAGLE3, DSpark); "" and "0" turn it off for DFlash too, and a value given always wins;
// - `--flashnext` with --spec-type draft-dflash: depth 7 and p-min 0 in place of the MTP head's 5 and 0.3, each only
//   where the user set nothing (the command line, LLAMA_ARG_*); the MTP head keeps 5 and 0.3; without the preset
//   upstream's defaults stay (the engine leaves llama.cpp's own options alone);
// - the start-up line: the chain's default in force for the draft type, the user's value marked "(yours)".
// The chain's wiring on a real draft (the MTP head unset: off) is test-speculative-chain-mtp.

#include "testing.h"

#include "arg.h"
#include "common.h"
#include "flashnext.h"
#include "speculative.h" // common_speculative_type_to_str

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

constexpr common_speculative_type DFLASH = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

void test_chain_default(testing & t) {
    t.assert_equal("draft-dflash unset: 0.5", 0.5, common_flashnext_chain_p(nullptr, DFLASH));
    t.assert_equal("the default is the constant", COMMON_FLASHNEXT_CHAIN_P_DFLASH, common_flashnext_chain_p(nullptr, DFLASH));
    for (auto type : { COMMON_SPECULATIVE_TYPE_NONE, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE,
                       COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
                       COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK }) {
        t.assert_equal(common_speculative_type_to_str(type) + " unset: off", 0.0, common_flashnext_chain_p(nullptr, type));
    }
    t.assert_equal("draft-dflash \"0\": off", 0.0, common_flashnext_chain_p("0", DFLASH));
    t.assert_equal("draft-dflash \"\": off", 0.0, common_flashnext_chain_p("", DFLASH));
    t.assert_equal("draft-dflash \"0.7\": the user's", 0.7, common_flashnext_chain_p("0.7", DFLASH));
    t.assert_equal("draft-dflash \"1\": the user's", 1.0, common_flashnext_chain_p("1", DFLASH));
    t.assert_equal("the MTP head \"0.5\": the user's", 0.5,
                   common_flashnext_chain_p("0.5", COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE));
    bool threw = false;
    try {
        common_flashnext_chain_p("half", DFLASH);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    t.assert_true("draft-dflash a bad value throws, default or not", threw);
}

// common_params_parse as llama-server does it
common_params parse(testing & t, std::vector<std::string> args) {
    args.insert(args.begin(), "llama-server");
    std::vector<char *> argv;
    for (auto & a : args) {
        argv.push_back(a.data());
    }
    common_params params;
    t.assert_true("parses", common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_SERVER));
    return params;
}

bool has(const std::vector<std::string> & v, const std::string & s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

bool is_dflash(const common_params & p) {
    return p.speculative.types == std::vector<common_speculative_type>{ COMMON_SPECULATIVE_TYPE_NONE, DFLASH };
}

void test_preset(testing & t) {
    const std::vector<std::string> dflash = { "-md", "/m/dflash.gguf", "--spec-type", "draft-dflash" };
    auto with = [&](std::vector<std::string> extra) {
        std::vector<std::string> args = { "--flashnext" };
        args.insert(args.end(), dflash.begin(), dflash.end());
        args.insert(args.end(), extra.begin(), extra.end());
        return args;
    };
    {
        const common_params p = parse(t, with({}));
        t.assert_true("the user's type alone", is_dflash(p));
        t.assert_equal("--spec-draft-n-max 7", 7, p.speculative.draft.n_max);
        t.assert_equal("--spec-draft-p-min 0", 0.0f, p.speculative.draft.p_min);
        t.assert_equal("--spec-draft-ngl 99, as for the MTP head", 99, p.speculative.draft.n_gpu_layers);
        t.assert_equal("--spec-max-prompt 0", 0, (int) p.speculative.n_prompt_max);
        t.assert_true("recorded: --spec-draft-n-max 7", has(p.flashnext_set, "--spec-draft-n-max 7"));
        t.assert_true("recorded: --spec-draft-p-min 0", has(p.flashnext_set, "--spec-draft-p-min 0"));
        t.assert_true("not recorded: the MTP head's 5", !has(p.flashnext_set, "--spec-draft-n-max 5"));
        t.assert_true("yours: --spec-type", has(p.flashnext_yours, "--spec-type"));
        t.assert_equal("the other twenty-one set", (size_t) 21, p.flashnext_set.size());
    }
    {
        const common_params p = parse(t, with({ "--spec-draft-n-max", "4" }));
        t.assert_equal("the user's n-max wins", 4, p.speculative.draft.n_max);
        t.assert_equal("p-min still DFlash's", 0.0f, p.speculative.draft.p_min);
        t.assert_true("yours: --spec-draft-n-max", has(p.flashnext_yours, "--spec-draft-n-max"));
    }
    {
        const common_params p = parse(t, with({ "--draft-p-min", "0.2" }));
        t.assert_equal("the user's p-min wins, in its other spelling", 0.2f, p.speculative.draft.p_min);
        t.assert_equal("n-max still DFlash's", 7, p.speculative.draft.n_max);
    }
    {
        setenv("LLAMA_ARG_SPEC_DRAFT_N_MAX", "3", 1);
        setenv("LLAMA_ARG_SPEC_DRAFT_P_MIN", "0.3", 1);
        const common_params p = parse(t, with({}));
        unsetenv("LLAMA_ARG_SPEC_DRAFT_N_MAX");
        unsetenv("LLAMA_ARG_SPEC_DRAFT_P_MIN");
        t.assert_equal("LLAMA_ARG_SPEC_DRAFT_N_MAX wins", 3, p.speculative.draft.n_max);
        t.assert_equal("LLAMA_ARG_SPEC_DRAFT_P_MIN wins", 0.3f, p.speculative.draft.p_min);
    }
    {
        // the MTP head (the preset's own type) keeps its values
        const common_params p = parse(t, { "--flashnext", "-md", "/m/draft.gguf" });
        t.assert_equal("the MTP head: n-max 5", 5, p.speculative.draft.n_max);
        t.assert_equal("the MTP head: p-min 0.3", 0.3f, p.speculative.draft.p_min);
    }
    {
        // without the preset: upstream's defaults (llama.cpp's own options are the preset's business)
        const common_params p = parse(t, dflash);
        const common_params upstream{};
        t.assert_true("DFlash", is_dflash(p));
        t.assert_equal("no preset: upstream's n-max", upstream.speculative.draft.n_max, p.speculative.draft.n_max);
        t.assert_equal("no preset: upstream's p-min", upstream.speculative.draft.p_min, p.speculative.draft.p_min);
    }
    t.assert_true("the help names DFlash's rows",
        common_flashnext_preset_str().find("; with --spec-type draft-dflash: --spec-draft-n-max 7 --spec-draft-p-min 0")
        != std::string::npos);
}

struct fake_env {
    std::map<std::string, std::string> vars;
    const char * operator()(const char * name) const {
        auto it = vars.find(name);
        return it == vars.end() ? nullptr : it->second.c_str();
    }
};

void test_line(testing & t) {
    common_params params;
    fake_env env;
    auto line = [&]() { return common_flashnext_load_line(params, std::cref(env)); };

    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE };
    t.assert_true("the MTP head: the chain off", line().find(", LLAMA_SPEC_CHAIN_P=off, ") != std::string::npos);

    params.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE, DFLASH };
    t.assert_true("draft-dflash: its default named",
        line().find(", LLAMA_SPEC_CHAIN_P=0.5 (draft-dflash), ") != std::string::npos);
    env.vars["LLAMA_SPEC_CHAIN_P"] = "0";
    t.assert_true("draft-dflash, \"0\": yours", line().find(", LLAMA_SPEC_CHAIN_P=0 (yours), ") != std::string::npos);
    env.vars["LLAMA_SPEC_CHAIN_P"] = "0.6";
    t.assert_true("draft-dflash, \"0.6\": yours", line().find(", LLAMA_SPEC_CHAIN_P=0.6 (yours), ") != std::string::npos);

    // the preset's part says what it set for DFlash and kept of the user's
    env.vars.clear();
    params.flashnext       = true;
    params.flashnext_set   = { "--spec-draft-n-max 7", "--spec-draft-p-min 0" };
    params.flashnext_yours = { "--spec-type" };
    params.speculative.draft.mparams.path = "/m/dflash.gguf";
    t.assert_true("the preset's DFlash rows on the line", line().find(
        "; preset --flashnext: on, set --spec-draft-n-max 7 --spec-draft-p-min 0, yours kept: --spec-type")
        != std::string::npos);
}

} // namespace

int main(int argc, char ** argv) {
    testing t(std::cout);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    // a config directory of our own: the user's config.ini must not move these results
    const auto dir = std::filesystem::temp_directory_path() / ("flashnext-dflash-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    setenv("XDG_CONFIG_HOME", dir.c_str(), 1);
    for (const char * e : { "LLAMA_ARG_FLASHNEXT", "LLAMA_ARG_SPEC_DRAFT_N_MAX", "LLAMA_ARG_SPEC_DRAFT_P_MIN",
                            "LLAMA_ARG_SPEC_TYPE", "LLAMA_SPEC_CHAIN_P" }) {
        unsetenv(e);
    }

    t.test("the chain stop's default", test_chain_default);
    t.test("the preset", test_preset);
    t.test("the start-up line", test_line);

    std::filesystem::remove_all(dir);
    return t.summary();
}

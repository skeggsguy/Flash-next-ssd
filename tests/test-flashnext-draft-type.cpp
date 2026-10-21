// --flashnext picks the draft type from the draft file (the C37 roll-out, step 4a, 2026-10-21): common/flashnext.h.
// No model loads; the drafts are tiny generated GGUFs (flashnext-draft-gguf.h) with only the metadata that is read.
//
// - common_flashnext_draft_type: DFlash's architecture is draft-dflash (draft-dspark with the Markov head), anything
//   else draft-mtp-adaptive, and a file that is not a readable GGUF throws, naming it;
// - through common_params_parse, as llama-server parses: a DFlash file with no --spec-type of the user's runs
//   draft-dflash with DFlash's depth 7, p-min 0 and the chain stop 0.5; an MTP-head file draft-mtp-adaptive with
//   5 / 0.3 and the chain off; the user's --spec-type (command line or LLAMA_ARG_SPEC_TYPE) always wins and the file
//   is then not read; no -md changes nothing; a draft still to be downloaded (-hfd) keeps the preset's type; an
//   unreadable -md fails the parse; without --flashnext the file is not read here;
// - the start-up line: the type and why ("(from the draft file)"), and the chain's default for that type.

#include "testing.h"

#include "arg.h"
#include "common.h"
#include "flashnext.h"
#include "flashnext-draft-gguf.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

constexpr common_speculative_type NONE   = COMMON_SPECULATIVE_TYPE_NONE;
constexpr common_speculative_type DFLASH = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
constexpr common_speculative_type MTP    = COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE;

std::string g_dir; // this run's own folder for the generated drafts

std::string draft(const std::string & name, const std::string & arch, bool markov = false) {
    const std::string path = g_dir + "/" + name;
    flashnext_write_draft_gguf(path, arch, markov);
    return path;
}

void test_draft_type(testing & t) {
    t.assert_true("dflash: draft-dflash", common_flashnext_draft_type(draft("df.gguf", "dflash")) == DFLASH);
    t.assert_true("dflash with the Markov head: draft-dspark",
        common_flashnext_draft_type(draft("ds.gguf", "dflash", true)) == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK);
    t.assert_true("qwen4exp (the MTP head): draft-mtp-adaptive",
        common_flashnext_draft_type(draft("mtp.gguf", "qwen4exp")) == MTP);
    t.assert_true("a Markov tensor without DFlash's architecture: draft-mtp-adaptive",
        common_flashnext_draft_type(draft("mk.gguf", "qwen4exp", true)) == MTP);
    t.assert_true("no architecture: draft-mtp-adaptive", common_flashnext_draft_type(draft("na.gguf", "")) == MTP);

    const std::string text = g_dir + "/notes.gguf";
    std::ofstream(text) << "not a gguf\n";
    for (const std::string & bad : { g_dir + "/missing.gguf", text }) {
        std::string what;
        try {
            common_flashnext_draft_type(bad);
        } catch (const std::invalid_argument & e) {
            what = e.what();
        }
        t.assert_true("unreadable " + bad + ": throws, naming the file", what.find("'" + bad + "'") != std::string::npos);
    }
}

// common_params_parse as llama-server does it; `ok` false when it refuses the flags
common_params parse(std::vector<std::string> args, bool & ok) {
    args.insert(args.begin(), "llama-server");
    std::vector<char *> argv;
    for (auto & a : args) {
        argv.push_back(a.data());
    }
    common_params params;
    ok = common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_SERVER);
    return params;
}

common_params parse(testing & t, const std::vector<std::string> & args) {
    bool ok = false;
    common_params p = parse(args, ok);
    t.assert_true("parses", ok);
    return p;
}

bool has(const std::vector<std::string> & v, const std::string & s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

bool types_are(const common_params & p, std::vector<common_speculative_type> types) {
    return p.speculative.types == types;
}

struct fake_env {
    std::map<std::string, std::string> vars;
    const char * operator()(const char * name) const {
        auto it = vars.find(name);
        return it == vars.end() ? nullptr : it->second.c_str();
    }
};

std::string line(const common_params & p) {
    const fake_env env; // nothing set: every default
    return common_flashnext_load_line(p, std::cref(env));
}

// the chain stop's threshold the draft opens with, the switch unset (common_speculative reads it per type)
double chain(const common_params & p) {
    return common_flashnext_chain_p(nullptr, p.speculative.types.back());
}

void test_from_file(testing & t) {
    const std::string df  = draft("Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf", "dflash");
    const std::string mtp = draft("mtp-shared-Q4_K_M.gguf", "qwen4exp");
    {
        const common_params p = parse(t, { "--flashnext", "-md", df });
        t.assert_true("a DFlash file: draft-dflash", types_are(p, { NONE, DFLASH }));
        t.assert_equal("DFlash's n-max 7", 7, p.speculative.draft.n_max);
        t.assert_equal("DFlash's p-min 0", 0.0f, p.speculative.draft.p_min);
        t.assert_equal("the chain stop 0.5", 0.5, chain(p));
        t.assert_true("recorded with why", has(p.flashnext_set, "--spec-type draft-dflash (from the draft file)"));
        t.assert_true("recorded: --spec-draft-n-max 7", has(p.flashnext_set, "--spec-draft-n-max 7"));
        t.assert_true("all twenty-two set", p.flashnext_set.size() == 22 && p.flashnext_yours.empty());
        const std::string l = line(p);
        t.assert_true("the line: the type and why", l.find("set -ngl 99 ") != std::string::npos &&
            l.find(" --spec-type draft-dflash (from the draft file) --spec-draft-n-max 7 --spec-draft-p-min 0 ")
            != std::string::npos);
        t.assert_true("the line: the chain's default", l.find(", LLAMA_SPEC_CHAIN_P=0.5 (draft-dflash), ")
            != std::string::npos);
    }
    {
        const common_params p = parse(t, { "--flashnext", "-md", mtp });
        t.assert_true("an MTP-head file: draft-mtp-adaptive", types_are(p, { NONE, MTP }));
        t.assert_equal("the head's n-max 5", 5, p.speculative.draft.n_max);
        t.assert_equal("the head's p-min 0.3", 0.3f, p.speculative.draft.p_min);
        t.assert_equal("the chain off", 0.0, chain(p));
        t.assert_true("recorded with why", has(p.flashnext_set, "--spec-type draft-mtp-adaptive (from the draft file)"));
        t.assert_true("the line: the chain off", line(p).find(", LLAMA_SPEC_CHAIN_P=off, ") != std::string::npos);
    }
}

void test_yours_wins(testing & t) {
    const std::string df = draft("dflash.gguf", "dflash");
    {
        const common_params p = parse(t, { "--flashnext", "-md", df, "--spec-type", "draft-mtp-adaptive" });
        t.assert_true("the user's type over the DFlash file", types_are(p, { NONE, MTP }));
        t.assert_equal("then the head's n-max", 5, p.speculative.draft.n_max);
        t.assert_equal("then the chain off", 0.0, chain(p));
        t.assert_true("yours: --spec-type", has(p.flashnext_yours, "--spec-type"));
    }
    {
        setenv("LLAMA_ARG_SPEC_TYPE", "draft-mtp", 1);
        const common_params p = parse(t, { "--flashnext", "-md", df });
        unsetenv("LLAMA_ARG_SPEC_TYPE");
        t.assert_true("LLAMA_ARG_SPEC_TYPE over the DFlash file", types_are(p, { NONE, COMMON_SPECULATIVE_TYPE_DRAFT_MTP }));
        t.assert_true("yours: --spec-type", has(p.flashnext_yours, "--spec-type"));
    }
    {
        // the user's type: the file is not read at all, as before (its load says what is wrong with it)
        const common_params p = parse(t, { "--flashnext", "-md", g_dir + "/missing.gguf", "--spec-type", "draft-dflash" });
        t.assert_true("the user's type, an unread file", types_are(p, { NONE, DFLASH }));
        t.assert_equal("DFlash's n-max", 7, p.speculative.draft.n_max);
    }
    {
        const common_params p = parse(t, { "--flashnext", "-md", df, "--spec-draft-n-max", "4" });
        t.assert_true("the file's type", types_are(p, { NONE, DFLASH }));
        t.assert_equal("the user's n-max still wins", 4, p.speculative.draft.n_max);
        t.assert_equal("p-min DFlash's", 0.0f, p.speculative.draft.p_min);
    }
}

void test_no_file(testing & t) {
    {
        const common_params p = parse(t, { "--flashnext" });
        t.assert_true("no -md: no type", types_are(p, { NONE }));
        t.assert_true("no -md: no draft rows", !has(p.flashnext_set, "--spec-draft-n-max 5") &&
                                               !has(p.flashnext_set, "--spec-draft-n-max 7"));
        t.assert_equal("no -md: seventeen set", (size_t) 17, p.flashnext_set.size());
    }
    {
        // a draft still to be downloaded can't be read at the parse: the preset's own type, as before
        const common_params p = parse(t, { "--flashnext", "-hfd", "someone/draft" });
        t.assert_true("-hfd: the preset's type", types_are(p, { NONE, MTP }));
        t.assert_true("-hfd: no why", has(p.flashnext_set, "--spec-type draft-mtp-adaptive"));
        // -md beside -hfd names a file in the repo, not one on this disk
        const common_params q = parse(t, { "--flashnext", "-hfd", "someone/draft", "-md", "draft-Q8_0.gguf" });
        t.assert_true("-hfd with -md: the preset's type, the name unread", types_are(q, { NONE, MTP }));
    }
    {
        // without the preset the parse leaves the type to upstream's own reading at the download step
        const common_params p = parse(t, { "-md", g_dir + "/missing.gguf" });
        t.assert_true("no --flashnext: the type untouched, the file unread", types_are(p, { NONE }));
    }
    for (const std::string & bad : { g_dir + "/missing.gguf", g_dir + "/notes.gguf" }) {
        bool ok = true;
        parse({ "--flashnext", "-md", bad }, ok);
        t.assert_true("an unreadable -md fails the start: " + bad, !ok);
    }
}

} // namespace

int main(int argc, char ** argv) {
    testing t(std::cout);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    // a folder of our own for the drafts and the config: the user's config.ini must not move these results
    const auto dir = std::filesystem::temp_directory_path() / ("flashnext-draft-type-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    g_dir = dir.string();
    setenv("XDG_CONFIG_HOME", dir.c_str(), 1);
    for (const char * e : { "LLAMA_ARG_FLASHNEXT", "LLAMA_ARG_SPEC_DRAFT_N_MAX", "LLAMA_ARG_SPEC_DRAFT_P_MIN",
                            "LLAMA_ARG_SPEC_TYPE", "LLAMA_ARG_SPEC_DRAFT_MODEL", "LLAMA_SPEC_CHAIN_P" }) {
        unsetenv(e);
    }

    t.test("the type from the file", test_draft_type);
    t.test("the preset reads the file", test_from_file);
    t.test("the user's type wins", test_yours_wins);
    t.test("no file to read", test_no_file);

    std::filesystem::remove_all(dir);
    return t.summary();
}

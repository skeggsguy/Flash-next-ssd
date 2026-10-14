// The fork's proven defaults, common's side: LLAMA_SPEC_ADAPTIVE_RATE's parse, the --flashnext preset and the
// start-up line. The rules are in flashnext.h.

#include "flashnext.h"

#include "arg.h"
#include "common.h"

#include "../src/llama-flashnext.h" // the libllama side's parsers: what an unset switch means

#include <cstdlib>
#include <cstring>
#include <string>

int common_flashnext_adaptive_rate(const char * value) {
    if (value == nullptr) {
        return 1;
    }
    if (*value == '\0' || strcmp(value, "0") == 0) {
        return 0;
    }
    return strcmp(value, "2") == 0 ? 2 : 1;
}

// the everyday values (the study's serve.py launch, 2026-10-15); the machine choices (the desk, the second SSD,
// host/port, the chat window, the opening copy) and the downloads (-md, --mmproj, the guessers) are not here
const std::vector<common_flashnext_setting> & common_flashnext_preset() {
    static const std::vector<common_flashnext_setting> preset = {
        { "-ngl",                   "99",                 false },
        { "--moe-stream",           "",                   false },
        { "--moe-stream-direct",    "",                   false },
        { "--moe-stream-io-threads","8",                  false },
        { "-c",                     "200000",             false },
        { "-b",                     "4096",               false },
        { "-ub",                    "4096",               false },
        { "-cms",                   "8192",               false },
        { "-ctxcp",                 "3",                  false },
        { "-np",                    "1",                  false },
        { "-fa",                    "on",                 false },
        { "--cache-reuse",          "0",                  false },
        { "--cache-ram",            "0",                  false },
        { "--jinja",                "",                   false },
        { "--reasoning-format",     "deepseek",           false },
        { "--spec-type",            "draft-mtp-adaptive", true  },
        { "--spec-draft-n-max",     "5",                  true  },
        { "--spec-draft-p-min",     "0.3",                true  },
        { "--spec-draft-ngl",       "99",                 true  },
        { "--spec-max-prompt",      "0",                  true  },
    };
    return preset;
}

static std::string flashnext_setting_str(const common_flashnext_setting & s) {
    return *s.value == '\0' ? std::string(s.arg) : std::string(s.arg) + " " + s.value;
}

std::string common_flashnext_preset_str() {
    std::string res;
    bool draft = false;
    for (const auto & s : common_flashnext_preset()) {
        if (s.draft && !draft) {
            res += ", with -md:";
            draft = true;
        }
        res += (res.empty() ? "" : " ") + flashnext_setting_str(s);
    }
    return res;
}

// as the command line hands a flag to its option (common_params_parse_ex's parse_cli_args)
static void flashnext_set(common_params & params, const common_arg & opt, const char * value) {
    if (opt.handler_void) {
        opt.handler_void(params);
    } else if (opt.handler_bool) {
        opt.handler_bool(params, true);
    } else if (opt.handler_int) {
        opt.handler_int(params, std::stoi(value));
    } else if (opt.handler_string) {
        opt.handler_string(params, value);
    } else {
        GGML_ABORT("--flashnext: %s takes two values", opt.args[0]);
    }
}

void common_flashnext_apply(common_params & params,
                            const std::function<const common_arg * (const std::string &)> & find,
                            const std::set<std::string> & given) {
    params.flashnext_set.clear();
    params.flashnext_yours.clear();
    const bool has_draft = params.speculative.has_dft();
    for (const auto & s : common_flashnext_preset()) {
        const common_arg * opt = find(s.arg);
        if (opt == nullptr) {
            continue; // not a flag of this program
        }
        if (given.count(opt->args[0]) > 0) {
            params.flashnext_yours.push_back(s.arg);
            continue;
        }
        if (s.draft && !has_draft) {
            continue; // nothing to draft with: the line says the draft settings wait for -md
        }
        flashnext_set(params, *opt, s.value);
        params.flashnext_set.push_back(flashnext_setting_str(s));
    }
}

// "NAME=value" for a switch: the user's value marked so, else the default
static std::string flashnext_switch(const char * name, const char * value, const std::string & dflt) {
    return value != nullptr ? std::string(name) + "=" + value + " (yours)" : std::string(name) + "=" + dflt;
}

static std::string flashnext_float(float v) {
    return string_format("%g", v);
}

static std::string flashnext_join(const std::vector<std::string> & parts, const char * sep) {
    std::string res;
    for (const auto & p : parts) {
        res += (res.empty() ? "" : sep) + p;
    }
    return res;
}

std::string common_flashnext_load_line(const common_params & params,
                                       const std::function<const char * (const char *)> & env) {
    // the engine's own switches: each one's default comes from the parser the engine reads it with
    const std::vector<std::string> engine = {
        flashnext_switch("LLAMA_MOE_STREAM_LRU", env("LLAMA_MOE_STREAM_LRU"),
                llama_flashnext_lru(nullptr) ? "1" : "0"),
        flashnext_switch("LLAMA_RS_SKIP_SLOTS", env("LLAMA_RS_SKIP_SLOTS"),
                llama_flashnext_rs_skip_slots(nullptr) ? "1" : "0"),
        flashnext_switch("LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB", env("LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB"),
                std::to_string(llama_flashnext_alloc_chunk_mib(nullptr))),
        flashnext_switch("LLAMA_SPEC_ADAPTIVE_RATE", env("LLAMA_SPEC_ADAPTIVE_RATE"),
                std::to_string(common_flashnext_adaptive_rate(nullptr))),
        flashnext_switch("GGML_METAL_STAFF_MV", env("GGML_METAL_STAFF_MV"),
                std::to_string(COMMON_FLASHNEXT_STAFF_MV)),
        flashnext_switch("GGML_METAL_RESIDENCY_KEEP_ALIVE_S", env("GGML_METAL_RESIDENCY_KEEP_ALIVE_S"),
                std::to_string(COMMON_FLASHNEXT_KEEP_ALIVE_S)),
    };
    std::string line = "flashnext: engine defaults: " + flashnext_join(engine, ", ");

    // the trained two-floor guesser's tuning: the everyday values only with the guesser's file named
    const bool guesser = llama_flashnext_guesser_named(env("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER"));
    struct tuning { const char * name; std::string dflt; };
    const tuning tunings[] = {
        { "LLAMA_MOE_STREAM_LOOKAHEAD_ALL",    llama_flashnext_lookahead_all(nullptr, guesser) ? "1" : "0" },
        { "LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2", std::to_string(llama_flashnext_lookahead_depth2(nullptr, guesser)) },
        { "LLAMA_MOE_STREAM_LOOKAHEAD_KEEP",   llama_flashnext_lookahead_keep(nullptr, guesser) ? "1" : "0" },
        { "LLAMA_MOE_STREAM_LOOKAHEAD_CONF",   flashnext_float(llama_flashnext_lookahead_conf(nullptr, guesser)) },
    };
    std::vector<std::string> tuned;
    for (const auto & t : tunings) {
        const char * value = env(t.name);
        if (guesser || value != nullptr) {
            tuned.push_back(flashnext_switch(t.name, value, t.dflt));
        }
    }
    if (guesser) {
        line += "; the guesser's tuning: " + flashnext_join(tuned, ", ");
    } else {
        line += "; the guesser's tuning: off, no guesser named";
        if (!tuned.empty()) {
            line += ", but " + flashnext_join(tuned, ", ");
        }
    }

    if (!params.flashnext) {
        return line + "; preset --flashnext: off";
    }
    line += "; preset --flashnext: on, set " + (params.flashnext_set.empty() ? std::string("nothing") :
            flashnext_join(params.flashnext_set, " "));
    if (!params.flashnext_yours.empty()) {
        line += ", yours kept: " + flashnext_join(params.flashnext_yours, " ");
    }
    if (!params.speculative.has_dft()) {
        line += ", the draft settings wait for -md";
    }
    return line;
}

std::string common_flashnext_load_line(const common_params & params) {
    return common_flashnext_load_line(params, [](const char * name) { return (const char *) getenv(name); });
}

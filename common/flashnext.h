#pragma once

// The fork's proven defaults, common's side (the Flash-Next study's standard roll-out, 2026-10-15; the libllama side
// is src/llama-flashnext.h).
//
// - LLAMA_SPEC_ADAPTIVE_RATE: the draft depth that seeks the most words a second, the everyday value, now the default.
// - LLAMA_SPEC_CHAIN_P: the chain stop (CHAIN-PLAN step 2). Unset it is 0.5 with --spec-type draft-dflash (C37:
//   +10.8% chat, +8.7% code at depth 7) and off for every other draft (the apprentice + chain missed the bar).
// - `--flashnext`: one preset flag that sets the everyday values of llama.cpp's general flags, each only where the
//   user did not pass that flag (on the command line, as LLAMA_ARG_* or in a config file); upstream's defaults are
//   untouched without it. The draft settings wait for a draft model (-md): without one there is nothing to draft.
//   With --spec-type draft-dflash the user's, its depth and p-min are DFlash's own (7 and 0, C37), not the MTP head's.
// - the start-up line: one line naming the defaults in force and whether the preset is on. Its first word is
//   "flashnext:", which the study's log readers (runner/roomstats.py) never take for a desk line.

#include "common.h" // common_speculative_type

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

struct common_params;
struct common_arg;

// LLAMA_SPEC_ADAPTIVE_RATE, with --spec-type draft-mtp-adaptive: 1 the measured depth (speculative-rate.h), 2 the
// priced depth (speculative-rate2.h), 0 the acceptance controller (the old way). Unset: 1. "" or "0": 0. "2": 2.
// Anything else: 1 (the switch has always read this way).
int common_flashnext_adaptive_rate(const char * value);

// LLAMA_SPEC_CHAIN_P, the chain stop's threshold (common/speculative-chain.h) for a draft of `type`: unset is
// COMMON_FLASHNEXT_CHAIN_P_DFLASH for draft-dflash and off (0) for every other type (draft-dspark included); "" or "0"
// is off for any type; a number in (0, 1] is the threshold, read as a double so the study's replay cuts where the
// engine does; anything else throws, so a typo fails the start instead of running with the stop off.
// LLAMA_SPEC_ADAPTIVE_RATE=2 ignores it (the price stops that draft), with a warning.
double common_flashnext_chain_p(const char * value, common_speculative_type type);

// the chain stop's threshold with draft-dflash when LLAMA_SPEC_CHAIN_P is unset (C37's winner, 2026-10-21)
constexpr double COMMON_FLASHNEXT_CHAIN_P_DFLASH = 0.5;

// ggml-metal's two, as its parsers read them unset (ggml-metal-staff-mv.h, ggml-metal-keep-alive.h); kept equal by
// test-flashnext-metal, so the start-up line cannot name a default the device does not run
constexpr int     COMMON_FLASHNEXT_STAFF_MV     = 0;        // GGML_METAL_STAFF_MV: today's route ("1": matrix units)
constexpr int64_t COMMON_FLASHNEXT_KEEP_ALIVE_S = 10000000; // GGML_METAL_RESIDENCY_KEEP_ALIVE_S ("0": 180, upstream's)

// one of the preset's settings: the flag (its first spelling, as `common_arg::args[0]`) and its value; a flag that
// takes no value has "" (it is switched on)
struct common_flashnext_setting {
    const char * arg;
    const char * value;
    bool         draft; // a draft setting: only with a draft model
};

// the preset, in the order the help and the start-up line give it
const std::vector<common_flashnext_setting> & common_flashnext_preset();

// the draft rows whose value differs with --spec-type draft-dflash (the user's choice; the preset's own type is the
// MTP head's): DFlash's depth and p-min, which replace the preset's rows of the same flag
const std::vector<common_flashnext_setting> & common_flashnext_preset_dflash();

// "-ngl 99 --moe-stream ..." (draft settings after "with -md:", DFlash's after "with --spec-type draft-dflash:"), for
// --flashnext's help
std::string common_flashnext_preset_str();

// Applies the preset to `params` (call after every source of flags has been read). `find`: the option a flag names
// in this program, or nullptr (a flag the program does not have is skipped). `given`: the `args[0]` of every option
// the user passed. Records what it set in params.flashnext_set and what it left to the user in flashnext_yours.
void common_flashnext_apply(common_params & params,
                            const std::function<const common_arg * (const std::string &)> & find,
                            const std::set<std::string> & given);

// the start-up line, from the environment as it is now (`getenv`) and what the preset did
std::string common_flashnext_load_line(const common_params & params);

// the same, from a given environment (tests): `env(name)` is the switch's value or nullptr when unset
std::string common_flashnext_load_line(const common_params & params,
                                       const std::function<const char * (const char *)> & env);

#pragma once

// The harness shared by test-moe-room-model.cpp and test-moe-room-model-paths.cpp: the 64-book fixtures
// with the reading room (--moe-stream-room) against the same model with every book resident. The room's
// promise is that a floor split between the desk and the belt gives the same bytes as no streaming at
// all, whatever happens to be on the desk, so every comparison is memcmp: the logits at every step and
// every floor's MoE output (ffn_moe_out, via the eval callback) at every ubatch. The fixture's tiny
// weights leave the logits blind to a wrong book; the MoE outputs are not.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-model.h"
#include "../src/llama-moe-room.h"
#include "../src/llama-moe-stream.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

struct config {
    bool     stream     = true;
    uint32_t slots      = 55;   // budget in slots: 55 with auto leaves a desk of 40 (2 floors, 1.25 of 24 books)
    int32_t  room_mode  = LLAMA_MOE_ROOM_AUTO;
    float    room_value = 0.0f;
    int32_t  parts      = 4;
    uint32_t ubatch     = 0;     // the batch the model is told its contexts read in with (0 = not said)
    bool     direct     = false; // --moe-stream-direct: the study's path, staging + tensor_set onto the belt
    std::string alt;             // --moe-stream-alt-path: two runners, a byte-identical copy
};

struct segment {
    std::vector<llama_token> prompt;
    int  n_write = 0;
    bool keep    = true; // outputs of this segment are compared
    bool all     = true; // logits for every prompt position, or only the last (as a server asks)
};

struct outputs {
    std::vector<float> logits; // every prompt position, then one row per written token
    std::vector<float> moe;    // every ffn_moe_out, in compute order
};

const char * const G_SWEEP_MIN_TOKENS = "35"; // the threshold the scenarios run at (setup says why)

int g_ngl = 99;
std::string g_summary;  // the last "reading room =" line print_stats wrote
std::string g_room_log; // every log line about the room (startup line, warnings, the belt's allocation)
std::string g_path;

bool capture_moe_out(ggml_tensor * t, bool ask, void * user_data) {
    const bool want = strncmp(t->name, "ffn_moe_out-", strlen("ffn_moe_out-")) == 0;
    if (ask || !want) {
        return want;
    }
    auto * o = (outputs *) user_data;
    if (o != nullptr) {
        const size_t n0 = o->moe.size();
        o->moe.resize(n0 + ggml_nelements(t));
        ggml_backend_tensor_get(t, o->moe.data() + n0, 0, ggml_nbytes(t));
    }
    return true;
}

llama_model * load(const config & c) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers          = g_ngl;
    mp.use_extra_bufts       = false; // no CPU repacking: the reference must run the same kernels
    mp.moe_stream            = c.stream;
    mp.moe_stream_slots      = c.slots;
    mp.moe_stream_room_mode  = c.room_mode;
    mp.moe_stream_room_value = c.room_value;
    mp.moe_stream_room_parts = c.parts;
    mp.moe_stream_room_ubatch = c.ubatch;
    mp.moe_stream_direct     = c.direct;
    mp.moe_stream_alt_path   = c.alt.empty() ? nullptr : c.alt.c_str();
    return llama_model_load_from_file(g_path.c_str(), mp);
}

int64_t g_n_reused = 0; // graphs the last run's context reused instead of rebuilding

// runs the segments on one context, clearing its memory between them (the model, and so the desk, stays);
// `between` runs after each segment but the last, in place of whatever another context would do there
bool run(llama_model * model, uint32_t n_ubatch, const std::vector<segment> & segs, outputs & out, bool capture,
        const std::function<void()> & between = {}) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    outputs scratch;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 1024;
    cp.n_batch  = 512;
    cp.n_ubatch = n_ubatch;
    cp.no_perf  = false; // for n_reused
    cp.op_offload = false; // streaming turns it off (host desk), so the reference must not offload either
    if (capture) {
        cp.cb_eval           = capture_moe_out;
        cp.cb_eval_user_data = &scratch;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        return false;
    }
    bool ok = true;
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (const segment & s : segs) {
        llama_memory_clear(llama_get_memory(ctx), true);
        scratch.moe.clear();
        std::vector<float> logits;
        common_batch_clear(batch);
        for (size_t i = 0; i < s.prompt.size(); i++) {
            common_batch_add(batch, s.prompt[i], (llama_pos) i, { 0 }, s.all || i + 1 == s.prompt.size());
        }
        ok = ok && llama_decode(ctx, batch) == 0;
        if (ok) {
            const float * l = s.all ? llama_get_logits(ctx) : llama_get_logits_ith(ctx, -1);
            logits.assign(l, l + (s.all ? s.prompt.size() : 1)*n_vocab);
        }
        for (int i = 0; ok && i < s.n_write; i++) {
            const float * row = logits.data() + logits.size() - n_vocab;
            llama_token best = 0;
            for (int v = 1; v < n_vocab; v++) {
                best = row[v] > row[best] ? v : best;
            }
            common_batch_clear(batch);
            common_batch_add(batch, best, (llama_pos) (s.prompt.size() + i), { 0 }, true);
            ok = llama_decode(ctx, batch) == 0;
            if (ok) {
                const float * l = llama_get_logits_ith(ctx, -1);
                logits.insert(logits.end(), l, l + n_vocab);
            }
        }
        if (s.keep) {
            out.logits.insert(out.logits.end(), logits.begin(), logits.end());
            out.moe.insert(out.moe.end(), scratch.moe.begin(), scratch.moe.end());
        }
        if (between && &s != &segs.back()) {
            between();
        }
    }
    g_n_reused = llama_perf_context(ctx).n_reused;
    llama_batch_free(batch);
    llama_free(ctx);
    return ok;
}

llama_moe_room * room_of(llama_model * model) {
    return model->moe_stream() ? model->moe_stream()->room.get() : nullptr;
}

int64_t n_streamed_floors(llama_model * model) {
    return model->moe_stream() ? model->moe_stream()->n_streamed_layers() : 0;
}

// what is on each floor's desk, slot by slot (-1 = EMPTY)
std::vector<int32_t> desk_of(llama_model * model) {
    std::vector<int32_t> d;
    for (const auto & sl : model->moe_stream()->layers) {
        for (uint32_t s = 0; sl && s < sl->n_slots; s++) {
            d.push_back(sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY ? -1 : sl->slot_expert[s]);
        }
    }
    return d;
}

std::vector<llama_token> random_tokens(std::mt19937 & rng, size_t n, int n_vocab) {
    std::vector<llama_token> t(n);
    for (auto & x : t) {
        x = (llama_token) (rng() % (uint32_t) n_vocab);
    }
    return t;
}

int g_failures = 0;

void check(bool ok, const std::string & what) {
    printf("  %-72s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

// the compared MoE outputs must carry signal: with the fixture's weights near 0 a wrong book can round
// to the same zero (the e64 fixtures scale weights around 1 for this)
bool nonzero(const std::vector<float> & v) {
    size_t n = 0;
    for (const float x : v) {
        n += x != 0.0f;
    }
    return n*100 >= v.size()*99;
}

double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double ab = 0.0, a0 = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        ab += (double) (a[i] - b[i])*(a[i] - b[i]);
        a0 += (double) a[i]*a[i];
    }
    return a.size() == b.size() && a0 > 0.0 ? ab/a0 : 1.0;
}

// runs segs on a fresh model of config c and compares the kept segments with the reference; the room's
// groups must have run room_ubatches ubatches' worth on every floor
void scenario(const std::string & name, const config & c, uint32_t n_ubatch, const std::vector<segment> & segs,
        const outputs & ref, int64_t room_ubatches, bool capture = true,
        const std::function<void(const llama_model *)> & also = {}) {
    llama_model * model = load(c);
    outputs got;
    const bool ok = model != nullptr && run(model, n_ubatch, segs, got, capture);
    check(ok, name + ": ran");
    if (!ok) {
        llama_model_free(model);
        return;
    }
    if (also) {
        also(model);
    }
    llama_moe_room * room = room_of(model);
    const int64_t groups  = room ? room->stats.n_groups : 0;
    const int64_t want    = room_ubatches*n_streamed_floors(model)*(1 + c.parts);
    check(groups == want, name + ": room groups ran " + std::to_string(groups) + " of " + std::to_string(want));
    if (room != nullptr) {
        g_summary.clear();
        llama_moe_stream_print_stats(model);
        check(g_summary.find(" " + std::to_string(groups) + " groups,") != std::string::npos,
                name + ": the run's summary counts them");
    }
    if (room_ubatches == 0) {
        printf("  %-72s nmse logits %.3e moe %.3e\n", (name + ": waves vs reference (printed)").c_str(),
                nmse(ref.logits, got.logits), nmse(ref.moe, got.moe));
    } else {
        check(same(ref.logits, got.logits), name + ": logits byte-identical to no streaming");
        if (capture) {
            check(same(ref.moe, got.moe) && nonzero(ref.moe), name + ": every ffn_moe_out byte-identical, nonzero");
        }
    }
    llama_model_free(model);
}

// -m and -ngl from argv, a silent log that keeps the room summary, the backends; returns the reference
// model (every book resident) or null after saying why
llama_model * setup(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            g_path = argv[++i];
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            g_ngl = atoi(argv[++i]);
        }
    }
    if (g_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [-ngl N]\n", argv[0]);
        return nullptr;
    }
    // silent, except that the run's room summary (print_stats) and what loading said about the room are
    // caught, to check they are written
    llama_log_set([](ggml_log_level, const char * text, void *) {
        if (strstr(text, "moe stream: reading room = ") != nullptr) {
            g_summary = text;
        }
        if (strstr(text, "reading room") != nullptr || strstr(text, "moe-stream-room") != nullptr) {
            g_room_log += text;
        }
    }, nullptr);
    ggml_backend_load_all();

    // The fixtures' own threshold is 160 tokens (20 slips per book at 8 of 64), but these scenarios want
    // several small room batches in a 1,024-token context, the smallest batch the room takes (-ub 35) and
    // a floor under the threshold beside room floors, so they run at the old 35 through the rung's own
    // override, read at each model load; test-moe-room-model.cpp's "default threshold" unsets it.
    setenv("LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS", G_SWEEP_MIN_TOKENS, 1);

    config ref_cfg;
    ref_cfg.stream = false;
    llama_model * ref_model = load(ref_cfg);
    if (ref_model == nullptr) {
        fprintf(stderr, "failed to load %s\n", g_path.c_str());
    }
    return ref_model;
}

int finish() {
    printf("%s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}

} // namespace

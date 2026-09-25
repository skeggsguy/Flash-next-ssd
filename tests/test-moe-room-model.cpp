// The reading room (--moe-stream-room) on the 64-book fixtures, against the same model with every book
// resident: the room's promise is that a floor split between the desk and the belt gives the same bytes
// as no streaming at all, whatever happens to be on the desk. So every comparison here is memcmp, of the
// logits at every step and of every floor's MoE output (ffn_moe_out, via the eval callback) at every
// ubatch; the fixture's tiny weights leave the logits blind to a wrong book, the MoE outputs are not.
//
// Scenarios, each with the room's groups counted so a test that silently fell back to waves fails:
//   long      300 tokens read in at -ub 128 (128, 128, 44) then 8 written, parts 4 and parts 1
//   t_min     105 tokens at -ub 35, the smallest ubatch the room takes
//   empty     a desk of 62 of 64 books, so two of the four parts hold nothing (all-skip links)
//   desks     prompt B after A and after C on two fresh models: different desks, the same bytes
//   short     20 tokens, under the threshold: waves, not the room (printed against the reference)
//   poison    LLAMA_MOE_ROOM_POISON=1 fills handed-back parts with 0xFF; run WITHOUT the eval callback,
//             which would make every split synchronous and hide a part handed back too early
// Waves against the reference is printed, not asserted: their exactness is not the room's promise.

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
};

struct segment {
    std::vector<llama_token> prompt;
    int  n_write = 0;
    bool keep    = true; // outputs of this segment are compared
};

struct outputs {
    std::vector<float> logits; // every prompt position, then one row per written token
    std::vector<float> moe;    // every ffn_moe_out, in compute order
};

int g_ngl = 99;
std::string g_summary; // the last "reading room =" line print_stats wrote
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
    return llama_model_load_from_file(g_path.c_str(), mp);
}

// runs the segments on one context, clearing its memory between them (the model, and so the desk, stays)
bool run(llama_model * model, uint32_t n_ubatch, const std::vector<segment> & segs, outputs & out, bool capture) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    outputs scratch;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 1024;
    cp.n_batch  = 512;
    cp.n_ubatch = n_ubatch;
    cp.no_perf  = true;
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
            common_batch_add(batch, s.prompt[i], (llama_pos) i, { 0 }, true);
        }
        ok = ok && llama_decode(ctx, batch) == 0;
        if (ok) {
            const float * l = llama_get_logits(ctx);
            logits.assign(l, l + s.prompt.size()*n_vocab);
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
    }
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
        const outputs & ref, int64_t room_ubatches, bool capture = true) {
    llama_model * model = load(c);
    outputs got;
    const bool ok = model != nullptr && run(model, n_ubatch, segs, got, capture);
    check(ok, name + ": ran");
    if (!ok) {
        llama_model_free(model);
        return;
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

} // namespace

int main(int argc, char ** argv) {
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
        return 1;
    }
    // silent, except that the run's room summary (print_stats) is caught to check it is written
    llama_log_set([](ggml_log_level, const char * text, void *) {
        if (strstr(text, "moe stream: reading room = ") != nullptr) {
            g_summary = text;
        }
    }, nullptr);
    ggml_backend_load_all();

    config ref_cfg;
    ref_cfg.stream = false;
    llama_model * ref_model = load(ref_cfg);
    if (ref_model == nullptr) {
        fprintf(stderr, "failed to load %s\n", g_path.c_str());
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(ref_model));
    std::mt19937 rng(1234);
    const segment A = { random_tokens(rng, 300, n_vocab), 8, false };
    const segment B = { random_tokens(rng, 300, n_vocab), 8, true };
    const segment C = { random_tokens(rng, 300, n_vocab), 8, false };
    const segment D = { random_tokens(rng, 20, n_vocab), 0, false };
    const segment E = { random_tokens(rng, 20, n_vocab), 0, false };
    const segment S = { random_tokens(rng, 20, n_vocab), 4, true };
    const segment T = { random_tokens(rng, 105, n_vocab), 4, true };

    outputs ref_b, ref_s, ref_t;
    bool ok = run(ref_model, 128, { B }, ref_b, true) && run(ref_model, 128, { S }, ref_s, true) &&
              run(ref_model, 35, { T }, ref_t, true);
    llama_model_free(ref_model);
    if (!ok) {
        fprintf(stderr, "the reference run failed\n");
        return 1;
    }
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);

    config room4, room1 = room4, waves = room4, empty = room4;
    room1.parts     = 1;
    waves.room_mode = LLAMA_MOE_ROOM_OFF;
    waves.slots     = 40;
    empty.slots      = 63;                  // 2.25 MiB of room leaves 62 slots: 2 books a floor on the belt
    empty.room_mode  = LLAMA_MOE_ROOM_GIB;
    empty.room_value = 2.25f/1024.0f;

    // 300 tokens at -ub 128 is three ubatches, all over the 35-token threshold
    scenario("long, parts 4", room4, 128, { B }, ref_b, 3);
    scenario("long, parts 1", room1, 128, { B }, ref_b, 3);
    scenario("long, waves",   waves, 128, { B }, ref_b, 0);
    scenario("t_min: -ub 35", room4, 35, { T }, ref_t, 3);
    scenario("short: 20 tokens take waves", room4, 128, { S }, ref_s, 0);
    {
        llama_model * model = load(empty);
        outputs got;
        check(model != nullptr && run(model, 128, { B }, got, true), "empty parts: ran");
        size_t n_empty = 0;
        for (const auto & F : room_of(model)->floors) {
            for (const auto & p : F.parts) {
                n_empty += F.sl && p.empty();
            }
        }
        check(n_empty == 2*(size_t) n_streamed_floors(model), "empty parts: two of four parts empty on every floor");
        check(same(ref_b.logits, got.logits) && same(ref_b.moe, got.moe), "empty parts: byte-identical");
        llama_model_free(model);
    }
    {
        // the strongest check: the same prompt after different histories, so different desks
        llama_model * m1 = load(room4);
        llama_model * m2 = load(room4);
        outputs o1, o2, none;
        const bool ran = m1 && m2 && run(m1, 128, { A, D }, none, false) && run(m2, 128, { C, E }, none, false);
        const std::vector<int32_t> d1 = ran ? desk_of(m1) : std::vector<int32_t>();
        const std::vector<int32_t> d2 = ran ? desk_of(m2) : std::vector<int32_t>();
        size_t n_diff = 0;
        for (size_t i = 0; i < d1.size() && i < d2.size(); i++) {
            n_diff += d1[i] != d2[i];
        }
        check(ran && n_diff > 0, "desks: after A and after C the desks differ (" + std::to_string(n_diff) + " slots)");
        check(ran && run(m1, 128, { B }, o1, true) && run(m2, 128, { B }, o2, true), "desks: B ran on both");
        check(same(o1.logits, o2.logits) && same(o1.moe, o2.moe), "desks: B after A == B after C, byte for byte");
        check(same(ref_b.logits, o1.logits) && same(ref_b.moe, o1.moe), "desks: and == no streaming");
        llama_model_free(m1);
        llama_model_free(m2);
    }
    {
        // poison: a part handed back while a GEMM still reads it would feed that GEMM 0xFF books
        setenv("LLAMA_MOE_ROOM_POISON", "1", 1);
        scenario("poison, no eval callback, parts 4", room4, 128, { B }, ref_b, 3, false);
        scenario("poison, no eval callback, parts 1", room1, 128, { B }, ref_b, 3, false);
        unsetenv("LLAMA_MOE_ROOM_POISON");
    }

    printf("%s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}

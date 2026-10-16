// Desk books first (study #39, LLAMA_MOE_STREAM_DESK_FIRST, src/llama-moe-stream-desk-first.h) on the 64-book
// fixtures. The floor stop no longer waits for its trips: the GPU runs the books already on the desk (and the shared
// expert) while a gate waits, then the arrivals. That must change no word, so every check is byte for byte against
// no streaming at all: the logits and every floor's MoE output (ffn_moe_out). The runners are slowed (3 ms before
// each read, test_land_delay_us) so trips are still on their way at the gate, and with the lookahead off
// (LOOKAHEAD=0) the desk does not depend on timing, so off and on must also leave the same desk. The workload: a
// 20-token prompt (waves: desk books first is off), 48 single-token writes, 6 checks of 4 tokens (the apprentice's
// shape) and one of 16 (more books than the desk holds: waves again). Off is "0", on "1"; unset is on (the default
// since #39 kept). Checked:
//   same    off, on and unset give the reference's bytes, with and without the eval callback (which splits the graph
//           at every ffn_moe_out), and with the lookahead at its default width (its guesses held for the gate)
//   ran     on and unset: every one-wave batch's floors stopped desk-first, the 16-token check's did not; some split
//           (books on both planes), the gates really waited, and Metal ran one gate per stop, the cut made
//   desk    off and on leave the same book in every slot, the same LRU stamps, hotness and counts
//   CPU     -ngl 0 (a desk the GPU does not read): desk books first stays off and the bytes are the same
//   off     with the state log or the GPU slot check on, desk books first stays off
//   abort   a decode abandoned between a floor's stop and its gate (the abort callback fires right after the stop)
//           returns 2, and the next decode on the same context runs and gives a fresh context's bytes
// A gate that returns before the arrivals land turns "same" red (mutation, by hand).
//
// --stops: the floor-stop log, the stats lines and "0"'s off line instead (run_stops says what it checks).
// --fingerprint prints the off arm's hashes (logits, MoE outputs, desk) and nothing else. It compiles against the
// engine before desk books first too (DF_BUILT unset), so an old build of this file is the reference for the plain
// remap, whose classify moved into llama_moe_stream_classify_locked (gotchas: a same-binary test proves only the
// desk lookup). Run both on one fixture file and diff the lines.

#include "test-moe-room-model.h"
#include "../src/llama-moe-stream-stops.h"

#include <unistd.h>

#if __has_include("../src/llama-moe-stream-desk-first.h")
#define DF_BUILT 1
#endif

namespace {

const char * const ENV_DF = "LLAMA_MOE_STREAM_DESK_FIRST";
const int N_WRITE = 48, N_CHECK = 6;

void (*g_gate_counts)(int64_t *, int64_t *) = nullptr; // Metal's gate fn calls, and of those the cut ones

// the workload above; with capture, every floor's MoE output through the eval callback
bool run_df(llama_model * model, outputs & out, bool capture) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 1024;
    cp.n_batch    = 512;
    cp.n_ubatch   = 512;
    cp.op_offload = false;
    if (capture) {
        cp.cb_eval           = capture_moe_out;
        cp.cb_eval_user_data = &out;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        return false;
    }
    std::mt19937 rng(11);
    const std::vector<llama_token> toks = random_tokens(rng, 20 + N_CHECK*4 + 16, n_vocab);
    llama_batch batch = llama_batch_init(512, 0, 1);
    bool ok = true;
    llama_pos pos = 0;
    auto decode = [&](std::vector<llama_token>::const_iterator b, std::vector<llama_token>::const_iterator e) {
        common_batch_clear(batch);
        for (auto it = b; it != e; ++it) {
            common_batch_add(batch, *it, pos++, { 0 }, true);
        }
        ok = ok && llama_decode(ctx, batch) == 0;
        if (ok) {
            const float * l = llama_get_logits(ctx);
            out.logits.insert(out.logits.end(), l, l + (e - b)*n_vocab);
        }
    };
    decode(toks.begin(), toks.begin() + 20);
    for (int i = 0; ok && i < N_WRITE; i++) {
        const float * row = out.logits.data() + out.logits.size() - n_vocab;
        llama_token best = 0;
        for (int v = 1; v < n_vocab; v++) {
            best = row[v] > row[best] ? v : best;
        }
        const std::vector<llama_token> one = { best };
        decode(one.begin(), one.end());
    }
    for (int c = 0; ok && c < N_CHECK; c++) {
        decode(toks.begin() + 20 + 4*c, toks.begin() + 24 + 4*c);
    }
    decode(toks.end() - 16, toks.end());
    llama_batch_free(batch);
    llama_free(ctx);
    return ok;
}

// everything the desk decides with, floor by floor (as test-moe-stream-c2.cpp's)
std::vector<int64_t> desk_state_of(llama_model * model) {
    llama_moe_stream * ms = model->moe_stream();
    std::vector<int64_t> d;
    for (const int32_t s : desk_of(model)) {
        d.push_back(s);
    }
    std::unique_lock<std::mutex> lk(ms->mtx);
    for (const auto & sl : ms->layers) {
        if (sl) {
            d.insert(d.end(), sl->slot_last_use.begin(), sl->slot_last_use.end());
            d.insert(d.end(), sl->route_hotness.begin(), sl->route_hotness.end());
            d.insert(d.end(), { sl->use_counter, sl->n_hit, sl->n_miss, sl->n_miss_cold });
        }
    }
    d.insert(d.end(), { ms->stats.n_calls, ms->stats.n_hit, ms->stats.n_miss, ms->stats.n_hit_ready,
            ms->stats.n_miss_cold });
    return d;
}

struct df_counts {
    int64_t stops = 0, split = 0, desk = 0, arr = 0, waits = 0, wait_us = 0, calls = 0, cut = 0, guesses = 0,
            hit_loading = 0;
};

std::string str(const df_counts & c) {
    return "stops " + std::to_string(c.stops) + ", split " + std::to_string(c.split) + ", picks desk/arriving " +
           std::to_string(c.desk) + "/" + std::to_string(c.arr) + ", gate waits " + std::to_string(c.waits) + " (" +
           std::to_string(c.wait_us) + " us), Metal gates " + std::to_string(c.calls) + " (cut " +
           std::to_string(c.cut) + "), guesses " + std::to_string(c.guesses) + ", hit loading " +
           std::to_string(c.hit_loading);
}

struct arm {
    bool ran = false;
    outputs out;
    std::vector<int64_t> desk;
    df_counts n;
    int64_t floors = 0; // streamed floors
};

// a fresh streamed model of config c, the switch at df ("0", "1", or nullptr unset), runners slowed by delay_us
arm run_arm(const config & c, const char * df, int64_t delay_us, bool capture, bool print = false) {
    arm a;
    df != nullptr ? setenv(ENV_DF, df, 1) : unsetenv(ENV_DF);
    llama_model * m = load(c);
    if (m == nullptr) {
        unsetenv(ENV_DF);
        return a;
    }
    llama_moe_stream * ms = m->moe_stream();
    ms->test_land_delay_us = delay_us;
    int64_t calls0 = 0, cut0 = 0;
    if (g_gate_counts) {
        g_gate_counts(&calls0, &cut0);
    }
    a.ran  = run_df(m, a.out, capture); // the context reads the switch when it opens
    unsetenv(ENV_DF);
    a.desk   = desk_state_of(m);
    a.floors = n_streamed_floors(m);
    std::unique_lock<std::mutex> lk(ms->mtx);
#ifdef DF_BUILT
    a.n.stops   = ms->stats.n_df_stops;
    a.n.split   = ms->stats.n_df_split;
    a.n.desk    = ms->stats.n_df_rows_desk;
    a.n.arr     = ms->stats.n_df_rows_arr;
    a.n.waits   = ms->stats.n_df_gate_waits;
    a.n.wait_us = ms->stats.t_df_gate_wait_us;
#endif
    a.n.guesses     = ms->stats.n_preload_issued;
    a.n.hit_loading = ms->stats.n_hit_loading;
    lk.unlock();
    if (g_gate_counts) {
        g_gate_counts(&a.n.calls, &a.n.cut);
        a.n.calls -= calls0;
        a.n.cut   -= cut0;
    }
    if (print) {
        llama_moe_stream_print_stats(m);
    }
    llama_model_free(m);
    return a;
}

uint64_t fnv(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ ((const uint8_t *) p)[i])*1099511628211ULL;
    }
    return h;
}

std::string g_cblog;           // --stops: the timeline's path; its .stops log is the floor-stop log
std::string g_df_window, g_df_run; // the "desk first:" stats lines: every window's, and the run's last
int g_df_off = 0;                  // the "off" lines a context turned off by "0" prints

void remove_cblog() { // registered before the stops log's own exit handler, so it runs after the write
    std::filesystem::remove(g_cblog);
    std::filesystem::remove(g_cblog + ".stops");
}

// --stops: the floor-stop log (GGML_METAL_CBLOG, llama-moe-stream-stops.h) with the stats dump every 1 ms. Desk-first
// stops put their lines at their gates (the held ones) or at once: off and on must put as many. The stats lines say
// what the counters say, per window and for the run.
int run_stops(const config & c, int64_t delay) {
    llama_moe_stream_stops * st = llama_moe_stream_stops::get();
    check(st != nullptr, "the floor-stop log is on");
    if (st == nullptr) {
        return finish();
    }
    llama_log_set([](ggml_log_level, const char * text, void *) {
        if (strstr(text, "maybe_dump_stats_locked: moe stream: desk first:") != nullptr) {
            g_df_window += text;
        } else if (strstr(text, "print_stats: moe stream: desk first:") != nullptr) {
            g_df_run = text;
        } else if (strstr(text, "moe stream: desk first: off (LLAMA_MOE_STREAM_DESK_FIRST=0)") != nullptr) {
            g_df_off++; // "0" says it is off, once a context
        }
    }, nullptr);
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD", "0", 1);
    setenv("LLAMA_MOE_STREAM_STATS_MS", "1", 1);
    const size_t r0 = st->n_records();
    const arm off = run_arm(c, "0", delay, false);
    const size_t r1 = st->n_records();
    const arm on = run_arm(c, "1", delay, false, true);
    const size_t r2 = st->n_records();
    unsetenv("LLAMA_MOE_STREAM_STATS_MS"), unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD");
    printf("  stop lines: off %zu, on %zu\n  on: %s\n  the run's line: %s", r1 - r0, r2 - r1, str(on.n).c_str(),
            g_df_run.c_str());
    check(off.ran && on.ran && same(off.out.logits, on.out.logits), "stops log on: off and on ran, the same logits");
    check(r1 > r0 && r2 - r1 == r1 - r0, "every floor stop put its line, desk-first ones too (held ones at the gate)");
    check(on.n.stops > 0 && on.n.split > 0, "on: desk-first stops, some split");
    check(g_df_off == 1, "\"0\": one off line, from the off arm's one context (" + std::to_string(g_df_off) + ")");
    check(!g_df_window.empty() && g_df_window.find("desk first: floors ") != std::string::npos,
            "the stats dump's windows carry the desk-first line");
    const std::string want = "desk first: floors " + std::to_string(on.n.stops) + ", split " + std::to_string(on.n.split) +
            " | picks desk/arriving " + std::to_string(on.n.desk) + "/" + std::to_string(on.n.arr) + " | gate waits " +
            std::to_string(on.n.waits) + ", ";
    check(g_df_run.find(want) != std::string::npos, "the run's line gives the counters");
    return finish();
}

#ifdef DF_BUILT
// the abort callback fires once, right after the first desk-first stop: the graph ends there and the gate never runs
struct abort_once { llama_moe_stream * ms; bool fired = false; };

void check_abandoned(const config & c) {
    setenv(ENV_DF, "1", 1);
    llama_model * m = load(c);
    unsetenv(ENV_DF);
    check(m != nullptr, "abandon: model loaded");
    if (m == nullptr) { return; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256, cp.n_batch = cp.n_ubatch = 64, cp.op_offload = false;
    setenv(ENV_DF, "1", 1);
    llama_context * ctx = llama_init_from_model(m, cp);
    llama_context * ref = llama_init_from_model(m, cp);
    unsetenv(ENV_DF);
    abort_once ab = { m->moe_stream() };
    check(ctx != nullptr && ref != nullptr, "abandon: contexts opened");
    if (ctx != nullptr && ref != nullptr) {
        llama_set_abort_callback(ctx, [](void * d) {
            auto * a = (abort_once *) d;
            const bool go = !a->fired && a->ms->stats.n_df_stops > 0;
            a->fired = a->fired || go;
            return go;
        }, &ab);
        llama_batch batch = llama_batch_init(1, 0, 1);
        auto one = [&](llama_context * x) { common_batch_clear(batch); common_batch_add(batch, 5, 0, { 0 }, true);
                return llama_decode(x, batch); };
        const int r_abort = one(ctx);
        llama_set_abort_callback(ctx, nullptr, nullptr);
        check(ab.fired && r_abort == 2, "abandon: the decode was aborted between a stop and its gate");
        llama_memory_clear(llama_get_memory(ctx), true);
        const int r_next = one(ctx); // a stale armed floor aborts the process here
        const int r_ref  = one(ref);
        check(r_next == 0 && r_ref == 0 && memcmp(llama_get_logits(ctx), llama_get_logits(ref), n_vocab*sizeof(float)) == 0,
                "abandon: the next decode on the same context ran, with a fresh context's logits byte for byte");
        llama_batch_free(batch);
    }
    llama_free(ctx);
    llama_free(ref);
    llama_model_free(m);
}
#endif

} // namespace

int main(int argc, char ** argv) {
    bool fingerprint = false, stops = false;
    for (int i = 1; i < argc; i++) {
        fingerprint = fingerprint || strcmp(argv[i], "--fingerprint") == 0;
        stops       = stops || strcmp(argv[i], "--stops") == 0;
    }
    if (stops) { // before anything opens the log
        g_cblog = (std::filesystem::temp_directory_path() /
                ("test-moe-stream-desk-first-" + std::to_string(getpid()) + ".cblog")).string();
        setenv("GGML_METAL_CBLOG", g_cblog.c_str(), 1);
        unsetenv("GGML_METAL_CBLOG_SIDECARS");
        std::atexit(remove_cblog);
    }
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    if (ggml_backend_reg_t reg = ggml_backend_reg_by_name("MTL")) {
        g_gate_counts = (void (*)(int64_t *, int64_t *)) ggml_backend_reg_get_proc_address(reg,
                "ggml_backend_metal_moe_gate_counts");
    }
    const bool gpu = g_ngl > 0 && ggml_backend_reg_by_name("MTL") != nullptr;
    config c; // 55 slots with the room: a desk of 40 of 64 books a floor, so most floors take trips
    const int64_t delay = 3000;

    if (stops) {
        llama_model_free(ref_model);
        return run_stops(c, delay);
    }
    if (fingerprint) { // the plain remap's bytes and desk, for an old build of this file to match
        setenv("LLAMA_MOE_STREAM_LOOKAHEAD", "0", 1);
        const arm off = run_arm(c, "0", delay, true);
        unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD");
        const arm la = run_arm(c, "0", delay, true);
        printf("fingerprint ran %d %d\n", off.ran, la.ran);
        printf("fingerprint logits %016llx moe %016llx desk %016llx\n",
                (unsigned long long) fnv(off.out.logits.data(), off.out.logits.size()*sizeof(float)),
                (unsigned long long) fnv(off.out.moe.data(), off.out.moe.size()*sizeof(float)),
                (unsigned long long) fnv(off.desk.data(), off.desk.size()*sizeof(int64_t)));
        printf("fingerprint lookahead logits %016llx moe %016llx\n",
                (unsigned long long) fnv(la.out.logits.data(), la.out.logits.size()*sizeof(float)),
                (unsigned long long) fnv(la.out.moe.data(), la.out.moe.size()*sizeof(float)));
        llama_model_free(ref_model);
        return 0;
    }

    outputs ref, ref_plain;
    check(run_df(ref_model, ref, true), "no streaming: ran");
    check(run_df(ref_model, ref_plain, false), "no streaming, no eval callback: ran");
    check(nonzero(ref.logits) && nonzero(ref.moe), "the reference logits and MoE outputs are not all zero");
    llama_model_free(ref_model);

    setenv("LLAMA_MOE_STREAM_LOOKAHEAD", "0", 1);
    const arm off = run_arm(c, "0", delay, true);
    const arm on  = run_arm(c, "1", delay, true);
    const arm on_plain = run_arm(c, "1", delay, false);
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD");
    const arm on_la = run_arm(c, "1", delay, true); // the lookahead at its default width
    const arm dflt  = run_arm(c, nullptr, delay, false); // the switch unset: the engine's default (on, since #39 kept)
    // what it is off with, read at model load: the state log, and the GPU slot servicer (at its verify stage)
    const std::string slog = (std::filesystem::temp_directory_path() /
            ("test-moe-stream-desk-first-" + std::to_string(getpid()) + ".statelog")).string();
    setenv("LLAMA_MOE_STREAM_STATE_LOG", slog.c_str(), 1);
    const arm on_slog = run_arm(c, "1", 0, false);
    unsetenv("LLAMA_MOE_STREAM_STATE_LOG");
    std::filesystem::remove(slog);
    setenv("LLAMA_MOE_STREAM_GPU_SLOT", "1", 1);
    const arm on_slot = gpu ? run_arm(c, "1", 0, false) : on_slog; // a CPU desk refuses GPU_SLOT at load
    unsetenv("LLAMA_MOE_STREAM_GPU_SLOT");
    check(on_slog.ran && on_slog.n.stops == 0 && on_slog.n.calls == 0 && same(ref_plain.logits, on_slog.out.logits),
            "on with the state log: desk books first stays off, the reference's logits");
    check(on_slot.ran && on_slot.n.stops == 0 && on_slot.n.calls == 0 && same(ref_plain.logits, on_slot.out.logits),
            "on with the GPU slot check (GPU_SLOT=1): desk books first stays off, the reference's logits");

    printf("  off: %s\n  on: %s\n  on, no eval callback: %s\n  on, lookahead: %s\n", str(off.n).c_str(),
            str(on.n).c_str(), str(on_plain.n).c_str(), str(on_la.n).c_str());
    check(off.ran && on.ran && on_plain.ran && on_la.ran && dflt.ran, "every arm ran");
    check(same(ref_plain.logits, dflt.out.logits), "unset: logits byte-identical to no streaming");
    check(same(ref.logits, off.out.logits) && same(ref.moe, off.out.moe), "off: logits and every ffn_moe_out are the reference's");
    check(off.n.stops == 0 && off.n.calls == 0, "off: no desk-first stop, no gate");
    check(same(ref.logits, on.out.logits), "on: logits byte-identical to no streaming");
    check(same(ref.moe, on.out.moe), "on: every ffn_moe_out byte-identical to no streaming");
    check(same(ref_plain.logits, on_plain.out.logits), "on, no eval callback: logits byte-identical to no streaming");
    check(same(ref.logits, on_la.out.logits) && same(ref.moe, on_la.out.moe),
            "on, lookahead: logits and every ffn_moe_out byte-identical to no streaming");
    check(!off.desk.empty() && off.desk == on.desk, "on: the same desk as off (books, LRU stamps, hotness, counts)");

    const int64_t floors  = on.floors;
    check(floors > 0, "the streamed model has streamed floors (" + std::to_string(floors) + ")");
    const int64_t m_stops = gpu ? (int64_t) (N_WRITE + N_CHECK)*floors : 0;
    if (gpu) {
        // the stops log's and the gates' counts are per floor stop: writes and 4-token checks, not the prompt
        // (waves) or the 16-token check (64 books > a desk of 40: waves)
        for (const arm * a : { &on, &on_plain, &on_la, &dflt }) { // unset among them: on by default
            check(a->n.stops == m_stops, "on: every one-wave batch's floors stopped desk-first (" +
                    std::to_string(a->n.stops) + " of " + std::to_string(m_stops) + ")");
            check(a->n.calls == a->n.stops, "on: Metal ran one gate per desk-first stop");
        }
        check(on.n.desk + on.n.arr == (int64_t) (N_WRITE + N_CHECK*4)*8*floors,
                "on: every pick on exactly one plane (desk + arriving = picks)");
        check(on.n.split > 0 && on.n.desk > 0 && on.n.arr > 0, "on: floors split, picks on both planes");
        check(on.n.waits > 0 && on.n.wait_us > 0, "on: the gates really waited for trips");
        check(on_plain.n.cut > 0 && on_plain.n.cut == on_plain.n.calls,
                "on, no eval callback: every gate cut its command buffers (the desk's books sent first)");
        check(on_la.n.guesses > 0 && on_la.n.split > 0, "on, lookahead: guesses sent beside split floors");
    } else {
        check(on.n.stops == 0 && on.n.calls == 0 && on_la.n.stops == 0 && dflt.n.stops == 0,
                "CPU desk: desk books first stays off (no stop, no gate)");
        check(on.desk == off.desk, "CPU desk: the same desk as off");
    }
#ifdef DF_BUILT
    if (gpu) {
        check_abandoned(c);
    }
#endif
    return finish();
}

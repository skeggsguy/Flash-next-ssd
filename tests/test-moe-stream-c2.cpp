// C2, cheaper floor stops (src/llama-moe-stream-quick.cpp), on the 64-book fixtures: each switch only changes
// how long a floor stop takes, never a word, so the check is byte for byte. A short prompt, single-token writes
// and a few 4-token checks (the apprentice's shape) are run on fresh streamed models with every switch off, every
// switch on, and each on alone, and every run gives the logits and every floor's MoE output of no streaming at
// all. Each run also says its new path ran (the quick path's and the lookahead's lock-free calls, the one-lookup
// calls, the pings), or it could pass by doing nothing. Then, with the lookahead off so the desk does not depend
// on timing, all off and all on leave the same desk: the same book in every slot, the same LRU stamps, the same
// route hotness and the same hit and trip counts.

#include "test-moe-room-model.h"


namespace {

const char * const ENV_NOLOCK = "LLAMA_MOE_STREAM_NOLOCK";
const char * const ENV_ONE    = "LLAMA_MOE_STREAM_ONE_LOOKUP";
const char * const ENV_AWAKE  = "LLAMA_MOE_STREAM_KEEP_AWAKE";

// the switches for one model load: nullptr leaves the variable unset (the default: on), "0" turns it off
struct switches {
    const char * nolock;
    const char * one;
    const char * awake;
};

void set_or_unset(const char * name, const char * v) {
    if (v == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, v, 1);
    }
}

bool run_c2(llama_model * model, outputs & out, bool capture = true);

// the Metal ping counter (ggml-metal-awake.m), or null without Metal
int64_t (*g_ping_count)() = nullptr;
// the Metal encode-ahead counts (ggml-metal-context.m): graphs encoded ahead, and of those committed
void (*g_ahead_counts)(int64_t *, int64_t *) = nullptr;

int64_t ahead_encoded() {
    int64_t e = 0, u = 0;
    if (g_ahead_counts) {
        g_ahead_counts(&e, &u);
    }
    return e;
}

int64_t ahead_used() {
    int64_t e = 0, u = 0;
    if (g_ahead_counts) {
        g_ahead_counts(&e, &u);
    }
    return u;
}

// GGML_METAL_ENCODE_AHEAD, read when a context makes its Metal backend: the workload without the eval callback on
// a fresh streamed model (every C2 switch at its default), logits only, and how many graphs were encoded ahead
// and committed
bool run_ahead(const config & c, const char * ahead, outputs & out, int64_t & encoded, int64_t & used) {
    llama_model * m = load(c);
    if (m == nullptr) {
        return false;
    }
    if (ahead == nullptr) {
        unsetenv("GGML_METAL_ENCODE_AHEAD");
    } else {
        setenv("GGML_METAL_ENCODE_AHEAD", ahead, 1);
    }
    const int64_t e0 = ahead_encoded(), u0 = ahead_used();
    const bool ok = run_c2(m, out, false);
    encoded = ahead_encoded() - e0;
    used    = ahead_used() - u0;
    unsetenv("GGML_METAL_ENCODE_AHEAD");
    llama_model_free(m);
    return ok;
}

int64_t pings() {
    return g_ping_count ? g_ping_count() : 0;
}

llama_model * load_with(const config & c, const switches & sw) {
    set_or_unset(ENV_NOLOCK, sw.nolock);
    set_or_unset(ENV_ONE,    sw.one);
    set_or_unset(ENV_AWAKE,  sw.awake);
    llama_model * m = load(c);
    unsetenv(ENV_NOLOCK);
    unsetenv(ENV_ONE);
    unsetenv(ENV_AWAKE);
    if (m != nullptr && m->moe_stream() && m->moe_stream()->keep_awake_us > 0) {
        m->moe_stream()->keep_awake_us = 1; // every wait on a trip pings, so a short fixture run sends some
    }
    return m;
}

// a 20-token prompt, 64 single-token writes (greedy), then 8 checks of 4 random tokens each
// with capture, every floor's MoE output too, through the eval callback (which turns encoding ahead off: the
// scheduler then computes and synchronizes node by node)
bool run_c2(llama_model * model, outputs & out, bool capture) {
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
    std::mt19937 rng(7);
    const std::vector<llama_token> toks = random_tokens(rng, 20 + 8*4, n_vocab);
    llama_batch batch = llama_batch_init(512, 0, 1);
    bool ok = true;
    llama_pos pos = 0;
    auto decode = [&](const std::vector<llama_token> & t) {
        common_batch_clear(batch);
        for (const llama_token x : t) {
            common_batch_add(batch, x, pos++, { 0 }, true);
        }
        ok = ok && llama_decode(ctx, batch) == 0;
        if (ok) {
            const float * l = llama_get_logits(ctx);
            out.logits.insert(out.logits.end(), l, l + t.size()*n_vocab);
        }
    };
    decode(std::vector<llama_token>(toks.begin(), toks.begin() + 20));
    for (int i = 0; ok && i < 64; i++) {
        const float * row = out.logits.data() + out.logits.size() - n_vocab;
        llama_token best = 0;
        for (int v = 1; v < n_vocab; v++) {
            best = row[v] > row[best] ? v : best;
        }
        decode({ best });
    }
    for (int b = 0; ok && b < 8; b++) {
        decode(std::vector<llama_token>(toks.begin() + 20 + 4*b, toks.begin() + 24 + 4*b));
    }
    llama_batch_free(batch);
    llama_free(ctx);
    return ok;
}

struct counts {
    int64_t quick = 0, la_quick = 0, one = 0, awake = 0, pinged = 0, preloads = 0;
};

counts counts_of(llama_model * model, int64_t pings0) {
    llama_moe_stream * ms = model->moe_stream();
    std::unique_lock<std::mutex> lk(ms->mtx);
    counts c;
    c.quick    = ms->n_quick;
    c.la_quick = ms->n_la_quick;
    c.one      = ms->n_one_lookup;
    c.awake    = ms->n_awake;
    c.pinged   = pings() - pings0;
    c.preloads = ms->stats.n_preload_issued;
    return c;
}

std::string str(const counts & c) {
    return "quick " + std::to_string(c.quick) + ", lookahead quick " + std::to_string(c.la_quick) + ", one lookup " +
           std::to_string(c.one) + ", awake " + std::to_string(c.awake) + ", pings " + std::to_string(c.pinged) +
           ", guesses " + std::to_string(c.preloads);
}

// runs the workload on a fresh model with these switches, checks it against the reference, returns the counts
// slow runners: each runner sleeps 3 ms before its read (a test-only knob: a slow drive), so a
// floor's guesses are still on their way when it asks for them. The quick path, which takes no lock, must then
// see them as not on the desk and leave the floor to the locked path, which waits for them.
counts scenario_c2(const std::string & name, const config & cfg, const switches & sw, const outputs & ref,
        int64_t * kept = nullptr, int64_t * hit_loading = nullptr) {
    counts c;
    llama_model * model = load_with(cfg, sw);
    check(model != nullptr, name + ": loaded");
    if (model == nullptr) {
        return c;
    }
    const int64_t p0 = pings();
    outputs got;
    if (hit_loading != nullptr) {
        model->moe_stream()->test_land_delay_us = 3000;
    }
    check(run_c2(model, got), name + ": ran");
    c = counts_of(model, p0);
    if (hit_loading != nullptr) {
        *hit_loading = model->moe_stream()->stats.n_hit_loading;
    }
    if (kept != nullptr) {
        *kept = room_of(model) ? room_of(model)->lstats.n_kept : 0;
    }
    llama_model_free(model);
    check(same(ref.logits, got.logits), name + ": logits byte-identical to no streaming");
    check(same(ref.moe, got.moe), name + ": every ffn_moe_out byte-identical to no streaming");
    printf("  %s: %s\n", name.c_str(), str(c).c_str());
    return c;
}

// everything the desk decides with, floor by floor
struct desk_state {
    std::vector<int32_t> slots;
    std::vector<int64_t> last_use;
    std::vector<uint32_t> hot;
    std::vector<int64_t> counters;
};

desk_state desk_state_of(llama_model * model) {
    llama_moe_stream * ms = model->moe_stream();
    desk_state d;
    d.slots = desk_of(model);
    std::unique_lock<std::mutex> lk(ms->mtx);
    for (const auto & sl : ms->layers) {
        if (sl) {
            d.last_use.insert(d.last_use.end(), sl->slot_last_use.begin(), sl->slot_last_use.end());
            d.hot.insert(d.hot.end(), sl->route_hotness.begin(), sl->route_hotness.end());
            d.counters.insert(d.counters.end(), { sl->use_counter, sl->n_hit, sl->n_miss, sl->n_miss_cold });
        }
    }
    d.counters.insert(d.counters.end(), { ms->stats.n_calls, ms->stats.n_hit, ms->stats.n_miss,
            ms->stats.n_hit_ready, ms->stats.n_hit_loading, ms->stats.n_miss_cold });
    return d;
}

} // namespace

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    if (ggml_backend_reg_t reg = ggml_backend_reg_by_name("MTL")) {
        g_ping_count = (int64_t (*)()) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_keep_awake_count");
        g_ahead_counts = (void (*)(int64_t *, int64_t *)) ggml_backend_reg_get_proc_address(reg,
                "ggml_backend_metal_encode_ahead_counts");
    }
    const bool gpu = g_ngl > 0 && g_ping_count != nullptr; // a desk the GPU reads: the pings are sent

    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD"); // the lookahead at its default width, as in every rung
    outputs ref;
    check(run_c2(ref_model, ref), "no streaming: ran");
    check(nonzero(ref.logits) && nonzero(ref.moe), "the reference logits and MoE outputs are not all zero");

    // the room and the lent belt, on a desk big enough that many floors find every book on it (the fixture's
    // routing is near uniform, so on the harness's 40-slot desk a floor almost never does) and small enough
    // that others still take trips
    config c;
    c.slots = 60;
    const switches off = { "0", "0", "0" }, on = { nullptr, nullptr, nullptr };

    const counts c_off = scenario_c2("all off", c, off, ref);
    check(c_off.quick == 0 && c_off.la_quick == 0 && c_off.one == 0 && c_off.awake == 0 && c_off.pinged == 0,
            "all off: no new path ran");

    const counts c_on = scenario_c2("all on", c, on, ref);
    check(c_on.quick > 0 && c_on.la_quick > 0 && c_on.one > 0, "all on: the quick path, the lookahead's and one lookup ran");
    check(gpu ? c_on.awake > 0 && c_on.pinged == c_on.awake : c_on.awake == 0 && c_on.pinged == 0,
            gpu ? "all on: floors waiting on a trip pinged the GPU" : "all on: a CPU desk sends no pings");

    const counts c_lock = scenario_c2("NOLOCK alone", c, { nullptr, "0", "0" }, ref);
    check(c_lock.quick > 0 && c_lock.la_quick > 0 && c_lock.one == 0 && c_lock.awake == 0, "NOLOCK alone: only it ran");

    // The lookahead's lock-free skip must only skip guesses the issue loop would skip anyway. A wrong skip changes
    // no word, so only the number of guesses issued can see it. That number depends on timing in general (a slot
    // still loading is never a victim), but on a desk of 60 of 64 books with the fixture's fast reads it has been
    // the same in every run (36 of 36 while this test was written); a mismatch here is worth a second look either way.
    check(c_lock.preloads == c_off.preloads && c_on.preloads == c_off.preloads,
            "NOLOCK: the lookahead issued the same guesses as with it off");

    const counts c_one = scenario_c2("ONE_LOOKUP alone", c, { "0", nullptr, "0" }, ref);
    check(c_one.quick == 0 && c_one.la_quick == 0 && c_one.one > 0 && c_one.awake == 0, "ONE_LOOKUP alone: only it ran");

    const counts c_awake = scenario_c2("KEEP_AWAKE alone", c, { "0", "0", nullptr }, ref);
    check(c_awake.quick == 0 && c_awake.one == 0 && (gpu ? c_awake.awake > 0 : c_awake.awake == 0),
            "KEEP_AWAKE alone: only it ran");

    config cd = c;
    cd.direct = true; // the study's path: staged reads, the lent belt's gate before the upload
    int64_t kept = 0;
    const counts c_direct = scenario_c2("all on, direct reads", cd, on, ref, &kept);
    check(c_direct.quick > 0 && c_direct.la_quick > 0 && c_direct.one > 0 && (!gpu || c_direct.awake > 0),
            "all on, direct reads: every new path ran");
    check(kept > 0, "all on, direct reads: the lent belt kept books (" + std::to_string(kept) + ")");

    int64_t hit_loading = 0;
    // guess every book of the next floor: the few not on the desk are then on their way when it asks for them,
    // beside books that are on it (at the default width the fixture's guesses are rarely right)
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD", "64", 1);
    const counts c_slow = scenario_c2("all on, slow runners", c, on, ref, nullptr, &hit_loading);
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD");
    check(c_slow.quick > 0 && hit_loading > 0, "all on, slow runners: the quick path ran and floors found guesses on their way (" +
            std::to_string(hit_loading) + ")");

    // encoding ahead: the GPU split after each floor's remap is encoded while the GPU runs the one before it
    {
        outputs ref_plain, a_off, a_on;
        int64_t enc_off = 0, used_off = 0, enc_on = 0, used_on = 0;
        check(run_c2(ref_model, ref_plain, false), "no streaming, no eval callback: ran");
        check(same(ref_plain.logits, ref.logits), "no streaming: the eval callback changes no logit");
        check(run_ahead(c, "0", a_off, enc_off, used_off), "encode ahead off: ran");
        check(run_ahead(c, nullptr, a_on, enc_on, used_on), "encode ahead on: ran");
        printf("  encode ahead: off %lld encoded, %lld committed; on %lld encoded, %lld committed\n",
                (long long) enc_off, (long long) used_off, (long long) enc_on, (long long) used_on);
        check(same(a_off.logits, ref.logits), "encode ahead off: logits byte-identical to no streaming");
        check(same(a_on.logits, ref.logits), "encode ahead on: logits byte-identical to no streaming");
        check(enc_off == 0 && used_off == 0, "encode ahead off: nothing encoded ahead");
        check(gpu ? used_on > 0 && used_on == enc_on : enc_on == 0,
                gpu ? "encode ahead on: graphs encoded ahead, every one committed" : "encode ahead on: no GPU split, nothing ahead");
    }

    // the desk: no lookahead (its guesses land when they land), no room, a desk of 56 of 64 books a floor
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD", "0", 1);
    config cq;
    cq.room_mode = LLAMA_MOE_ROOM_OFF;
    cq.slots     = 56;
    desk_state d_off, d_on;
    outputs o_off, o_on;
    counts k_on;
    for (int pass = 0; pass < 2; pass++) {
        llama_model * m = load_with(cq, pass == 0 ? off : on);
        check(m != nullptr, pass == 0 ? "desk, all off: loaded" : "desk, all on: loaded");
        if (m == nullptr) {
            continue;
        }
        const int64_t p0 = pings();
        check(run_c2(m, pass == 0 ? o_off : o_on), pass == 0 ? "desk, all off: ran" : "desk, all on: ran");
        (pass == 0 ? d_off : d_on) = desk_state_of(m);
        if (pass == 1) {
            k_on = counts_of(m, p0);
        }
        llama_model_free(m);
    }
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD");
    printf("  desk, all on: %s\n", str(k_on).c_str());
    check(k_on.quick > 0 && k_on.one > 0, "desk, all on: the quick path and one lookup ran");
    check(same(o_off.logits, o_on.logits) && same(o_off.moe, o_on.moe), "desk: the same logits and MoE outputs");
    check(!d_off.slots.empty() && d_off.slots == d_on.slots, "desk: the same book in every slot");
    check(!d_off.last_use.empty() && d_off.last_use == d_on.last_use, "desk: the same LRU stamps");
    check(!d_off.hot.empty() && d_off.hot == d_on.hot, "desk: the same route hotness");
    check(!d_off.counters.empty() && d_off.counters == d_on.counters, "desk: the same calls, hits and trips");

    llama_model_free(ref_model);
    return finish();
}

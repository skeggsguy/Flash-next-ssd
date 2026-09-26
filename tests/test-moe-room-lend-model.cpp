// The lent belt (src/llama-moe-room-lend-ops.cpp) on the 64-book fixtures: while writing, the reading
// room's belt keeps copies of the books the writing remap and the lookahead's guesses put back, and a
// later trip for one of them (a miss, or a guess) copies it back instead of reading it. Its promise is
// that nothing but where a load's bytes come from changes, so every scenario compares byte for byte
// (logits, and every ffn_moe_out where the eval callback runs) against the same model with lending off
// (LLAMA_MOE_ROOM_LEND=0) and with no streaming (the harness: test-moe-room-model.h), and each asserts
// the belt was really lent, or it would pass by never lending. The desk and the drive's slab count are
// compared with the lookahead prefetch off (LLAMA_MOE_STREAM_LOOKAHEAD=0): with it on, a wrong guess
// still loading is skipped as a victim, so the desk's choices follow the runners' timing and two runs
// with lending off end with different desks about half the time; the bytes do not depend on it.
//   writing    300 tokens read in through the room, then 48 written with a 40-slot desk: books kept and
//              copied back; byte-identical to lending off and to no streaming; the same desk as lending
//              off; and the drive read exactly one book's slabs fewer per book copied back. The lines it
//              writes, and GPU_SLOT=3 (on the GPU) loading with lending off
//   guesses    the same with the lookahead on (the study's setting): guesses keep and are copied back
//   read-in    writing, then a second read-in and 48 more written, with LLAMA_MOE_ROOM_POISON=1 (copies
//              the room takes back are filled with 0xFF; every slab copied back is checked against a read
//              of the file) and no eval callback, which would make every split synchronous; once with
//              the lookahead off, once on with one slow runner and guesses queued right before the
//              read-in, so the take-back meets guesses still queued (and perhaps in flight)
//   desks      prompt B after A and after C (each written on, so lent), on two fresh models: B's bytes the
//              same, and the same as no streaming
//   paths      direct I/O (a staging read, then the upload the gate holds) with two runners on a twin copy,
//              against the zero-copy path of "writing" (the read lands in the slot, the gate holds the read)
//   stress     four read-ins with 48 written after each, the lookahead on and POISON, on one slow runner
//              with direct I/O and on eighteen fast ones; LLAMA_TEST_LEND_STRESS=N repeats it N times

#include "test-moe-room-model.h"

#include <algorithm>

namespace {

std::string g_lend_log; // what loading said about the lent belt, the window lines and the run's total

const char * g_lend_env = "LLAMA_MOE_ROOM_LEND";

// the harness's own log capture (setup), plus the lent belt's lines
void capture_logs() {
    llama_log_set([](ggml_log_level, const char * text, void *) {
        if (strstr(text, "moe stream: reading room = ") != nullptr) {
            g_summary = text;
        }
        if (strstr(text, "reading room") != nullptr || strstr(text, "moe-stream-room") != nullptr) {
            g_room_log += text;
        }
        if (strstr(text, "lent belt") != nullptr) {
            g_lend_log += text;
        }
    }, nullptr);
}

struct lend_run {
    outputs out;
    std::vector<int32_t> desk;
    llama_moe_lend_stats lent;
    int64_t slabs  = 0;  // slabs the runners read (desk and belt)
    size_t  n_w    = 0;  // weights a book has on these floors
    bool    ok     = false;
};

const char * g_lookahead_env = "LLAMA_MOE_STREAM_LOOKAHEAD";

// Queues guesses the way the lookahead does (llama_moe_stream_prefetch_next), without waking the runners:
// they sit in q_spec until the next read-in's first desk op takes the belt back, so the take-back meets,
// every time, a guess whose restore is still queued and a guess whose keep is still pending (with the
// fixtures' two floors the runners otherwise drain the queue before the next decode). Their books are
// real loads: the slots turn LOADING and the desk op waits for them. Under the manager's lock, as the
// remap op would be.
void queue_guesses(llama_model * model) {
    llama_moe_stream * ms   = model->moe_stream();
    llama_moe_room *   room = room_of(model);
    std::unique_lock<std::mutex> lk(ms->mtx);
    if (!room->lent_out) {
        return;
    }
    for (auto & slp : ms->layers) {
        llama_moe_stream_layer * sl = slp.get();
        if (sl == nullptr) {
            continue;
        }
        uint32_t n_kept = 0;
        std::vector<int32_t> taken; // books guessed into the desk by this call
        auto off_desk = [&](int32_t not_this) {
            for (int32_t e = 0; e < (int32_t) sl->n_expert; e++) {
                if (e != not_this && sl->expert_slot.find(e) == sl->expert_slot.end() &&
                    std::find(taken.begin(), taken.end(), e) == taken.end()) {
                    return e;
                }
            }
            return -1;
        };
        // a guess of book e into victim v, as the lookahead queues it; returns the keep's seq (0: not kept)
        auto guess = [&](int32_t e, int32_t v) {
            const uint64_t lent = room->lend_find_locked(*sl, e, true);
            const uint64_t save = room->lend_keep_locked(*sl, v, n_kept);
            ms->reserve_slot_locked(*sl, e, v);
            sl->slot_pending[v] = (uint8_t) sl->weights.size();
            for (size_t wi = 0; wi < sl->weights.size(); wi++) {
                llama_moe_stream_work w = { sl, e, v, (int32_t) wi, sl->slot_gen[v] };
                w.lent = lent;
                w.save = save;
                ms->q_spec.push_back(w);
            }
            taken.push_back(e);
            return save;
        };

        // a guess whose keep stays pending (a victim whose book has no copy yet); first, before any pin
        // below can stand in a placement's way
        for (int tries = 0; tries < 8; tries++) {
            const int32_t v = ms->pick_victim_locked(*sl, nullptr);
            const int32_t e = off_desk(-1);
            if (v < 0 || e < 0 || guess(e, v) != 0) {
                break;
            }
        }
        // a book off the desk with a complete copy on the belt: make one if there is none (keep a resident
        // book, copy it out now as the op thread's help would, and guess another book into its slot)
        int32_t y = -1;
        for (int32_t e = 0; e < (int32_t) sl->n_expert && y < 0; e++) {
            y = sl->expert_slot.find(e) == sl->expert_slot.end() && room->lend.find(sl->il, e) != nullptr ? e : -1;
        }
        if (y < 0) {
            const int32_t v = ms->pick_victim_locked(*sl, nullptr);
            const int32_t e = off_desk(-1);
            if (v < 0 || e < 0) {
                continue;
            }
            y = sl->slot_expert[v];
            room->lend_keep_locked(*sl, v, n_kept);
            room->lend_help_locked(lk);
            guess(e, v);
            y = room->lend.find(sl->il, y) != nullptr ? y : -1;
        }
        // its guess back from the belt: a restore that stays queued
        if (y >= 0) {
            const int32_t v = ms->pick_victim_locked(*sl, nullptr);
            if (v >= 0) {
                guess(y, v);
            }
        }
    }
}

// segs on a fresh model of config c, lending on or off; `between` runs between segments
lend_run run_lend(const config & c, bool lend, const std::vector<segment> & segs, bool capture,
        const std::function<void(llama_model *)> & also = {}, const std::function<void(llama_model *)> & between = {}) {
    if (lend) {
        unsetenv(g_lend_env);
    } else {
        setenv(g_lend_env, "0", 1);
    }
    lend_run r;
    llama_model * model = load(c);
    r.ok = model != nullptr && room_of(model) != nullptr &&
           run(model, 128, segs, r.out, capture, between ? std::function<void()>([&] { between(model); }) : std::function<void()>());
    if (r.ok) {
        llama_moe_stream * ms = model->moe_stream();
        r.desk  = desk_of(model);
        r.lent  = room_of(model)->lstats;
        r.slabs = ms->stats.n_slabs_read;
        for (const auto & sl : ms->layers) {
            r.n_w = r.n_w == 0 && sl ? sl->weights.size() : r.n_w;
        }
        check(room_of(model)->lend_on == lend, std::string("lending is ") + (lend ? "on" : "off") + " as asked");
        if (also) {
            also(model);
        }
    }
    llama_model_free(model);
    unsetenv(g_lend_env);
    return r;
}

std::string counts(const lend_run & r) {
    const llama_moe_lend_stats & l = r.lent;
    return std::to_string(l.n_kept) + " kept, " + std::to_string(l.n_lent) + " of " + std::to_string(l.n_miss) +
           " misses and " + std::to_string(l.n_guess_lent) + " of " + std::to_string(l.n_guess) + " guesses copied back";
}

// the lookahead on or off around a run (the study runs with it on; the desk checks need it off)
lend_run with_lookahead(bool on, const std::function<lend_run()> & f) {
    if (on) {
        unsetenv(g_lookahead_env);
    }
    const lend_run r = f();
    setenv(g_lookahead_env, "0", 1);
    return r;
}

} // namespace

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    capture_logs();
    setenv("LLAMA_MOE_STREAM_STATS_MS", "1", 1); // a stats window every millisecond, so the lent line is written
    setenv(g_lookahead_env, "0", 1);              // a desk that does not follow the runners' timing (see the top)
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(ref_model));
    std::mt19937 rng(4321);
    const segment A = { random_tokens(rng, 300, n_vocab), 32, false };
    const segment B = { random_tokens(rng, 300, n_vocab), 48, true };
    const segment C = { random_tokens(rng, 300, n_vocab), 32, false };
    const segment W = { random_tokens(rng, 300, n_vocab), 48, true };
    const std::vector<segment> S = { B, W, { A.prompt, 48, true }, { C.prompt, 48, true } }; // the stress run

    outputs ref_b, ref_bw, ref_s;
    bool ok = run(ref_model, 128, { B }, ref_b, true) && run(ref_model, 128, { B, W }, ref_bw, false) &&
              run(ref_model, 128, S, ref_s, false);
    llama_model_free(ref_model);
    if (!ok) {
        fprintf(stderr, "the reference run failed\n");
        return 1;
    }
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);

    const config room4; // 55 slots with auto: a desk of 40 of 64 books, the belt beside it
    {
        g_lend_log.clear();
        const lend_run on  = run_lend(room4, true, { B }, true, [](llama_model * model) {
            g_summary.clear();
            llama_moe_stream_print_stats(model);
        });
        const lend_run off = run_lend(room4, false, { B }, true);
        check(on.ok && off.ok, "writing: ran with lending on and off");
        check(on.lent.n_kept > 0 && on.lent.n_lent > 0, "writing: books kept and copied back (" + counts(on) + ")");
        check(off.lent.n_kept == 0 && off.lent.n_miss == 0, "writing: lending off keeps nothing");
        check(same(ref_b.logits, on.out.logits) && same(ref_b.moe, on.out.moe) && nonzero(ref_b.moe),
                "writing: logits and every ffn_moe_out byte-identical to no streaming");
        check(same(off.out.logits, on.out.logits) && same(off.out.moe, on.out.moe), "writing: and to lending off");
        check(!on.desk.empty() && on.desk == off.desk, "writing: the desk the same as with lending off");
        check(off.slabs - on.slabs == (int64_t) on.n_w*on.lent.n_lent,
                "writing: the drive read " + std::to_string(off.slabs - on.slabs) + " slabs fewer = " +
                std::to_string(on.n_w) + " x " + std::to_string(on.lent.n_lent) + " books copied back");
        check(on.lent.n_bytes_out > 0 && on.lent.n_bytes_in > 0 && on.lent.n_taken_back == 0,
                "writing: bytes copied both ways, the belt never taken back");

        const size_t at_load  = g_lend_log.find("lent belt: while writing, the belt keeps up to ~");
        const size_t at_line  = g_lend_log.find("moe stream: lent belt: ");
        const size_t at_total = g_lend_log.find("moe stream: lent belt = ");
        const size_t at_off   = g_lend_log.find("lent belt: off; LLAMA_MOE_ROOM_LEND=0");
        check(at_load != std::string::npos && at_line != std::string::npos && at_total != std::string::npos &&
              at_off != std::string::npos, "writing: the load line, the window line, the total and the off line");
        check(g_lend_log.find("reading room:") == std::string::npos &&
              g_lend_log.find("moe stream: room") == std::string::npos && g_lend_log.find("moe stream: drives") == std::string::npos,
              "writing: none of them reads as the room's lines to the study's parser");
        // printed: the load line, the window that copied the most books back, and the run's total
        size_t at_busiest = at_line;
        int    most       = -1;
        for (size_t at = at_line; at != std::string::npos; at = g_lend_log.find("moe stream: lent belt: ", at + 1)) {
            int x = 0;
            const size_t bar = g_lend_log.find("| ", at);
            if (bar != std::string::npos && sscanf(g_lend_log.c_str() + bar + 2, "%d of", &x) == 1 && x > most) {
                most       = x;
                at_busiest = at;
            }
        }
        for (const size_t at : { at_load, at_busiest, at_total }) {
            if (at != std::string::npos) {
                printf("  | %s", g_lend_log.substr(at, g_lend_log.find('\n', at) - at + 1).c_str());
            }
        }

        // GPU_SLOT=3 moves the desk's bookkeeping to the GPU, where the remap that lends never runs: off, saying why
        if (g_ngl > 0 && llama_supports_gpu_offload()) {
            setenv("LLAMA_MOE_STREAM_GPU_SLOT", "3", 1);
            g_lend_log.clear();
            llama_model * m = load(room4);
            check(m != nullptr && room_of(m) != nullptr && !room_of(m)->lend_on &&
                  g_lend_log.find("lent belt: off; LLAMA_MOE_STREAM_GPU_SLOT=3") != std::string::npos,
                  "writing, GPU_SLOT=3: the room without lending, and the load line says why");
            llama_model_free(m);
            unsetenv("LLAMA_MOE_STREAM_GPU_SLOT");
        }
    }
    {
        // the study's default: the lookahead prefetch on. Its guesses keep the books they put back and
        // are copied back from the belt too; the bytes are the same
        const lend_run la = with_lookahead(true, [&] { return run_lend(room4, true, { B }, true); });
        check(la.ok && la.lent.n_lent > 0 && la.lent.n_guess > 0 && la.lent.n_guess_lent > 0,
                "guesses: misses and guesses copied back (" + counts(la) + ")");
        check(la.ok && same(ref_b.logits, la.out.logits) && same(ref_b.moe, la.out.moe), "guesses: byte-identical to no streaming");
    }
    {
        // a read-in after lending: the room takes the belt back and reads over the copies; POISON fills them
        // with 0xFF as it does, so a copy found after that gives wrong words, and checks each slab copied
        // back against the file; no eval callback, which would make every split synchronous
        setenv("LLAMA_MOE_ROOM_POISON", "1", 1);
        const lend_run on  = run_lend(room4, true, { B, W }, false);
        const lend_run off = run_lend(room4, false, { B, W }, false);
        check(on.ok && off.ok, "read-in after lending, poison: ran");
        check(on.lent.n_taken_back == 1 && on.lent.n_cleared > 0 && on.lent.n_lent > 0,
                "read-in after lending, poison: taken back once, " + std::to_string(on.lent.n_cleared) + " copies dropped, " + counts(on));
        check(same(ref_bw.logits, on.out.logits), "read-in after lending, poison: logits byte-identical to no streaming");
        check(same(off.out.logits, on.out.logits) && on.desk == off.desk, "read-in after lending, poison: and to lending off, the same desk");
        check(off.slabs - on.slabs == (int64_t) on.n_w*on.lent.n_lent, "read-in after lending, poison: slabs fewer by the books copied back");

        // the same with the lookahead on and one runner reading with direct I/O (slow), and guesses queued
        // just before the read-in, so the take-back meets guesses still queued (their restores turn into
        // reads, their keeps are forgotten) and perhaps in flight (waited for)
        config slow = room4;
        slow.direct     = true;
        slow.io_threads = 1;
        const lend_run la = with_lookahead(true, [&] { return run_lend(slow, true, { B, W }, false, {}, queue_guesses); });
        unsetenv("LLAMA_MOE_ROOM_POISON");
        // (a slow runner's own queued restores pin copies that can block queue_guesses' keeps, so the queued
        // counts are printed here and required over the stress runs below)
        check(la.ok && la.lent.n_taken_back == 1 && la.lent.n_guess_lent > 0,
                "read-in after guesses, poison, one slow runner: taken back once, " + counts(la) + ", " +
                std::to_string(la.lent.n_guess_unlent) + " queued guesses read instead, " +
                std::to_string(la.lent.n_kept_forgotten) + " queued keeps forgotten");
        check(la.ok && same(ref_bw.logits, la.out.logits), "read-in after guesses, poison, one slow runner: byte-identical to no streaming");
    }
    {
        // B after two different histories, each written on so the belt was lent: different desks and belts,
        // the same bytes for B (and B's own writing lends too); the lookahead on, as the study runs
        const lend_run a = with_lookahead(true, [&] { return run_lend(room4, true, { A, B }, true); });
        const lend_run c = with_lookahead(true, [&] { return run_lend(room4, true, { C, B }, true); });
        size_t n_diff = 0;
        for (size_t i = 0; i < a.desk.size() && i < c.desk.size(); i++) {
            n_diff += a.desk[i] != c.desk[i];
        }
        check(a.ok && c.ok && n_diff > 0, "desks: after A and after C the desks differ (" + std::to_string(n_diff) + " slots)");
        check(a.lent.n_lent > 0 && c.lent.n_lent > 0 && a.lent.n_taken_back == 1 && c.lent.n_taken_back == 1,
                "desks: both lent, both taken back for B (" + counts(a) + "; " + counts(c) + ")");
        check(same(a.out.logits, c.out.logits) && same(a.out.moe, c.out.moe), "desks: B after A == B after C, byte for byte");
        check(same(ref_b.logits, a.out.logits) && same(ref_b.moe, a.out.moe), "desks: and == no streaming");
    }
    {
        // the study's path: direct I/O, where a book's new slab lands in staging and the gate holds its upload,
        // with two runners on a byte-identical copy (the zero-copy path, where the gate holds the read into the
        // slot itself, is "writing" above)
        config prod = room4;
        prod.direct = true;
        prod.alt    = g_path + ".lend-twin-ngl" + std::to_string(g_ngl) + ".gguf";
        std::filesystem::copy_file(g_path, prod.alt, std::filesystem::copy_options::overwrite_existing);
        const lend_run on = run_lend(prod, true, { B }, true, [](llama_model * model) {
            const llama_moe_stream * ms = model->moe_stream();
            check(ms->use_direct_io && !ms->files_alt.empty() && ms->stats.n_bytes_file > 0 && ms->stats.n_bytes_alt > 0,
                    "direct I/O, two runners: direct reads, and both runners read");
        });
        const lend_run off = run_lend(prod, false, { B }, true);
        std::filesystem::remove(prod.alt);
        check(on.ok && off.ok && on.lent.n_kept > 0 && on.lent.n_lent > 0, "direct I/O, two runners: lent (" + counts(on) + ")");
        check(same(ref_b.logits, on.out.logits) && same(ref_b.moe, on.out.moe), "direct I/O, two runners: byte-identical to no streaming");
        check(same(off.out.logits, on.out.logits) && on.desk == off.desk, "direct I/O, two runners: and to lending off, the same desk");
        check(off.slabs - on.slabs == (int64_t) on.n_w*on.lent.n_lent, "direct I/O, two runners: slabs fewer by the books copied back");
    }
    {
        // stress: the lookahead on and POISON, four read-ins each followed by writing (three take-backs, each
        // meeting guesses queued for it and whatever the runners have in flight), on one slow runner with
        // direct I/O and on eighteen fast zero-copy ones; repeated LLAMA_TEST_LEND_STRESS times (once by
        // default) and compared with no streaming every time
        const char * env   = getenv("LLAMA_TEST_LEND_STRESS");
        const int    times = env ? std::max(1, atoi(env)) : 1;
        config slow = room4, fast = room4;
        slow.direct     = true;
        slow.io_threads = 1;
        fast.io_threads = 18;
        setenv("LLAMA_MOE_ROOM_POISON", "1", 1);
        int64_t n_back = 0, n_unlent = 0, n_forgotten = 0, n_busy = 0;
        bool    all = true;
        for (int i = 0; i < times; i++) {
            for (const config * c : { &slow, &fast }) {
                const lend_run r = with_lookahead(true, [&] { return run_lend(*c, true, S, false, {}, queue_guesses); });
                const bool good = r.ok && r.lent.n_taken_back == 3 && r.lent.n_lent > 0 && r.lent.n_guess_lent > 0 &&
                                  same(ref_s.logits, r.out.logits);
                if (!good) {
                    check(false, "stress, repeat " + std::to_string(i) + (c == &slow ? ", one slow runner" : ", eighteen runners") +
                                 ": taken back " + std::to_string(r.lent.n_taken_back) + " of 3, " + counts(r) +
                                 (r.ok && same(ref_s.logits, r.out.logits) ? "" : ", logits DIFFER from no streaming"));
                }
                all         = all && good;
                n_back      += r.lent.n_lent + r.lent.n_guess_lent;
                n_unlent    += r.lent.n_guess_unlent;
                n_forgotten += r.lent.n_kept_forgotten;
                n_busy      += r.lent.n_not_kept_busy;
            }
        }
        unsetenv("LLAMA_MOE_ROOM_POISON");
        check(all && n_unlent > 0 && n_forgotten > 0,
              "stress: " + std::to_string(2*times) + " runs byte-identical to no streaming, taken back 3 times each (" +
                   std::to_string(n_back) + " books copied back, " + std::to_string(n_unlent) + " queued guesses read instead, " +
                   std::to_string(n_forgotten) + " queued keeps forgotten, " + std::to_string(n_busy) + " keeps blocked by a busy copy)");
    }
    return finish();
}

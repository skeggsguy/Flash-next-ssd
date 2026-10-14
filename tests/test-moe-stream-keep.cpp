// The keep rule's victim choices (LLAMA_MOE_STREAM_LOOKAHEAD_KEEP, src/llama-moe-stream-keep.cpp), on a desk made
// by hand, with no model and no runners: a guessed book stays on its floor's desk until the floor stops.
//
//   - a guess never takes a kept slot, and the issue loop stops when only kept slots are left;
//   - the floor's own need takes an unkept slot first and a kept one rather than wait (n_keep_forced): it gives
//     -1, and so waits, exactly when the desk without the rule would;
//   - the floor's stop (llama_moe_stream_remap_out) counts the guesses it reads and releases every kept slot,
//     read or not; a reused slot forgets its tags, and counts a kept book put back unread;
//   - off, every choice is the old rule's.
// Hand-made cases first, then seeded random floor stops checked against those promises after every step.

#include "testing.h"

#include "llama.h"

#include "../src/llama-moe-stream.h"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

// one floor with n_slots desk slots and n_expert books, sized as create_cache_tensor sizes a streamed floor; no
// weights, so a reservation queues no reads and the test settles each slot itself
struct desk {
    std::unique_ptr<llama_moe_stream> mgr;
    llama_moe_stream_layer * sl = nullptr;

    desk(uint32_t n_slots, uint32_t n_expert, bool keep_rule) {
        mgr = std::make_unique<llama_moe_stream>(/*n_layer =*/ 1, n_slots, /*n_io_threads =*/ 1, /*direct =*/ false);
        mgr->la_keep    = keep_rule;
        mgr->one_lookup = false; // the stop looks each pick up in expert_slot
        mgr->q_spec_max = 1000;
        mgr->layers[0] = std::make_unique<llama_moe_stream_layer>();
        sl = mgr->layers[0].get();
        sl->mgr = mgr.get();
        sl->il  = 0;
        sl->n_expert = n_expert;
        sl->n_slots  = n_slots;
        sl->slot_expert  .resize(n_slots, -1);
        sl->slot_state   .resize(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        sl->slot_pending .resize(n_slots, 0);
        sl->slot_gen     .resize(n_slots, 0);
        sl->slot_last_use.resize(n_slots, 0);
        sl->route_hotness.resize(n_expert, 0);
        sl->seen         .resize(n_expert, 0);
        sl->keep         .resize(n_slots, 0);
        sl->slot_of      .resize(n_expert, -1);
        sl->slot_la1     .resize(n_slots, -1);
        sl->slot_la2     .resize(n_slots, -1);
        sl->slot_kept    .resize(n_slots, 0);
    }

    // book e on the desk in slot s, landed, with route hotness hot
    void put(int32_t s, int32_t e, uint32_t hot) {
        mgr->reserve_slot_locked(*sl, e, s);
        sl->slot_state[s] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        sl->route_hotness[e] = hot;
    }

    // the guesses' runners all land at once
    void land() {
        for (auto & st : sl->slot_state) {
            if (st == LLAMA_MOE_STREAM_SLOT_LOADING) {
                st = LLAMA_MOE_STREAM_SLOT_RESIDENT;
            }
        }
    }

    void guess(const std::vector<int32_t> & books, bool two_ahead) {
        llama_moe_stream_lookahead_issue(*sl, books, two_ahead);
    }

    // the floor's stop reads `books`: each pick's slot, the guesses' counts, the release
    void stop(const std::vector<int32_t> & books) {
        std::vector<int32_t> out(books.size(), -1);
        llama_moe_stream_remap_out(*sl, books.data(), out.data(), (int64_t) books.size());
    }

    int32_t slot_of(int32_t e) const {
        const auto it = sl->expert_slot.find(e);
        return it == sl->expert_slot.end() ? -1 : it->second;
    }

    int32_t n_kept() const {
        int32_t n = 0;
        for (const uint8_t k : sl->slot_kept) {
            n += k != 0;
        }
        return n;
    }
};

// the old rule, written out from its comment: empty first, then the coldest (oldest use breaks ties), never a
// LOADING or keep slot. The rule under test must agree with it whenever kept slots are not spared.
int32_t old_rule(const llama_moe_stream_layer & sl, const uint8_t * keep) {
    int32_t v = -1;
    for (uint32_t s = 0; s < sl.n_slots; s++) {
        if ((keep && keep[s]) || sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            continue;
        }
        if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
            return (int32_t) s;
        }
        const uint32_t hs = sl.route_hotness[sl.slot_expert[s]];
        if (v < 0 || hs < sl.route_hotness[sl.slot_expert[v]] ||
                (hs == sl.route_hotness[sl.slot_expert[v]] && sl.slot_last_use[s] < sl.slot_last_use[v])) {
            v = (int32_t) s;
        }
    }
    return v;
}

// a desk of 6 full slots: books 0..5 in slots 0..5, book s with hotness 1 + s, so slot 0 is the coldest
desk full_desk(bool keep_rule) {
    desk d(6, 32, keep_rule);
    for (int32_t s = 0; s < 6; s++) {
        d.put(s, s, 1 + (uint32_t) s);
    }
    return d;
}

} // namespace

int main(int argc, char ** argv) {
    testing t;
    // the keep rule's victim choices on the put-back rule it was written beside (hotness): LRU, the default since
    // 2026-10-15, picks by idle time, which these hand-made desks do not set
    setenv("LLAMA_MOE_STREAM_LRU", "0", 1);

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }
    if (!t.verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("guesses_skip_kept_slots", [&](testing & t) {
        desk d = full_desk(true);
        d.guess({ 10, 11 }, /*two_ahead =*/ true); // the two coldest slots, 0 and 1, now hold two-floor guesses
        d.land();
        t.assert_equal("the two-floor guesses took the coldest slots", 0, d.slot_of(10));
        t.assert_equal("the two-floor guesses took the coldest slots", 1, d.slot_of(11));
        d.sl->route_hotness[10] = d.sl->route_hotness[11] = 0; // and stay the coldest books on the desk

        d.guess({ 20, 21 }, /*two_ahead =*/ false);
        t.assert_true("the one-floor guesses spared the kept books",
                d.slot_of(10) == 0 && d.slot_of(11) == 1 && d.sl->slot_kept[0] == 2 && d.sl->slot_kept[1] == 2);
        t.assert_true("and took the next coldest unkept slots", d.slot_of(20) == 2 && d.slot_of(21) == 3);
        t.assert_equal("nothing was put back unread", (int64_t) 0, d.mgr->stats.n_la2_put_back + d.mgr->stats.n_la1_put_back);
        t.assert_equal("one-floor guesses counted", (int64_t) 2, d.mgr->stats.n_la1_issued);
        t.assert_equal("two-floor guesses counted", (int64_t) 2, d.mgr->stats.n_la2_issued);
    });

    t.test("off_a_guess_takes_the_coldest_kept_book", [&](testing & t) {
        desk d = full_desk(false);
        d.guess({ 10 }, true);
        d.land();
        d.sl->route_hotness[10] = 0;
        d.guess({ 20 }, false);
        t.assert_equal("the old rule: the coldest slot, kept or not", 0, d.slot_of(20));
        t.assert_equal("the two-floor guess was put back unread", (int64_t) 1, d.mgr->stats.n_la2_put_back);
        t.assert_equal("and its slot now holds a one-floor guess", 1, (int) d.sl->slot_kept[0]);
        t.assert_true("its two-floor tag is gone", d.sl->slot_la2[0] == -1 && d.sl->slot_la1[0] == 20);
    });

    t.test("the_issue_loop_stops_when_only_kept_slots_are_left", [&](testing & t) {
        desk d = full_desk(true);
        d.guess({ 10, 11, 12, 13, 14, 15 }, true);
        d.land();
        t.assert_equal("every slot holds a kept guess", 6, d.n_kept());
        d.guess({ 20, 21 }, false);
        t.assert_true("no guess was issued", d.slot_of(20) < 0 && d.slot_of(21) < 0);
        t.assert_equal("one-floor issued", (int64_t) 0, d.mgr->stats.n_la1_issued);
        t.assert_equal("a guess's own pick sees no victim", -1, d.mgr->pick_victim_locked(*d.sl, nullptr, true));
    });

    t.test("need_prefers_unkept_then_takes_kept_rather_than_wait", [&](testing & t) {
        desk d = full_desk(true);
        d.guess({ 10, 11, 12, 13, 14 }, true); // slots 0..4 kept, slot 5 (book 5) the hottest and unkept
        d.land();
        std::vector<uint8_t> keep(6, 0);
        t.assert_equal("an unkept slot first, however hot", 5, d.mgr->pick_victim_need_locked(*d.sl, keep.data()));
        t.assert_equal("not forced", (int64_t) 0, d.mgr->stats.n_keep_forced);

        keep[5] = 1; // this stop reads book 5
        for (int32_t e = 10; e <= 14; e++) {
            d.sl->route_hotness[e] = e == 12 ? 0 : 5; // book 12, in slot 2, is the coldest kept book
        }
        const int32_t v = d.mgr->pick_victim_need_locked(*d.sl, keep.data());
        t.assert_equal("only kept slots left: the coldest kept one, not a wait", 2, v);
        t.assert_equal("forced once", (int64_t) 1, d.mgr->stats.n_keep_forced);
        d.mgr->reserve_slot_locked(*d.sl, 30, v);
        t.assert_equal("the kept book was put back unread", (int64_t) 1, d.mgr->stats.n_la2_put_back);

        // a LOADING slot is still never a victim: with every other slot loading or needed, -1 as before
        for (uint32_t s = 0; s < 6; s++) {
            d.sl->slot_state[s] = s == 5 ? LLAMA_MOE_STREAM_SLOT_RESIDENT : LLAMA_MOE_STREAM_SLOT_LOADING;
        }
        t.assert_equal("every allowed slot loading: wait, as the old rule", -1, d.mgr->pick_victim_need_locked(*d.sl, keep.data()));
        t.assert_equal("a wait is not a forced take", (int64_t) 1, d.mgr->stats.n_keep_forced);
    });

    t.test("the_stop_counts_reads_and_releases_every_kept_slot", [&](testing & t) {
        desk d = full_desk(true);
        d.guess({ 10, 11 }, true);
        d.guess({ 20, 21 }, false);
        d.land();
        t.assert_equal("four kept", 4, d.n_kept());

        d.stop({ 10, 20, 4 }); // reads one guess of each kind, and a book that was no guess
        t.assert_equal("two-floor used", (int64_t) 1, d.mgr->stats.n_la2_used);
        t.assert_equal("one-floor used", (int64_t) 1, d.mgr->stats.n_la1_used);
        t.assert_equal("released, read or not", 0, d.n_kept());
        t.assert_equal("the floor's kept count is zero too", 0, d.sl->n_slot_kept);
        t.assert_true("the unread guesses keep their tags", d.sl->slot_la2[d.slot_of(11)] == 11 && d.sl->slot_la1[d.slot_of(21)] == 21);

        d.sl->route_hotness[11] = 0;
        t.assert_equal("after the stop a guess may take an unread book", d.slot_of(11), d.mgr->pick_victim_locked(*d.sl, nullptr, true));
        d.stop({ 21 }); // read on a later stop, while still in its slot: used
        t.assert_equal("a later read still counts", (int64_t) 2, d.mgr->stats.n_la1_used);
        d.mgr->reserve_slot_locked(*d.sl, 30, d.slot_of(11));
        t.assert_equal("a put-back after the stop is not 'before its floor stopped'", (int64_t) 0,
                d.mgr->stats.n_la2_put_back);
    });

    t.test("a_reused_slot_forgets_its_tags", [&](testing & t) {
        desk d = full_desk(false);
        d.guess({ 10 }, true);
        d.land();
        const int32_t s = d.slot_of(10);
        d.stop({ 1 });                            // the stop passes without reading book 10
        d.mgr->reserve_slot_locked(*d.sl, 30, s); // the slot goes to another book
        d.sl->slot_state[s] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        d.mgr->reserve_slot_locked(*d.sl, 10, s); // and book 10 comes back to it by demand, no guess
        d.sl->slot_state[s] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        d.stop({ 10 });
        t.assert_equal("a demand load is not a guess read", (int64_t) 0, d.mgr->stats.n_la2_used);
    });

    // seeded random floor stops: guesses one and two floors ahead, then a stop that reads some books and loads
    // the missing ones as the remap does (hits first, then each miss through the need rule, then remap_out)
    for (const bool keep_rule : { true, false }) {
        t.test(std::string("random_stops_keep_") + (keep_rule ? "on" : "off"), [&](testing & t) {
            bool ok = true;
            std::string why;
            auto fail = [&](const std::string & what) {
                if (ok) {
                    why = what;
                }
                ok = false;
            };
            for (uint32_t seed = 1; seed <= 200 && ok; seed++) {
                std::mt19937 rng(seed);
                const uint32_t n_slots = 4 + rng() % 12, n_expert = n_slots + 4 + rng() % 24;
                desk d(n_slots, n_expert, keep_rule);
                for (int step = 0; step < 60 && ok; step++) {
                    std::vector<int32_t> g1, g2, need;
                    for (uint32_t i = rng() % 6; i > 0; i--) { g2.push_back((int32_t) (rng() % n_expert)); }
                    for (uint32_t i = rng() % 8; i > 0; i--) { g1.push_back((int32_t) (rng() % n_expert)); }
                    for (uint32_t i = 1 + rng() % n_slots; i > 0; i--) { need.push_back((int32_t) (rng() % n_expert)); }
                    for (auto & h : d.sl->route_hotness) { h = rng() % 4 == 0 ? rng() % 8 : h; }

                    // the guesses: a guess's pick never lands on a kept slot with the rule on
                    std::vector<uint8_t> kept_before = d.sl->slot_kept;
                    std::vector<int32_t> held_before = d.sl->slot_expert;
                    d.guess(g2, true);
                    d.guess(g1, false);
                    for (uint32_t s = 0; s < n_slots && keep_rule; s++) {
                        if (kept_before[s] && d.sl->slot_expert[s] != held_before[s]) {
                            fail("seed " + std::to_string(seed) + ": a guess took a kept slot");
                        }
                    }
                    if (rng() % 3) {
                        d.land(); // sometimes the stop comes before the guesses land
                    }

                    // the stop: hits are this call's keep, each miss takes the need rule's victim
                    std::fill(d.sl->keep.begin(), d.sl->keep.end(), 0);
                    std::vector<int32_t> uniq;
                    for (const int32_t e : need) {
                        if (std::find(uniq.begin(), uniq.end(), e) == uniq.end()) {
                            uniq.push_back(e);
                        }
                    }
                    for (const int32_t e : uniq) {
                        const int32_t s = d.slot_of(e);
                        if (s >= 0) {
                            d.sl->keep[s] = 1;
                            continue;
                        }
                        const int32_t want = old_rule(*d.sl, d.sl->keep.data());
                        bool unkept = false;
                        for (uint32_t s2 = 0; s2 < n_slots; s2++) {
                            unkept = unkept || (!d.sl->keep[s2] && d.sl->slot_state[s2] != LLAMA_MOE_STREAM_SLOT_LOADING &&
                                    !d.sl->slot_kept[s2]);
                        }
                        const int64_t forced0 = d.mgr->stats.n_keep_forced;
                        const int32_t v = d.mgr->pick_victim_need_locked(*d.sl, d.sl->keep.data());
                        if ((v < 0) != (want < 0)) {
                            fail("seed " + std::to_string(seed) + ": the need rule waits where the old rule would not, or the reverse");
                        }
                        if (!keep_rule && v != want) {
                            fail("seed " + std::to_string(seed) + ": off, the need rule is not the old rule");
                        }
                        if (keep_rule && v >= 0 && d.sl->slot_kept[v] && unkept) {
                            fail("seed " + std::to_string(seed) + ": took a kept slot while an unkept one was free");
                        }
                        if ((d.mgr->stats.n_keep_forced > forced0) != (keep_rule && v >= 0 && d.sl->slot_kept[v] != 0)) {
                            fail("seed " + std::to_string(seed) + ": n_keep_forced is not the kept slots taken");
                        }
                        if (v < 0) {
                            // every allowed slot loading: the remap waits for one to land; here it lands now
                            d.land();
                            continue;
                        }
                        d.mgr->reserve_slot_locked(*d.sl, e, v);
                        d.sl->keep[v] = 1;
                    }
                    d.land();
                    std::vector<int32_t> picks;
                    for (const int32_t e : uniq) {
                        if (d.slot_of(e) >= 0) {
                            picks.push_back(e);
                        }
                    }
                    d.stop(picks);

                    if (d.n_kept() != 0 || d.sl->n_slot_kept != 0) {
                        fail("seed " + std::to_string(seed) + ": a kept slot outlived its floor's stop");
                    }
                    for (uint32_t s = 0; s < n_slots; s++) {
                        const int32_t e = d.sl->slot_expert[s];
                        if ((d.sl->slot_la1[s] >= 0 && d.sl->slot_la1[s] != e) || (d.sl->slot_la2[s] >= 0 && d.sl->slot_la2[s] != e)) {
                            fail("seed " + std::to_string(seed) + ": a tag names a book its slot no longer holds");
                        }
                    }
                    const auto & st = d.mgr->stats;
                    if (keep_rule && st.n_la1_put_back + st.n_la2_put_back != st.n_keep_forced) {
                        fail("seed " + std::to_string(seed) + ": with the rule on, a kept book was put back other than by a forced need");
                    }
                    if (st.n_la1_used + st.n_la1_put_back > st.n_la1_issued || st.n_la2_used + st.n_la2_put_back > st.n_la2_issued) {
                        fail("seed " + std::to_string(seed) + ": more guesses used or put back than issued");
                    }
                }
            }
            t.assert_true(why, ok);
        });
    }

    return t.summary();
}

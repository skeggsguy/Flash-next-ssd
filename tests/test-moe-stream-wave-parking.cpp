// The masked waves' parking slots (stage_wave_locked / emit_wave_slots, src/llama-moe-stream-wave-stage.cpp) on a
// desk made by hand, with no model and no runners. A ubatch that touches more books than its wave cap runs its
// expert GEMMs in waves; every wave covers every (token, book) pair, the pairs of other waves weighted zero and
// parked on resident slots. The Metal kernel takes a slot at most once per token row, so a wave needs n_ids
// DISTINCT resident slots: its own books' slots plus slots it borrows.
//
// The planner spreads the books evenly over the waves (plan_waves_locked), so with a small wave cap even wave 0
// can hold fewer books than n_ids. Wave 0 then borrows from the pool the floor's LAST ubatch left behind - stale:
// a slot of it may hold one of this wave's own books (counted twice), or have been put back since. On desk 8
// (41 slots, cap 15, 10 books a token) a 5-token check touching 16-19 books did exactly that and aborted at
// emit_wave_slots' "pi < plan_pool.size()" (the desk sweep, 2026-10-16).
//
// The promises, after every wave: the pool holds at least n_ids slots, none twice, all resident; each token row's
// slots are distinct, its own pairs on their book's slot, its parked pairs on pool slots. Hand-made cases first,
// then seeded random desks and ubatches checked against those promises.

#include "testing.h"

#include "llama.h"

#include "../src/llama-moe-stream.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace {

// one floor with n_slots desk slots and n_expert books, as test-moe-stream-keep.cpp makes it. Every book a test
// touches is put on the desk first, so no wave misses or preloads a book and no read is ever queued
struct desk {
    std::unique_ptr<llama_moe_stream> mgr;
    llama_moe_stream_layer * sl = nullptr;
    std::vector<int32_t> pool0; // wave 0's pool, as the last ubatch staged it

    desk(uint32_t n_slots, uint32_t n_expert, uint32_t cap) {
        mgr = std::make_unique<llama_moe_stream>(/*n_layer =*/ 1, n_slots, /*n_io_threads =*/ 1, /*direct =*/ false);
        mgr->workers_started = true; // no reads to run: no runner threads
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
        sl->plan_capacity = cap; // what the graph sets at build (wave_userdata)
    }

    // book e on the desk in slot s, landed
    void put(int32_t s, int32_t e) {
        mgr->reserve_slot_locked(*sl, e, s);
        sl->slot_state[s] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
    }

    // one ubatch through n_waves graph waves, as llama_moe_stream_wave_ids runs them: plan on wave 0, then stage
    // and emit each wave in order. Returns "" or the first broken promise; it stops before an emit whose pool is
    // already broken, so a broken pool is a failed check rather than an abort
    std::string ubatch(const std::vector<int32_t> & ids, uint32_t n_ids, int32_t n_waves) {
        const int64_t n_tok = (int64_t) ids.size()/n_ids;
        std::vector<int32_t> out(ids.size(), -1);
        std::vector<int>     own(ids.size(), 0); // the waves each pair was really computed in

        std::unique_lock<std::mutex> lk(mgr->mtx);
        for (int32_t w = 0; w < n_waves; w++) {
            if (w == 0) {
                mgr->plan_waves_locked(*sl, ids.data(), (int64_t) ids.size());
            }
            mgr->stage_wave_locked(lk, *sl, w, n_ids);
            sl->plan_next_wave = w + 1;
            if (w == 0) {
                pool0 = sl->plan_pool;
            }

            const std::string at = "wave " + std::to_string(w) + ": ";
            std::vector<int32_t> pool(sl->plan_pool);
            if (pool.size() < n_ids) {
                return at + "the pool holds " + std::to_string(pool.size()) + " slots, fewer than n_ids";
            }
            for (const int32_t s : pool) {
                if (s < 0 || s >= (int32_t) sl->n_slots || sl->slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return at + "pool slot " + std::to_string(s) + " is not a resident slot";
                }
            }
            std::sort(pool.begin(), pool.end());
            if (std::adjacent_find(pool.begin(), pool.end()) != pool.end()) {
                return at + "the pool holds a slot twice";
            }

            mgr->emit_wave_slots(*sl, ids.data(), out.data(), w, n_ids, n_tok);
            for (int64_t t = 0; t < n_tok; t++) {
                std::vector<int32_t> row(out.begin() + t*n_ids, out.begin() + (t + 1)*n_ids);
                for (uint32_t kk = 0; kk < n_ids; kk++) {
                    const int64_t i = t*n_ids + kk;
                    const int32_t s = out[i];
                    if (sl->expert_wave[ids[i]] == (uint8_t) w) {
                        own[i]++;
                        if (s != sl->expert_slot.at(ids[i])) {
                            return at + "an own pair is not on its book's slot";
                        }
                    } else if (std::find(sl->plan_pool.begin(), sl->plan_pool.end(), s) == sl->plan_pool.end()) {
                        return at + "a parked pair is not on a pool slot";
                    }
                }
                std::sort(row.begin(), row.end());
                if (std::adjacent_find(row.begin(), row.end()) != row.end()) {
                    return at + "token row " + std::to_string(t) + " takes a slot twice";
                }
            }
        }
        for (const int c : own) {
            if (c != 1) {
                return "a pair was computed in " + std::to_string(c) + " waves, not exactly one";
            }
        }
        return "";
    }
};

// desk 8's floor: 41 slots, 10 books a token, cap (41 - 10)/2 = 15, books 0..40 in slots 0..40
desk desk8() {
    desk d(41, 64, 15);
    for (int32_t b = 0; b < 41; b++) {
        d.put(b, b);
    }
    return d;
}

// a 5-token check touching books 0..15 in that first-use order: 16 books over cap 15 plan 2 waves of 8 (the even
// spread), fewer than n_ids = 10 each; the graph built 4 waves for the worst case (50 pairs), so 2 run empty
std::vector<int32_t> check16() {
    std::vector<int32_t> ids;
    for (int32_t t = 0; t < 5; t++) {
        for (int32_t k = 0; k < 10; k++) {
            ids.push_back(t == 0 ? k : t == 1 ? (k < 6 ? 10 + k : k - 6) : (3*t + k) % 16);
        }
    }
    return ids;
}

} // namespace

int main(int argc, char ** argv) {
    testing t;
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

    t.test("the_check_plans_two_short_waves", [&](testing & t) {
        desk d = desk8();
        const std::vector<int32_t> ids = check16();
        std::unique_lock<std::mutex> lk(d.mgr->mtx);
        d.mgr->plan_waves_locked(*d.sl, ids.data(), (int64_t) ids.size());
        t.assert_equal("16 books touched", (size_t) 16, d.sl->uniq.size());
        t.assert_equal("two waves planned", (uint32_t) 2, d.sl->plan_n_waves);
        t.assert_true("of 8 books each, fewer than n_ids", d.sl->wave_count[0] == 8 && d.sl->wave_count[1] == 8);
    });

    // the night's abort: the floor's last check left a pool of slots 0..9, whose first two hold wave 0's own books
    t.test("wave_0_skips_its_own_books_in_the_last_pool", [&](testing & t) {
        desk d = desk8();
        d.sl->plan_pool = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        t.assert_equal("every promise kept", std::string(""), d.ubatch(check16(), 10, 4));
        t.assert_true("wave 0 parks on slots 8 and 9, not twice on 0 and 1",
                d.pool0 == std::vector<int32_t>({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }));
    });

    // the last pool's slots may have been put back since, or be landing another book
    t.test("wave_0_skips_last_pool_slots_no_longer_resident", [&](testing & t) {
        desk d(41, 64, 15);
        for (int32_t b = 0; b < 36; b++) {
            d.put(b, b); // slots 36..40 empty
        }
        d.put(36, 50);
        d.sl->slot_state[36] = LLAMA_MOE_STREAM_SLOT_LOADING;
        d.sl->plan_pool = { 37, 36, 3, 38, 20, 21, 22 };
        t.assert_equal("every promise kept", std::string(""), d.ubatch(check16(), 10, 4));
        t.assert_true("wave 0 borrowed the last pool's first resident slots not its own, in its order",
                d.pool0 == std::vector<int32_t>({ 0, 1, 2, 3, 4, 5, 6, 7, 20, 21 }));
    });

    // a pool short of slots (or empty: the first masked ubatch on a floor) borrows from the rest of the desk
    t.test("a_short_last_pool_borrows_from_the_desk", [&](testing & t) {
        desk d = desk8();
        d.sl->plan_pool = { 3 };
        t.assert_equal("every promise kept", std::string(""), d.ubatch(check16(), 10, 4));
        t.assert_true("slot 3 is wave 0's own, so both come from the desk's first free slots",
                d.pool0 == std::vector<int32_t>({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }));
        desk e = desk8();
        e.sl->plan_pool.clear();
        t.assert_equal("an empty pool too", std::string(""), e.ubatch(check16(), 10, 4));
    });

    // seeded random desks, each running a few ubatches in a row so wave 0 meets real leftover pools
    t.test("random_desks_keep_every_promise", [&](testing & t) {
        std::string why;
        for (uint32_t seed = 1; seed <= 300 && why.empty(); seed++) {
            std::mt19937 rng(seed);
            auto pick = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };

            const uint32_t n_expert = 64;
            const uint32_t n_ids    = pick(0, 1) ? 8 : 10;
            const uint32_t n_slots  = pick(3*n_ids, 60);
            const uint32_t cap      = (n_slots - n_ids)/2;
            desk d(n_slots, n_expert, cap);

            std::vector<int32_t> books(n_expert);
            for (uint32_t b = 0; b < n_expert; b++) {
                books[b] = (int32_t) b;
            }
            std::shuffle(books.begin(), books.end(), rng);
            const uint32_t n_empty = pick(0, 3);
            std::vector<int32_t> shelf; // the books on the desk
            for (uint32_t s = 0; s + n_empty < n_slots; s++) {
                d.put((int32_t) s, books[s]);
                shelf.push_back(books[s]);
            }
            for (uint32_t k = pick(0, n_ids); k > 0; k--) {
                d.sl->plan_pool.push_back((int32_t) pick(0, n_slots - 1)); // any leftover: own, empty, repeated
            }

            for (int u = 0; u < 3 && why.empty(); u++) {
                const uint32_t n_tok = pick(1, 8);
                std::vector<int32_t> ids;
                for (uint32_t tk = 0; tk < n_tok; tk++) {
                    std::shuffle(shelf.begin(), shelf.end(), rng);
                    ids.insert(ids.end(), shelf.begin(), shelf.begin() + n_ids);
                }
                const uint32_t worst   = std::min<uint32_t>(n_tok*n_ids, n_expert);
                const int32_t  n_waves = (int32_t) ((worst + cap - 1)/cap);
                const std::string r = d.ubatch(ids, n_ids, n_waves);
                if (!r.empty()) {
                    why = "seed " + std::to_string(seed) + ", ubatch " + std::to_string(u) + ": " + r;
                }
            }
        }
        t.assert_equal("every promise kept", std::string(""), why);
    });

    return t.summary();
}

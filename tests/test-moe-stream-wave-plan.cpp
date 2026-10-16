// The partitioned waves' plan (plan_waves_locked / plan_pairs_locked, src/llama-moe-stream-wave-plan.cpp) on a desk
// made by hand, with no model and no runners. A read-in batch that touches more books than its wave cap runs its
// expert GEMMs in waves; on the partition path every (token, book) pair belongs to exactly one wave, and each wave's
// GEMM is a fixed chunk of pairs. The graph fixes the wave count and the chunk before the router has run, from the
// worst case (every book touched): llm_moe_stream_graph_shape and llm_moe_stream_pair_chunk, called here as built.
//
// The planner must then make exactly that many waves. It used to fall back to ceil(books/cap) waves whenever the
// batch touched fewer books than the graph had waves. On the desk sweep's desk 6 (42 slots, cap 16, 35 waves, chunk
// 4096) the cold 150K read-in's word list routed a 4096-token batch to 34 books on one floor: 3 waves against a graph
// of 35, so 40,960 pairs had room for 3 x 4096 and the server aborted at "wave 0 holds 13655 pairs but chunk is 4096
// after splitting (3 waves, 0 with pair room, 3 with a free expert slot, capacity 16, 34 experts staged)"
// (2026-10-17). Where the fallback made ONE wave (books <= cap), the planner made no pair plan at all and every wave
// wrote only the scratch row: the floor's books were never added in, with no abort.
//
// The promises: the plan has the graph's wave count; no wave holds more pairs than the chunk or more books than the
// cap; every pair is in exactly one wave, a wave that stages its book. Hand-made cases first, then seeded random
// routings over the sweep's desks, then the night's batch through the stager and the pair emit.

#include "testing.h"

#include "llama.h"

#include "../src/llama-graph.h"
#include "../src/llama-moe-stream.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t N_EXPERT = 512; // the library's books a floor
constexpr uint32_t N_IDS    = 10;  // and a token's picks

// one floor with n_slots desk slots, as test-moe-stream-wave-parking.cpp makes it; books 0..n_slots-1 on the desk
struct desk {
    std::unique_ptr<llama_moe_stream> mgr;
    llama_moe_stream_layer * sl = nullptr;

    explicit desk(uint32_t n_slots) {
        mgr = std::make_unique<llama_moe_stream>(/*n_layer =*/ 1, n_slots, /*n_io_threads =*/ 1, /*direct =*/ false);
        mgr->workers_started = true; // no reads to run: no runner threads
        mgr->layers[0] = std::make_unique<llama_moe_stream_layer>();
        sl = mgr->layers[0].get();
        sl->mgr = mgr.get();
        sl->il  = 0;
        sl->n_expert = N_EXPERT;
        sl->n_slots  = n_slots;
        sl->slot_expert  .resize(n_slots, -1);
        sl->slot_state   .resize(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        sl->slot_pending .resize(n_slots, 0);
        sl->slot_gen     .resize(n_slots, 0);
        sl->slot_last_use.resize(n_slots, 0);
        sl->route_hotness.resize(N_EXPERT, 0);
        sl->seen         .resize(N_EXPERT, 0);
        sl->keep         .resize(n_slots, 0);
        sl->slot_of      .resize(N_EXPERT, -1);
        sl->slot_la1     .resize(n_slots, -1);
        sl->slot_la2     .resize(n_slots, -1);
        sl->slot_kept    .resize(n_slots, 0);
        for (uint32_t s = 0; s < n_slots; s++) {
            mgr->reserve_slot_locked(*sl, (int32_t) s, (int32_t) s);
            sl->slot_state[s] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        }
    }

    // the graph's shape for a read-in batch of n_tok here, the reading room not taking it
    llm_moe_stream_shape shape(int64_t n_tok) const {
        return llm_moe_stream_graph_shape(sl, N_EXPERT, N_IDS, n_tok, /*room_allowed =*/ false);
    }

    // what build_moe_stream_partition and wave_userdata set at graph build, then wave 0's plan
    void plan(const std::vector<int32_t> & ids, const llm_moe_stream_shape & sh) {
        const int64_t n_tok = (int64_t) ids.size()/N_IDS;
        sl->plan_capacity   = sh.cap;
        sl->plan_waves_want = sh.n_waves;
        sl->plan_pair_chunk = (uint32_t) llm_moe_stream_pair_chunk((int64_t) ids.size(), n_tok, sh.n_waves);
        std::unique_lock<std::mutex> lk(mgr->mtx);
        mgr->plan_waves_locked(*sl, ids.data(), (int64_t) ids.size());
    }

    // "" or the first broken promise of the plan just made
    std::string check(const std::vector<int32_t> & ids, uint32_t n_waves) const {
        if (sl->plan_n_waves != n_waves || sl->plan_pair.size() != n_waves || sl->wave_count.size() != n_waves) {
            return "the plan has " + std::to_string(sl->plan_n_waves) + " waves (" +
                    std::to_string(sl->plan_pair.size()) + " pair lists) for a graph of " + std::to_string(n_waves);
        }
        std::vector<int> waves_of(ids.size(), 0);
        for (uint32_t w = 0; w < n_waves; w++) {
            const std::string at = "wave " + std::to_string(w) + ": ";
            if (sl->plan_pair[w].size() > sl->plan_pair_chunk) {
                return at + std::to_string(sl->plan_pair[w].size()) + " pairs, past the chunk of " +
                        std::to_string(sl->plan_pair_chunk);
            }
            if (sl->wave_count[w] > sl->plan_capacity) {
                return at + std::to_string(sl->wave_count[w]) + " books, past the cap";
            }
            const auto first = sl->uniq.begin() + sl->wave_first[w];
            const auto last  = first + sl->wave_count[w];
            for (const int32_t i : sl->plan_pair[w]) {
                if (i < 0 || (size_t) i >= ids.size()) {
                    return at + "pair index " + std::to_string(i) + " out of range";
                }
                waves_of[i]++;
                if (std::find(first, last, ids[i]) == last) {
                    return at + "a pair of book " + std::to_string(ids[i]) + ", which this wave does not stage";
                }
            }
        }
        for (size_t i = 0; i < ids.size(); i++) {
            if (waves_of[i] != 1) {
                return "pair " + std::to_string(i) + " is in " + std::to_string(waves_of[i]) + " waves, not one";
            }
        }
        return "";
    }
};

// n_tok tokens, each picking N_IDS distinct books of `books` by weight (heavier first): a seeded routing
std::vector<int32_t> route(std::mt19937 & rng, int64_t n_tok, const std::vector<int32_t> & books,
        const std::vector<double> & weight) {
    std::discrete_distribution<size_t> pick(weight.begin(), weight.end());
    std::vector<int32_t> ids;
    ids.reserve((size_t) n_tok*N_IDS);
    std::vector<uint8_t> used(books.size());
    for (int64_t t = 0; t < n_tok; t++) {
        std::fill(used.begin(), used.end(), 0);
        uint32_t n = 0;
        for (int tries = 0; n < N_IDS && tries < 50; tries++) {
            const size_t b = pick(rng);
            if (!used[b]) { used[b] = 1; ids.push_back(books[b]); n++; }
        }
        for (size_t b = 0; n < N_IDS; b++) { // too steep to fill by chance: the heaviest unused (books fall off by weight)
            if (!used[b]) { used[b] = 1; ids.push_back(books[b]); n++; }
        }
    }
    return ids;
}

// the night's batch: 4096 tokens over 34 books, book 0 on every token and the rest falling off steeply (Zipf 2),
// so one book holds 4096 pairs and a few others most of the rest
std::vector<int32_t> night_batch() {
    std::mt19937 rng(150);
    std::vector<int32_t> books;
    std::vector<double>  weight;
    for (int32_t b = 1; b < 34; b++) {
        books.push_back(b);
        weight.push_back(1.0/((double) b*b));
    }
    std::vector<int32_t> rest = route(rng, 4096, books, std::vector<double>(weight));
    // route() drew N_IDS of the 33; keep the first 9 of each token beside book 0
    std::vector<int32_t> ids;
    for (int64_t t = 0; t < 4096; t++) {
        ids.push_back(0);
        ids.insert(ids.end(), rest.begin() + t*N_IDS, rest.begin() + t*N_IDS + (N_IDS - 1));
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
    // the graph's shape as the engine ships it: no wave-cap or slack override
    for (const char * k : { "LLAMA_MOE_STREAM_WAVE_CAP", "LLAMA_MOE_STREAM_PAIR_SLACK", "LLAMA_MOE_WAVE_SLACK",
                            "LLAMA_MOE_STREAM_PARTITION" }) {
        unsetenv(k);
    }

    // the server log's own line at desk 6: "n_tokens = 4096 -> wave cap = 16 of 42 slots, 35 waves, partition ON"
    t.test("desk_6_builds_35_waves_of_4096", [&](testing & t) {
        desk d(42);
        const llm_moe_stream_shape sh = d.shape(4096);
        t.assert_equal("cap 16", (uint32_t) 16, sh.cap);
        t.assert_equal("35 waves", (uint32_t) 35, sh.n_waves);
        t.assert_true("partitioned", sh.partition && !sh.room);
        t.assert_equal("a chunk of 4096", (int64_t) 4096, llm_moe_stream_pair_chunk(40960, 4096, 35));
    });

    // the night's abort: 34 books, fewer than the graph's 35 waves
    t.test("the_150k_batch_plans_the_graphs_35_waves", [&](testing & t) {
        desk d(42);
        const std::vector<int32_t> ids = night_batch();
        const llm_moe_stream_shape sh = d.shape(4096);
        d.plan(ids, sh);
        std::vector<int32_t> touched(ids);
        std::sort(touched.begin(), touched.end());
        t.assert_equal("34 books touched", (ptrdiff_t) 34, std::unique(touched.begin(), touched.end()) - touched.begin());
        t.assert_equal("every promise kept", std::string(""), d.check(ids, sh.n_waves));
    });

    // desk 14's shape (85 slots, cap 37, 15 waves at 1005 tokens) with a batch on 12 books: they fit one wave of the
    // cap, and the old fallback planned that one wave and no pairs at all
    t.test("fewer_books_than_waves_within_one_cap", [&](testing & t) {
        desk d(85);
        const llm_moe_stream_shape sh = d.shape(1005);
        t.assert_true("15 partitioned waves, cap 37", sh.n_waves == 15 && sh.cap == 37 && sh.partition);
        std::mt19937 rng(14);
        std::vector<int32_t> books;
        std::vector<double>  weight;
        for (int32_t b = 0; b < 12; b++) {
            books.push_back(100 + b);
            weight.push_back(1.0/(1.0 + b));
        }
        const std::vector<int32_t> ids = route(rng, 1005, books, weight);
        d.plan(ids, sh);
        t.assert_equal("every promise kept", std::string(""), d.check(ids, sh.n_waves));
    });

    // seeded routings on every desk the sweep opens (42 slots at desk 6 to 189 at desk 28): any batch size, any
    // number of books touched, flat to steep, with or without one book on every token
    t.test("random_routings_keep_every_promise", [&](testing & t) {
        const uint32_t desks[] = { 42, 41, 56, 71, 85, 100, 115, 130, 144, 159, 174, 189 };
        std::string why;
        int n_plans = 0;
        for (uint32_t seed = 1; seed <= 600 && why.empty(); seed++) {
            std::mt19937 rng(seed);
            auto pick = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(rng); };

            desk d(desks[seed % (sizeof(desks)/sizeof(desks[0]))]);
            const int64_t n_tok = pick(0, 3) == 0 ? pick(60, 1023) : pick(1024, 4096);
            const llm_moe_stream_shape sh = d.shape(n_tok);
            if (!sh.partition || sh.n_waves < 2) {
                continue;
            }
            // a third of them around the wave count (fewer books than waves), the rest anywhere; never a range
            // whose top is under its bottom (undefined: it asked for gigabytes once)
            const uint32_t few     = std::min(N_EXPERT, std::max(N_IDS, sh.n_waves + 2));
            const uint32_t n_books = pick(0, 2) == 0 ? pick(N_IDS, few) : pick(N_IDS, N_EXPERT);
            const double   steep   = std::uniform_real_distribution<double>(0.0, 3.0)(rng);
            std::vector<int32_t> books(N_EXPERT);
            for (uint32_t b = 0; b < N_EXPERT; b++) {
                books[b] = (int32_t) b;
            }
            std::shuffle(books.begin(), books.end(), rng);
            books.resize(n_books);
            std::vector<double> weight(n_books);
            for (uint32_t b = 0; b < n_books; b++) {
                weight[b] = 1.0/std::pow(1.0 + b, steep);
            }
            if (pick(0, 1)) {
                weight[0] = 1e9; // one book on every token
            }
            const std::vector<int32_t> ids = route(rng, n_tok, books, weight);
            d.plan(ids, sh);
            n_plans++;
            const std::string r = d.check(ids, sh.n_waves);
            if (!r.empty()) {
                std::vector<int32_t> touched(ids);
                std::sort(touched.begin(), touched.end());
                const auto n_touched = std::unique(touched.begin(), touched.end()) - touched.begin();
                why = "seed " + std::to_string(seed) + " (" + std::to_string(d.sl->n_slots) + " slots, " +
                        std::to_string(n_tok) + " tokens, " + std::to_string(n_touched) + " books touched, " +
                        std::to_string(sh.n_waves) + " waves): " + r;
            }
        }
        t.assert_true("hundreds of partitioned plans made", n_plans >= 300);
        t.assert_equal("every promise kept", std::string(""), why);
    });

    // the night's batch through every wave as llama_moe_stream_wave_pairs runs it: stage, then emit the four rows.
    // Each real pair is written once, on its book's resident slot; an empty wave writes only the scratch row
    t.test("the_150k_batch_emits_each_pair_once", [&](testing & t) {
        desk d(42);
        const std::vector<int32_t> ids = night_batch();
        const llm_moe_stream_shape sh = d.shape(4096);
        d.plan(ids, sh);
        const int64_t n_pairs = (int64_t) ids.size();
        const int64_t chunk   = d.sl->plan_pair_chunk;
        std::vector<int>     written(n_pairs + 1, 0);
        std::vector<int32_t> out((size_t) LLAMA_MOE_PAIR_ROWS*chunk);
        std::string why;
        std::unique_lock<std::mutex> lk(d.mgr->mtx);
        for (uint32_t w = 0; w < sh.n_waves && why.empty(); w++) {
            d.mgr->stage_wave_locked(lk, *d.sl, (int32_t) w, N_IDS);
            d.sl->plan_next_wave = w + 1;
            d.mgr->emit_wave_pairs(*d.sl, ids.data(), out.data(), (int32_t) w, N_IDS, n_pairs, chunk);
            std::vector<uint8_t> seen(n_pairs + 1, 0);
            for (int64_t p = 0; p < chunk; p++) {
                const int32_t i = out[LLAMA_MOE_PAIR_PAIR*chunk + p];
                const int32_t s = out[LLAMA_MOE_PAIR_SLOT*chunk + p];
                if (d.sl->slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    why = "wave " + std::to_string(w) + " computes on a slot not resident";
                } else if (i < n_pairs && (s != d.sl->expert_slot.at(ids[i]) || out[LLAMA_MOE_PAIR_TOK*chunk + p] != i/(int32_t) N_IDS)) {
                    why = "wave " + std::to_string(w) + ": pair " + std::to_string(i) + " not on its book's slot and token";
                } else if (!seen[i]) {
                    seen[i] = 1; // padding repeats a wave's last pair: count each once a wave
                    written[i]++;
                }
            }
        }
        t.assert_equal("every row right", std::string(""), why);
        bool once = true;
        for (int64_t i = 0; i < n_pairs; i++) {
            once = once && written[i] == 1;
        }
        t.assert_true("every pair written by exactly one wave", once);
        t.assert_true("some wave ran empty, on the scratch row", written[n_pairs] >= 1);
    });

    return t.summary();
}

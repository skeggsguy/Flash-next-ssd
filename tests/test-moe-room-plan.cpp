// The reading room's plan for a floor and its id planes (src/llama-moe-room-plan.cpp).
//
// The plan decides which books the runners bring and where they land; the planes decide which GEMM group
// computes each (word, book) pair. A pair on no plane leaves an output row unwritten, a pair on two is
// written twice (and the row depends on which group ran last), and a book given the wrong slot or record
// is the wrong book: all three change the words without crashing. So the planes are checked here pair by
// pair over random desks, beside the plan's own promises.

#include "testing.h"

#include "../src/llama-moe-room-plan.h"

#include <algorithm>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

// the manager's rule for two runners (llama_moe_stream::use_alt): ids from alt_split percent up are the alt copy's
std::vector<uint8_t> runners(uint32_t n_expert, int32_t alt_split) {
    std::vector<uint8_t> is_alt(n_expert);
    for (uint32_t e = 0; e < n_expert; e++) {
        is_alt[e] = (int64_t) e*100/n_expert >= alt_split;
    }
    return is_alt;
}

// a desk of n_slots with n_books of them holding distinct random books, the rest EMPTY
std::vector<int32_t> random_desk(std::mt19937 & rng, uint32_t n_expert, uint32_t n_slots, uint32_t n_books) {
    std::vector<int32_t> books(n_expert);
    for (uint32_t e = 0; e < n_expert; e++) {
        books[e] = (int32_t) e;
    }
    std::shuffle(books.begin(), books.end(), rng);
    std::vector<int32_t> desk(n_slots, -1);
    std::vector<uint32_t> slots(n_slots);
    for (uint32_t s = 0; s < n_slots; s++) {
        slots[s] = s;
    }
    std::shuffle(slots.begin(), slots.end(), rng);
    for (uint32_t i = 0; i < n_books; i++) {
        desk[slots[i]] = books[i];
    }
    return desk;
}

void check_floor(testing & t, std::mt19937 & rng, uint32_t n_expert, uint32_t n_slots, uint32_t n_books,
        int32_t n_parts, int32_t alt_split, const std::string & what) {
    const std::vector<int32_t> desk   = random_desk(rng, n_expert, n_slots, n_books);
    const std::vector<uint8_t> is_alt = runners(n_expert, alt_split);
    const llama_moe_room_floor_plan plan = llama_moe_room_plan_floor(n_expert, desk, is_alt, n_parts);

    // the books brought = exactly the books not on the desk (a LOADING book counts as on it)
    std::set<int32_t> on_desk(desk.begin(), desk.end());
    on_desk.erase(-1);
    std::multiset<int32_t> brought;
    for (const auto & [e, s] : plan.fills) {
        brought.insert(e);
    }
    for (const auto & part : plan.parts) {
        brought.insert(part.begin(), part.end());
    }
    bool complement = brought.size() + on_desk.size() == n_expert;
    for (const int32_t e : brought) {
        complement = complement && brought.count(e) == 1 && on_desk.count(e) == 0;
    }
    t.assert_true(what + ": brings exactly the books not on the desk", complement);

    // a cold desk fills EMPTY slots only, as many as it has; a full one is never evicted
    const size_t n_empty = n_slots - n_books;
    bool into_empty = plan.fills.size() == std::min(n_empty, (size_t) (n_expert - n_books));
    std::set<int32_t> used;
    for (const auto & [e, s] : plan.fills) {
        into_empty = into_empty && desk[s] == -1 && used.insert(s).second;
    }
    t.assert_true(what + ": fills only EMPTY desk slots", into_empty);

    // parts within one book of each other, and each runner's share of every part within one book
    size_t lo = SIZE_MAX, hi = 0, n_home = 0, n_rest = 0;
    for (const auto & part : plan.parts) {
        lo = std::min(lo, part.size());
        hi = std::max(hi, part.size());
        n_rest += part.size();
        for (const int32_t e : part) {
            n_home += !is_alt[e];
        }
    }
    t.assert_true(what + ": parts balanced to one book", (int32_t) plan.parts.size() == n_parts && hi - lo <= 1);
    bool shared = true;
    for (const auto & part : plan.parts) {
        size_t h = 0;
        for (const int32_t e : part) {
            h += !is_alt[e];
        }
        const double want = n_rest ? (double) part.size()*n_home/n_rest : 0.0;
        shared = shared && h + 1 > want && h < want + 1;
    }
    t.assert_true(what + ": both runners on every part", shared);

    // the planes: desk slots (cold fills included) then each part's records, placed at some record base
    std::vector<std::vector<int32_t>> where(1 + n_parts, std::vector<int32_t>(n_expert, -1));
    std::vector<int32_t> slot_of(n_expert, -1), record_of(n_expert, -1), part_of(n_expert, -1);
    for (uint32_t s = 0; s < n_slots; s++) {
        if (desk[s] >= 0) {
            where[0][desk[s]] = slot_of[desk[s]] = (int32_t) s;
        }
    }
    for (const auto & [e, s] : plan.fills) {
        where[0][e] = slot_of[e] = s;
    }
    for (int32_t p = 0; p < n_parts; p++) {
        const int32_t base = (int32_t) (rng() % 100);
        for (size_t i = 0; i < plan.parts[p].size(); i++) {
            const int32_t e = plan.parts[p][i];
            where[1 + p][e] = record_of[e] = base + (int32_t) i;
            part_of[e] = p;
        }
    }

    const int64_t n_tokens = 1 + rng() % 300, n_used = 8;
    std::vector<int32_t> ids(n_tokens*n_used);
    std::vector<int32_t> all(n_expert);
    for (uint32_t e = 0; e < n_expert; e++) {
        all[e] = (int32_t) e;
    }
    for (int64_t tok = 0; tok < n_tokens; tok++) {
        std::shuffle(all.begin(), all.end(), rng);
        std::copy(all.begin(), all.begin() + n_used, ids.begin() + tok*n_used);
    }

    std::vector<std::vector<int32_t>> planes(1 + n_parts, std::vector<int32_t>(ids.size()));
    int64_t owned = 0;
    for (int32_t g = 0; g <= n_parts; g++) {
        owned += llama_moe_room_emit(ids.data(), (int64_t) ids.size(), where[g], planes[g].data());
    }
    bool once = owned == (int64_t) ids.size();
    for (size_t i = 0; i < ids.size(); i++) {
        const int32_t e = ids[i];
        int n_on = 0;
        for (int32_t g = 0; g <= n_parts; g++) {
            if (planes[g][i] < 0) {
                continue;
            }
            n_on++;
            once = once && (g == 0 ? planes[g][i] == slot_of[e] : g - 1 == part_of[e] && planes[g][i] == record_of[e]);
        }
        once = once && n_on == 1;
    }
    t.assert_true(what + ": every pair on exactly one plane, at its book's slot or record", once);
}

} // namespace

int main(int argc, char ** argv) {
    testing t;
    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    t.verbose = verbose && std::string(verbose) == "1";
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("a hand-made floor", [](testing & t) {
        // 12 books, ids 0-5 on the first runner and 6-11 on the second; slots hold 3 and 7, slot 2 is EMPTY
        const std::vector<int32_t> desk = { 3, 7, -1 };
        const auto plan = llama_moe_room_plan_floor(12, desk, runners(12, 50), 3);
        t.assert_equal("one fill", (size_t) 1, plan.fills.size());
        // missing: 0 1 2 4 5 on the first runner, 6 8 9 10 11 on the second; the order after m books holds
        // floor(m*5/10) of the first's: 6 0 8 1 9 2 10 4 11 5
        t.assert_true("the first book of the order into the EMPTY slot", plan.fills[0] == std::make_pair(6, 2));
        const std::vector<std::vector<int32_t>> want = { { 0, 8, 1 }, { 9, 2, 10 }, { 4, 11, 5 } };
        t.assert_true("interleaved and cut in three", plan.parts == want);
    });

    t.test("fewer books than parts", [](testing & t) {
        const std::vector<int32_t> desk = { 0, 1, 2, 3, 4, 5, 6 };
        const auto plan = llama_moe_room_plan_floor(9, desk, runners(9, 100), 4);
        t.assert_true("two parts of one book, two empty",
                plan.fills.empty() && plan.parts[0] == std::vector<int32_t>{ 7 } &&
                plan.parts[1] == std::vector<int32_t>{ 8 } && plan.parts[2].empty() && plan.parts[3].empty());
    });

    t.test("random desks", [](testing & t) {
        std::mt19937 rng(42);
        for (int i = 0; i < 40; i++) {
            check_floor(t, rng, 512, 220, 220, 4, 53, "warm library desk");
            check_floor(t, rng, 512, 220, (uint32_t) (rng() % 220), 1 + (int32_t) (rng() % 16), 53, "partly cold desk");
            check_floor(t, rng, 64, 40, 0, 4, 100, "cold desk, one runner");
            check_floor(t, rng, 64, 40, 38, 1 + (int32_t) (rng() % 16), 53, "desk with a few EMPTY slots");
            check_floor(t, rng, 64, 61, 61, 4, 53, "fewer books than parts");
        }
    });

    return t.summary();
}

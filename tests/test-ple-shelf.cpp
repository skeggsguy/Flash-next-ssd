// The phrasebook shelf's bookkeeping (src/llama-ple-shelf.h) and its lines (src/llama-ple-shelf-stats.h).
//
// The shelf fails silently when it is wrong: a row found in a slot that holds another row, or a slot given
// to two rows of one batch, hands the graph the wrong numbers and the words change without a crash. So
// besides the hand-worked cases (the rule, worked from ~/dev/ai/sim/ple_policies.py clock_hits with
// insert_bit 0, the simulator's CLOCK-cold), seeded random batches are checked after every plan against an
// oracle kept with a std::unordered_map (not the shelf's open-addressing table) and against invariants:
// every slot's row is found at that slot, no two picks of a batch share a slot, a batch passes rows only
// when it holds every slot, and the shelf never holds more than its capacity.

#include "../src/llama-ple-shelf.h"
#include "../src/llama-ple-shelf-stats.h"

#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

using picks_t = std::vector<llama_ple_shelf_pick>;

static picks_t plan(llama_ple_shelf & s, const std::vector<int32_t> & rows) {
    picks_t p(rows.size());
    s.plan(rows.data(), (int64_t) rows.size(), p.data());
    return p;
}

static std::string kinds(const picks_t & p) {
    std::string out;
    for (const auto & x : p) {
        out += x.kind == LLAMA_PLE_SHELF_HIT ? 'H' : x.kind == LLAMA_PLE_SHELF_MISS ? 'M' : 'P';
    }
    return out;
}

// one row a batch, three slots: the sequence worked by hand in the comment, as clock_hits(insert_bit=0) runs it
static void test_rule_by_hand() {
    llama_ple_shelf s(3);
    std::string got;
    for (int32_t r : {1, 2, 3, 1, 4, 5, 2, 4, 1}) {
        got += kinds(plan(s, {r}));
    }
    // 1 2 3 fill slots 0 1 2 (bits clear). 1 hits (slot 0's bit set). 4: the hand clears slot 0 and takes
    // slot 1 (row 2, cold). 5 takes slot 2 (row 3). 2: slot 0's bit is clear now, so row 1 goes. 4 hits
    // slot 1. 1: slot 1's bit is cleared and slot 2 (row 5) goes. 2 hits of 9.
    CHECK(got == "MMMHMMMHM", "the rule: %s", got.c_str());
    CHECK(s.row_at(0) == 2 && s.row_at(1) == 4 && s.row_at(2) == 1, "slots %d %d %d", s.row_at(0), s.row_at(1), s.row_at(2));
    CHECK(s.hand_at() == 0 && s.held() == 3, "hand %lld held %lld", (long long) s.hand_at(), (long long) s.held());
    CHECK(s.find(3) == -1 && s.find(5) == -1 && s.find(1) == 2, "find after evictions");

    // cold entry: a row read once goes before one asked for twice
    llama_ple_shelf c(2);
    got.clear();
    for (int32_t r : {7, 8, 7, 9, 7, 8}) {
        got += kinds(plan(c, {r}));
    }
    // 7 8 fill; 7 hits; 9: slot 0 (7) has its bit, cleared, slot 1 (8, cold) goes; 7 hits; 8: slot 0 (7)
    // set again, cleared, slot 1 (9, cold) goes
    CHECK(got == "MMHMHM", "cold entry: %s", got.c_str());
    CHECK(c.find(7) == 0 && c.find(8) == 1 && c.find(9) == -1, "cold entry slots");
}

// the one addition to the simulator's rule: a batch never takes back a slot it hit or filled
static void test_batch_keeps_its_slots() {
    llama_ple_shelf s(2);
    CHECK(kinds(plan(s, {1, 2})) == "MM", "fill");
    // 3 and 4 take slots 0 and 1; 5 finds every slot held by this batch: it passes (the simulator would
    // have put 5 over 3, a row the batch is still reading)
    picks_t p = plan(s, {3, 4, 5});
    CHECK(kinds(p) == "MMP" && p[2].slot == -1, "a full batch passes: %s", kinds(p).c_str());
    CHECK(s.find(3) == 0 && s.find(4) == 1 && s.find(5) == -1, "the passed row is not kept");

    // a hit protects its slot within the batch: 3 hits slot 0, and 6 must take slot 1 even though the hand
    // points at slot 0
    llama_ple_shelf h(2);
    plan(h, {3});
    plan(h, {4});
    p = plan(h, {3, 6});
    CHECK(kinds(p) == "HM" && p[0].slot == 0 && p[1].slot == 1, "a hit keeps its slot: %s %lld", kinds(p).c_str(), (long long) p[1].slot);

    // a shelf of one slot: every batch of several rows keeps its first miss and passes the rest
    llama_ple_shelf one(1);
    CHECK(kinds(plan(one, {10, 11, 12})) == "MPP", "one slot");
    CHECK(kinds(plan(one, {10, 13})) == "HP", "one slot, a hit holds it");
    CHECK(kinds(plan(one, {13, 10})) == "MP", "one slot, then a miss takes it");
}

// the oracle: the same rule over a std::unordered_map, slot by slot
struct oracle {
    int64_t cap, filled = 0, hand = 0;
    uint32_t call = 0;
    std::vector<int32_t> row;
    std::vector<uint8_t> ref;
    std::vector<uint32_t> stamp;
    std::unordered_map<int32_t, int64_t> where;

    explicit oracle(int64_t c) : cap(c), row(c, -1), ref(c, 0), stamp(c, 0) {}

    picks_t plan(const std::vector<int32_t> & rows) {
        ++call;
        int64_t own = 0;
        picks_t out;
        for (int32_t r : rows) {
            auto it = where.find(r);
            int64_t s;
            uint8_t kind = LLAMA_PLE_SHELF_MISS;
            if (it != where.end()) {
                s = it->second; ref[s] = 1; kind = LLAMA_PLE_SHELF_HIT;
            } else if (filled < cap) {
                s = filled++;
            } else if (own < cap) {
                for (;;) {
                    s = hand; hand = (hand + 1) % cap;
                    if (stamp[s] == call || ref[s]) { ref[s] = 0; continue; }
                    break;
                }
                where.erase(row[s]);
            } else {
                out.push_back({ -1, LLAMA_PLE_SHELF_PASS });
                continue;
            }
            if (kind == LLAMA_PLE_SHELF_MISS) { row[s] = r; where[r] = s; ref[s] = 0; }
            stamp[s] = call; ++own;
            out.push_back({ s, kind });
        }
        return out;
    }
};

// seeded random batches over a skewed set of rows, every capacity from one slot up, against the oracle
static void test_oracle() {
    std::mt19937 rng(20260927);
    for (int64_t cap : {1, 2, 3, 7, 64, 257, 1000, 5000}) {
        llama_ple_shelf s(cap);
        oracle o(cap);
        const int32_t space = cap < 100 ? 40 : (int32_t) cap * 3;
        std::geometric_distribution<int32_t> skew(cap < 100 ? 0.08 : 3.0 / space);
        int64_t hits = 0, asks = 0, bad = 0;
        for (int b = 0; b < 3000 && bad == 0; ++b) {
            const int n = 1 + (int) (rng() % (b % 5 == 0 ? 600 : 16)); // written tokens and read-in batches
            std::set<int32_t> seen;
            std::vector<int32_t> rows;
            for (int i = 0; i < n; ++i) {
                // mostly the skewed head, some rows from anywhere in a 31-bit id space (a 320M-row table)
                const int32_t r = rng() % 8 == 0 ? (int32_t) (rng() & 0x7fffffff) : skew(rng) % space;
                if (seen.insert(r).second) { rows.push_back(r); }
            }
            if (b % 2) { std::shuffle(rows.begin(), rows.end(), rng); } // the rule must not need sorted rows
            const picks_t got = plan(s, rows), want = o.plan(rows);
            std::set<int64_t> used;
            int64_t kept = 0;
            for (size_t i = 0; i < rows.size(); ++i) {
                if (got[i].kind != want[i].kind || got[i].slot != want[i].slot) { ++bad; }
                if (got[i].kind == LLAMA_PLE_SHELF_PASS) { continue; }
                ++kept;
                if (!used.insert(got[i].slot).second) { ++bad; }             // two rows of a batch in one slot
                if (s.row_at(got[i].slot) != rows[i]) { ++bad; }              // the slot holds the row served
                hits += got[i].kind == LLAMA_PLE_SHELF_HIT;
            }
            asks += (int64_t) rows.size();
            if (kept < (int64_t) rows.size() && kept != cap) { ++bad; }      // passes only when every slot is the batch's
            if (s.held() > cap || s.held() != o.filled || s.hand_at() != o.hand) { ++bad; }
        }
        // every slot's row is found at that slot, and nothing else is found
        for (int64_t k = 0; k < s.held(); ++k) {
            if (s.find(s.row_at(k)) != k) { ++bad; }
        }
        for (const auto & [r, k] : o.where) {
            if (s.find(r) != k) { ++bad; }
        }
        CHECK(bad == 0, "capacity %lld: %lld disagreements with the oracle", (long long) cap, (long long) bad);
        CHECK(hits > 0 && hits < asks, "capacity %lld: a mix of hits (%lld) and misses (%lld asks)",
                (long long) cap, (long long) hits, (long long) asks);
    }
}

// many evictions at a large shelf: the open-addressing table's deletions keep every row findable
static void test_table_churn() {
    const int64_t cap = 50000;
    llama_ple_shelf s(cap);
    std::mt19937 rng(7);
    int64_t bad = 0;
    for (int b = 0; b < 200; ++b) {
        std::set<int32_t> seen;
        std::vector<int32_t> rows;
        // rows in a narrow band that crowd the table's runs, and misses everywhere
        while (rows.size() < 4000) {
            const int32_t r = (int32_t) (rng() % (cap * 4));
            if (seen.insert(r).second) { rows.push_back(r); }
        }
        const picks_t p = plan(s, rows);
        for (size_t i = 0; i < rows.size(); ++i) {
            if (p[i].kind == LLAMA_PLE_SHELF_PASS || s.find(rows[i]) != p[i].slot) { ++bad; }
        }
    }
    for (int64_t k = 0; k < s.held(); ++k) {
        if (s.find(s.row_at(k)) != k) { ++bad; }
    }
    CHECK(bad == 0 && s.held() == cap, "churn: %lld bad, %lld held", (long long) bad, (long long) s.held());
}

static void test_stats() {
    CHECK(llama_ple_shelf_kind_of(1) == 0 && llama_ple_shelf_kind_of(6) == 0 && llama_ple_shelf_kind_of(7) == 1,
            "writing is up to 6 tokens, as the simulator's --check-max-tokens");

    // the size: auto is 128 MiB of raw rows; never more slots than rows, never none when on
    CHECK(llama_ple_shelf_slots(0, 90, 320001536) == 0, "0 is off");
    CHECK(llama_ple_shelf_slots(-1, 90, 320001536) == 128LL * 1048576 / 90, "auto: %lld", (long long) llama_ple_shelf_slots(-1, 90, 320001536));
    CHECK(llama_ple_shelf_slots(1, 90, 320001536) == 11650, "1 MiB");
    CHECK(llama_ple_shelf_slots(1024, 90, 1000) == 1000, "at most the table's rows");
    CHECK(llama_ple_shelf_slots(1, 4 << 20, 10) == 1, "at least one slot");

    llama_ple_shelf_stats a, b;
    CHECK(llama_ple_shelf_window_line(a, b, 0, 10).empty(), "a window without asks prints nothing");
    a.calls = 3; a.asks[0] = 32; a.hits[0] = 24; a.asks[1] = 1000; a.hits[1] = 750; a.reads[0] = 150; a.reads[1] = 108;
    a.bytes = 258 * 90; a.t_wait_us = 2500;
    const std::string w = llama_ple_shelf_window_line(a, b, 1234, 5000);
    const std::string t = llama_ple_shelf_total_line(a, 1234, 5000);
    for (const std::string & line : {w, t}) {
        // its own first word, and none the runner's parsers match (runner/roomstats.py)
        CHECK(line.rfind("phrasebook: ", 0) == 0, "prefix: %s", line.c_str());
        for (const char * word : {"moe stream", "room", "drives", "lent belt"}) {
            CHECK(line.find(word) == std::string::npos, "'%s' in %s", word, line.c_str());
        }
    }
    CHECK(w.find("asks 1032, on the shelf 774 (75.0%)") != std::string::npos, "window rates: %s", w.c_str());
    CHECK(w.find("writing 32, 24 (75.0%)") != std::string::npos && w.find("reading in 1000, 750 (75.0%)") != std::string::npos,
            "the split: %s", w.c_str());
    CHECK(w.find("read 258 rows (file 150, alt 108)") != std::string::npos && w.find("2.5 ms waited") != std::string::npos,
            "reads and wait: %s", w.c_str());
    CHECK(w.find("held 1234 of 5000 rows") != std::string::npos, "held: %s", w.c_str());
    b = a; b.asks[1] = 900; b.hits[1] = 700;
    const std::string w2 = llama_ple_shelf_window_line(a, b, 1, 5000);
    CHECK(w2.find("window asks 100, on the shelf 50 (50.0%)") != std::string::npos
            && w2.find("since start asks 1032, on the shelf 774 (75.0%)") != std::string::npos, "window vs since start: %s", w2.c_str());
    a.begun = 2;
    CHECK(llama_ple_shelf_total_line(a, 1, 5000).find("3 batches (2 started ahead)") != std::string::npos, "total: %s", t.c_str());

    const std::string on = llama_ple_shelf_startup_line(-1, 1491308, 90, 320001536, 2, true, true, "");
    CHECK(on.rfind("phrasebook shelf: 128.0 MiB holds 1491308", 0) == 0 && on.find("auto") != std::string::npos
            && on.find("both drives") != std::string::npos && on.find("locked in memory") != std::string::npos
            && on.find("--ple-shelf 0") != std::string::npos, "startup: %s", on.c_str());
    const std::string off = llama_ple_shelf_startup_line(64, 745654, 90, 320001536, 1, false, false, "Resource temporarily unavailable");
    CHECK(off.find("--ple-shelf 64 MiB") != std::string::npos && off.find("its one copy") != std::string::npos
            && off.find("file cache (no-cache reads were refused)") != std::string::npos
            && off.find("could not be locked in memory (Resource temporarily unavailable)") != std::string::npos,
            "startup, the other branches: %s", off.c_str());
}

int main() {
    test_rule_by_hand();
    test_batch_keeps_its_slots();
    test_oracle();
    test_table_churn();
    test_stats();
    if (n_fail) {
        fprintf(stderr, "test-ple-shelf: %d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-ple-shelf: all checks passed\n");
    return 0;
}

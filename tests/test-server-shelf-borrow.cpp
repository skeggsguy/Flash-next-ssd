// Borrowing from the bookmark shelf (test-server-shelf's third file; tools/server/server-shelf-borrow.cpp, study
// SHELF-BORROW-PLAN.md step 2), without a model:
//
// - the time rule: its constants, and a borrow taken only when it saves at least the margin
// - the seat's own reach: its shared tokens when its memory cuts anywhere, else its newest copy within them
// - the choice: the 25% rule still wins when it applies; a new chat after coding borrows the opening's copy; a
//   borrow is refused at or below the seat's own copy, past the shared tokens, at the task's last token, from an
//   entry whose apprentice is behind, or when it does not save time; the largest copy wins, then the smallest file
// - invariants over generated shelves: whatever is borrowed qualifies, and nothing that qualifies beats it
// - a borrowed entry stays on the shelf (its file too) and moves to the newest end, so it is not the first dropped
//   for room; an ordinary load still drops it
// - the copies' positions round-trip: an entry's positions are the ones its file gives back

#include "test-server-shelf.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static const uint64_t MiB = 1024ull * 1024ull;
static const uint64_t GB  = 1000ull * 1000ull * 1000ull;

// a saved copy at n tokens, as a recurrent memory's copy is (first and last position the copy's last token)
static common_prompt_checkpoint copy_at(int64_t n) {
    common_prompt_checkpoint c;
    c.n_tokens = n;
    c.pos_min  = (llama_pos) n - 1;
    c.pos_max  = (llama_pos) n - 1;
    return c;
}

static std::list<common_prompt_checkpoint> copies_at(const std::vector<int64_t> & at) {
    std::list<common_prompt_checkpoint> res;
    for (int64_t n : at) {
        res.push_back(copy_at(n));
    }
    return res;
}

static server_prompt seat_of(const llama_tokens & t, const std::vector<int64_t> & at) {
    server_prompt p;
    p.tokens      = st(t);
    p.checkpoints = copies_at(at);
    return p;
}

// a bookmark the borrowing rule sees, without a file
static void add_borrowable(server_shelf & shelf, uint64_t id, const llama_tokens & t, uint64_t size,
        const std::vector<int64_t> & at, bool spec_synced = true) {
    shelf.add(id, st(t), "", size, copies_at(at), spec_synced);
}

static void test_borrow_saves() {
    // the constants are the study's measured rates at their borrow-unfriendly ends (results/bookmark-shelf)
    CHECK(SERVER_SHELF_BORROW_READ_TPS == 440.0);
    CHECK(SERVER_SHELF_BORROW_LOAD_BPS == 3.7e9);
    CHECK(SERVER_SHELF_BORROW_MARGIN_S == 1.0);

    // sidekick's opening (4,200 tokens, ~9.5 s) from a 50K bookmark (1.86 GB, ~0.5 s): far past the margin
    CHECK(server_shelf_borrow_saves(4200, 1861813976ull));
    // ... and from a 150K one (4.64 GB, ~1.25 s)
    CHECK(server_shelf_borrow_saves(4200, 4640716256ull));
    // exactly the margin with nothing to load counts; a token less does not
    CHECK(server_shelf_borrow_saves(440, 0));
    CHECK(!server_shelf_borrow_saves(439, 0));
    // a short share from a big file: 600 tokens (1.36 s) against 1.86 GB (0.50 s) saves 0.86 s, refused; 700 saves 1.09
    CHECK(!server_shelf_borrow_saves(600, 1861813976ull));
    CHECK(server_shelf_borrow_saves(700, 1861813976ull));
    // a 3.7 GB file takes a second: it needs 2 s of reading, 880 tokens
    CHECK(server_shelf_borrow_saves(880, 37 * GB / 10));
    CHECK(!server_shelf_borrow_saves(879, 37 * GB / 10));
    // nothing to read, or less than nothing
    CHECK(!server_shelf_borrow_saves(0, 0));
    CHECK(!server_shelf_borrow_saves(-5, 0));
}

static void test_seat_reach() {
    const llama_tokens task = toks(0, 4200, 200000, 300);
    // a code seat parting from chat at token 3: nothing, whatever its copies
    CHECK(server_shelf_seat_reach(seat_of(toks(0, 3, 300000, 2000), { 1500 }), st(task), false) == 0);
    CHECK(server_shelf_seat_reach(seat_of(toks(0, 3, 300000, 2000), { 1500 }), st(task), true) == 3);
    // a chat seat sharing the opening: its copy at the opening's end, or at 3000 if that is its newest within
    CHECK(server_shelf_seat_reach(seat_of(toks(0, 4200, 400000, 9000), { 4200, 9000 }), st(task), false) == 4200);
    CHECK(server_shelf_seat_reach(seat_of(toks(0, 4200, 400000, 9000), { 3000, 4300 }), st(task), false) == 3000);
    // no copy within the share: nothing; a memory that cuts anywhere keeps the share itself
    CHECK(server_shelf_seat_reach(seat_of(toks(0, 4200, 400000, 9000), { 9000 }), st(task), false) == 0);
    CHECK(server_shelf_seat_reach(seat_of(toks(0, 4200, 400000, 9000), {}), st(task), true) == 4200);
    // an empty seat or an empty task reach nothing
    CHECK(server_shelf_seat_reach(seat_of({}, {}), st(task), true) == 0);
    CHECK(server_shelf_seat_reach(seat_of(toks(0, 4200, 400000, 9000), { 4200 }), st({}), false) == 0);
}

// the plan's case: A (a 50K chat) on the shelf, B (code) on the seat, C a new chat with the same opening
static void test_borrow_choice() {
    const llama_tokens a = toks(0, 4200, 100000, 45800);
    const llama_tokens c = toks(0, 4200, 200000, 300);
    const server_prompt b = seat_of(toks(0, 3, 300000, 2000), { 1500 });

    server_shelf shelf;
    add_borrowable(shelf, 1, a, 1861813976ull, { 4200, 30000, 49000 });

    // the 25% rule finds nothing (C shares 8% of A), so C borrows A's copy at the end of the opening
    CHECK(shelf.pick(b.tokens, st(c)) == 0);
    auto t = shelf.take(b, st(c), false);
    CHECK(t.id == 1 && t.borrow == 4200 && t.shared == 4200);
    CHECK(shelf.borrow_pick(b, st(c), false).id == 1);
    // the hand-over saves B and loads A, as for a pick
    const auto h = server_shelf_hand_on(b.tokens.size(), 3, true, t.id);
    CHECK(h.save && h.load);

    // A returning passes the 25% rule: the ordinary pick wins, no borrow
    t = shelf.take(b, st(extend(a, 500000, 40)), false);
    CHECK(t.id == 1 && t.borrow == -1);
    // ... and with a borrowable entry beside it, still the pick
    add_borrowable(shelf, 2, toks(0, 4200, 600000, 9000), 400 * MiB, { 4200 });
    t = shelf.take(b, st(extend(a, 500000, 40)), false);
    CHECK(t.id == 1 && t.borrow == -1);

    // refused at or below the seat's own copy: a chat seat with its copy at the opening's end needs no borrow
    server_shelf one;
    add_borrowable(one, 1, a, 1861813976ull, { 4200, 30000 });
    CHECK(one.take(seat_of(toks(0, 4200, 400000, 9000), { 4200 }), st(c), false).id == 0);
    // ... nor a seat whose memory cuts anywhere and shares the opening
    CHECK(one.take(seat_of(toks(0, 4200, 400000, 9000), {}), st(c), true).id == 0);
    // ... but a seat reaching only 3000 borrows, counting only the 1,200 tokens past it (2.7 s against 0.5 s)
    t = one.take(seat_of(toks(0, 4200, 400000, 9000), { 3000 }), st(c), false);
    CHECK(t.id == 1 && t.borrow == 4200);
    // ... while a seat reaching 3900 leaves 300 tokens, under a second's saving: refused
    CHECK(one.take(seat_of(toks(0, 4200, 400000, 9000), { 3900 }), st(c), false).id == 0);

    // refused past the shared tokens: A's only copy at 4300 while C shares 4200 of A
    server_shelf past;
    add_borrowable(past, 1, a, 1861813976ull, { 4300, 30000 });
    CHECK(past.take(b, st(c), false).id == 0);
    // a copy exactly at the share is used; one at the task's last token is not (nothing would be left to read)
    server_shelf edge;
    add_borrowable(edge, 1, toks(0, 4200, 100000, 1000), 100 * MiB, { 4200 });
    CHECK(edge.borrow_pick(b, st(c), false).borrow == 4200);
    CHECK(edge.borrow_pick(b, st(toks(0, 4200, 0, 0)), false).id == 0); // the task is the opening itself

    // refused when it does not save time: a 600-token share from a 1.86 GB file
    server_shelf slow;
    add_borrowable(slow, 1, toks(0, 600, 100000, 45800), 1861813976ull, { 600 });
    CHECK(slow.take(b, st(toks(0, 600, 200000, 300)), false).id == 0);
    // refused from an entry whose apprentice is behind (the read-in would start from 0), or with no copies
    server_shelf behind;
    add_borrowable(behind, 1, a, 1861813976ull, { 4200 }, false);
    add_borrowable(behind, 2, a, 1861813976ull, {});
    CHECK(behind.take(b, st(c), false).id == 0);
    // an empty task or an empty shelf: nothing
    CHECK(shelf.take(b, st({}), false).id == 0);
    CHECK(server_shelf().take(b, st(c), false).id == 0);

    // the largest copy wins, then the smallest file, then the oldest (50K bookmarks: under the 25% rule)
    server_shelf many;
    add_borrowable(many, 1, toks(0, 4200, 110000, 45800), 2 * GB, { 4200 });
    add_borrowable(many, 2, toks(0, 4200, 120000, 45800), 1 * GB, { 4200 });
    add_borrowable(many, 3, toks(0, 4200, 130000, 45800), 1 * GB, { 4200 });
    add_borrowable(many, 4, toks(0, 3000, 140000, 45800), 100 * MiB, { 3000 });
    t = many.take(b, st(c), false);
    CHECK(t.id == 2 && t.borrow == 4200);
    // a later, bigger bookmark sharing more of C (its copy at 4250) beats them all while it still saves time
    const llama_tokens c2 = toks(0, 4250, 200000, 300);
    add_borrowable(many, 5, toks(0, 4250, 150000, 45800), 5 * GB, { 4250 });
    t = many.take(b, st(c2), false);
    CHECK(t.id == 5 && t.borrow == 4250 && t.shared == 4250);
}

static fs::path borrow_dir();
static void touch(const std::string & p);
// an ordinary pick keeps the bookmark when the new chat only diverges from the entry (a short old chat sharing >=25%
// of itself with a new chat's opening), and drops it only on a true return (the task continues the whole entry)
static void test_pick_keeps_when_diverging() {
    const server_prompt b = seat_of(toks(0, 3, 300000, 2000), { 1500 });
    const llama_tokens old = toks(0, 4200, 100000, 8000);
    server_shelf shelf;
    add_borrowable(shelf, 1, old, 100 * MiB, { 4200 });
    // a new chat with the same 4.2K opening shares 34%: the 25% rule picks it, but it diverges
    auto t = shelf.take(b, st(toks(0, 4200, 200000, 300)), false);
    CHECK(t.id == 1 && t.borrow == -1 && t.shared == 4200 && t.keep);
    // the old chat returning, longer: the whole entry is shared, the bookmark leaves
    t = shelf.take(b, st(extend(old, 500000, 40)), false);
    CHECK(t.id == 1 && t.borrow == -1 && t.shared == (int64_t) old.size() && !t.keep);
    server_shelf big;
    add_borrowable(big, 1, toks(0, 4200, 100000, 45800), 1861813976ull, { 4200 });
    t = big.take(b, st(toks(0, 4200, 200000, 300)), false);
    CHECK(t.id == 1 && t.borrow == 4200 && t.keep);

    // keep: the file stays, the entry is newest; without it both go
    const fs::path d = borrow_dir();
    server_shelf s2;
    s2.dir = d.string();
    for (uint64_t id = 1; id <= 2; id++) {
        touch(s2.path_for(id));
        s2.add(id, st(toks(0, 4200, 100000 * (llama_token) id, 8000)), s2.path_for(id), 100 * MiB, copies_at({ 4200 }), true);
    }
    const auto pk = s2.pick_take(b.tokens, st(toks(0, 4200, 100000, 100)));
    CHECK(pk.id == 1 && pk.keep);
    s2.hand_out(pk.id, pk.keep || pk.borrow >= 0);
    CHECK(fs::exists(s2.path_for(1)) && s2.entries.back().id == 1);
    s2.hand_out(2, s2.pick_take(b.tokens, st(extend(toks(0, 4200, 200000, 8000), 9, 5))).keep);
    CHECK(!fs::exists(s2.path_for(2)) && s2.entries.size() == 1);
    std::error_code ec;
    fs::remove_all(d, ec);
}

// invariants over generated shelves: a borrow qualifies, nothing qualifying beats it, and the pick comes first
static void test_borrow_generated() {
    std::mt19937 rng(41);
    const auto uni = [&](int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng); };

    int n_borrowed = 0;
    for (int iter = 0; iter < 3000; iter++) {
        const int n_open = (int) uni(1, 6000);
        const llama_tokens task = toks(0, n_open, 900000, (int) uni(0, 400));

        server_shelf shelf;
        const int n_entries = (int) uni(0, 6);
        for (int i = 0; i < n_entries; i++) {
            const int s = (int) uni(0, 7000);
            const int r = (int) uni(1, 3000);
            std::vector<int64_t> at;
            const int n_copies = (int) uni(0, 3);
            for (int k = 0; k < n_copies; k++) {
                at.push_back(uni(1, s + r));
            }
            std::sort(at.begin(), at.end());
            add_borrowable(shelf, (uint64_t) i + 1, toks(0, s, 1000000 + 100000 * i, r),
                    (uint64_t) uni(1, 6000) * MiB, at, uni(0, 9) > 0);
        }
        const int s_seat = (int) uni(0, 6000);
        std::vector<int64_t> at_seat;
        if (uni(0, 1)) {
            at_seat.push_back(uni(1, s_seat + 100));
        }
        const server_prompt seat = seat_of(toks(0, s_seat, 2000000, 100), at_seat);
        const bool cuts = uni(0, 3) == 0;

        const int64_t reach = server_shelf_seat_reach(seat, st(task), cuts);
        const auto qualifies = [&](const server_shelf_entry & e, int64_t p) {
            const int64_t shared = (int64_t) e.tokens.get_common_prefix(st(task));
            return e.spec_synced && p > 0 && p <= shared && p < (int64_t) task.size() && p > reach &&
                   server_shelf_borrow_saves(p - reach, e.size);
        };

        const auto t = shelf.borrow_pick(seat, st(task), cuts);
        const server_shelf_entry * won = nullptr;
        for (const auto & e : shelf.entries) {
            if (e.id == t.id) {
                won = &e;
            }
        }
        if (t.id != 0) {
            n_borrowed++;
            CHECK(won != nullptr);
            CHECK(std::find(won->copies.begin(), won->copies.end(), t.borrow) != won->copies.end());
            CHECK(qualifies(*won, t.borrow));
            CHECK(t.shared == (int64_t) won->tokens.get_common_prefix(st(task)));
        } else {
            CHECK(t.borrow == -1);
        }
        for (const auto & e : shelf.entries) {
            for (const int64_t p : e.copies) {
                if (!qualifies(e, p)) {
                    continue;
                }
                CHECK(t.id != 0); // something qualified, so something is borrowed
                if (t.id != 0) {
                    CHECK(p < t.borrow || (p == t.borrow && e.size >= won->size));
                }
            }
        }

        // the 25% rule first: whenever it picks, take is that pick, not a borrow
        const uint64_t pick = shelf.pick(seat.tokens, st(task));
        const auto tk = shelf.take(seat, st(task), cuts);
        CHECK(pick != 0 ? (tk.id == pick && tk.borrow == -1) : (tk.id == t.id && tk.borrow == t.borrow));
    }
    std::printf("test-server-shelf: borrowing, %d of 3000 generated shelves borrowed\n", n_borrowed);
    CHECK(n_borrowed > 100); // the generator reaches borrows
}

static fs::path borrow_dir() {
    const fs::path d = fs::temp_directory_path() / ("test-server-shelf-borrow-" + std::to_string((long) getpid()));
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d;
}

static void touch(const std::string & p) {
    std::ofstream(p) << "x";
}

static std::vector<uint64_t> ids(const server_shelf & shelf) {
    std::vector<uint64_t> res;
    for (const auto & e : shelf.entries) {
        res.push_back(e.id);
    }
    return res;
}

static void test_hand_out() {
    const fs::path d = borrow_dir();
    server_shelf shelf;
    shelf.dir = d.string();
    const llama_tokens a = toks(0, 4200, 100000, 800);
    for (uint64_t id = 1; id <= 3; id++) {
        touch(shelf.path_for(id));
        shelf.add(id, st(id == 1 ? a : toks(0, 4200, 100000 * (llama_token) id + 100000, 800)), shelf.path_for(id),
                4 * GB, copies_at({ 4200 }), true);
    }

    // a borrow of the oldest (1): its tokens handed out, the entry and its file stay, now the newest
    server_tokens got = shelf.hand_out(1, true);
    CHECK(got.get_common_prefix(st(a)) == a.size() && got.size() == a.size());
    CHECK(ids(shelf) == std::vector<uint64_t>({ 2, 3, 1 }));
    CHECK(fs::exists(shelf.path_for(1)));
    CHECK(shelf.find(1) != nullptr && shelf.find(1)->tokens.size() == a.size());
    CHECK(shelf.find(1) != nullptr && shelf.find(1)->copies == std::vector<int64_t>({ 4200 }));

    // room for a save now drops 2, the oldest, not the borrowed opening
    shelf.free_bytes = [](uint64_t & out) { out = 1000 * GB; return true; };
    shelf.lim = server_shelf_cap_limit(12 * GB, 1000 * GB);
    const auto r = shelf.room_for(st(toks(0, 3, 300000, 10)), 4 * GB, 0);
    CHECK(r.fits && shelf.ids_at(r.drop) == std::vector<uint64_t>({ 2 }));

    // an ordinary load of 3: it leaves the shelf and its file goes
    got = shelf.hand_out(3, false);
    CHECK(got.size() == 5000);
    CHECK(ids(shelf) == std::vector<uint64_t>({ 2, 1 }));
    CHECK(!fs::exists(shelf.path_for(3)));
    // an id not on the shelf hands out nothing and changes nothing
    CHECK(shelf.hand_out(9, true).size() == 0 && ids(shelf) == std::vector<uint64_t>({ 2, 1 }));

    std::error_code ec;
    fs::remove_all(d, ec);
}

// the positions an entry holds are the copies its file gives back, so the borrow's choice is the copy the load brings
static void test_copy_positions() {
    std::list<common_prompt_checkpoint> ckpts = copies_at({ 4211, 5211, 6211 });
    int i = 0;
    for (auto & c : ckpts) {
        c.data_tgt.assign(1000 + i, (uint8_t) i);
        c.pinned = i++ == 0;
    }
    CHECK(server_shelf_copy_positions(ckpts) == std::vector<int64_t>({ 4211, 5211, 6211 }));
    CHECK(server_shelf_copy_positions({}).empty());

    const fs::path d = borrow_dir();
    server_shelf shelf;
    shelf.dir = d.string();
    const std::string path = shelf.path_for(1);
    const std::vector<uint8_t> tgt(4096, 7);
    {
        server_shelf_file f;
        CHECK(f.open_write(path));
        CHECK(server_shelf_write_record(f, {}, tgt.size(),
                [&](server_shelf_file & ff, uint64_t) { return ff.write(tgt.data(), tgt.size()); },
                0, nullptr, {}, ckpts));
        CHECK(f.close());
    }
    shelf.add(1, st(toks(0, 7000, 0, 0)), path, fs::file_size(path), ckpts, false);
    CHECK(shelf.entries.back().copies == std::vector<int64_t>({ 4211, 5211, 6211 }));
    CHECK(!shelf.entries.back().spec_synced);

    server_shelf_file f;
    server_shelf_head head;
    std::vector<uint8_t> spec;
    std::list<common_prompt_checkpoint> back;
    uint64_t size_dft = 0;
    CHECK(f.open_read(path));
    CHECK(server_shelf_read_record(f, head,
            [](server_shelf_file & ff, uint64_t size) { std::vector<uint8_t> v(size); return ff.read(v.data(), size); },
            nullptr, size_dft, spec, back));
    CHECK(server_shelf_copy_positions(back) == shelf.entries.back().copies);

    std::error_code ec;
    fs::remove_all(d, ec);
}

void test_borrow() {
    test_borrow_saves();
    test_seat_reach();
    test_borrow_choice();
    test_pick_keeps_when_diverging();
    test_borrow_generated();
    test_hand_out();
    test_copy_positions();
}

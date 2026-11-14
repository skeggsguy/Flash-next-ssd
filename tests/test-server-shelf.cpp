// The bookmark shelf's rules (--cache-disk, tools/server/server-shelf.h, study BOOKMARK-SHELF-PLAN.md step 2),
// without a model.
//
// - the limit at open: the asked GiB, capped to the drive's free space less a 10% reserve
// - the room a save needs: superseded bookmarks always go, then the oldest until the shelf is within its limit and
//   the drive keeps its reserve (a drive filled up by something else drops more, or skips the save); the bookmark
//   about to be loaded is never dropped; a save that cannot fit drops nothing. Examples, and invariants over
//   generated shelves
// - the choice: the store's rule (server_prompt_cache::load), the 25% rule, an empty seat, a seat that already
//   holds the conversation; and `holds` (a save that would add nothing)
// - room_for on a shelf with an injected drive
// - the hand-over: save when the task keeps under half of the seat or a bookmark replaces it, load the pick
// - the two options: off by default, 100 GiB, from the command line and environment, 0 refused
// - in test-server-shelf-file.cpp: the files (straight through, the file cache bypassed), the record and the folder

#include "test-server-shelf.h"

#include "arg.h"

#include <algorithm>
#include <cstdlib>
#include <random>

int n_fail = 0;

static const uint64_t GiB = 1024ull * 1024ull * 1024ull;
static const uint64_t MiB = 1024ull * 1024ull;

static void test_cap_limit() {
    // the annex: 1.3 TB free, 300 asked: under the free space less 10%, not capped
    auto l = server_shelf_cap_limit(300 * GiB, 1300 * GiB);
    CHECK(!l.capped);
    CHECK(l.limit == 300 * GiB);
    CHECK(l.reserve == 130 * GiB);

    // a small internal drive: 60 GiB free, 100 asked: capped to 54, 6 kept free
    l = server_shelf_cap_limit(100 * GiB, 60 * GiB);
    CHECK(l.capped);
    CHECK(l.limit == 54 * GiB);
    CHECK(l.reserve == 6 * GiB);

    // exactly the usable space is not a cap
    l = server_shelf_cap_limit(90 * GiB, 100 * GiB);
    CHECK(!l.capped && l.limit == 90 * GiB);

    // a full drive: nothing to use
    l = server_shelf_cap_limit(100 * GiB, 0);
    CHECK(l.capped && l.limit == 0);
}

static std::vector<server_shelf_room_item> items_of(const std::vector<uint64_t> & sizes) {
    std::vector<server_shelf_room_item> items;
    for (uint64_t s : sizes) {
        server_shelf_room_item it;
        it.size = s;
        items.push_back(it);
    }
    return items;
}

static void test_make_room_examples() {
    const uint64_t big = 1000 * GiB; // a drive with room to spare

    // room under the limit: nothing dropped
    auto r = server_shelf_make_room(items_of({10, 20}), 30, 100, big, 0);
    CHECK(r.fits && r.drop.empty());

    // over the limit: the oldest go first, only as many as needed
    r = server_shelf_make_room(items_of({40, 30, 20}), 30, 100, big, 0);
    CHECK(r.fits && r.drop == std::vector<size_t>({0}));
    r = server_shelf_make_room(items_of({40, 30, 20}), 75, 100, big, 0);
    CHECK(r.fits && r.drop == std::vector<size_t>({0, 1}));

    // bigger than the limit itself: never fits, nothing dropped
    r = server_shelf_make_room(items_of({40, 30}), 101, 100, big, 0);
    CHECK(!r.fits && r.drop.empty());

    // the bookmark about to be loaded is never dropped: the next oldest goes instead
    auto items = items_of({40, 30, 20});
    items[0].keep = true;
    r = server_shelf_make_room(items, 30, 100, big, 0);
    CHECK(r.fits && r.drop == std::vector<size_t>({1}));
    // ... and when only dropping it would make room, the save is skipped
    items = items_of({90});
    items[0].keep = true;
    r = server_shelf_make_room(items, 20, 100, big, 0);
    CHECK(!r.fits && r.drop.empty());

    // a superseded bookmark (the same conversation, shorter) goes whenever the save goes ahead, even with room
    items = items_of({10, 20, 30});
    items[1].superseded = true;
    r = server_shelf_make_room(items, 5, 100, big, 0);
    CHECK(r.fits && r.drop == std::vector<size_t>({1}));
    // ... and it counts toward the room: dropping it is enough here
    r = server_shelf_make_room(items, 60, 100, big, 0);
    CHECK(r.fits && r.drop == std::vector<size_t>({1}));
    // ... but not when the save cannot fit
    r = server_shelf_make_room(items, 500, 100, big, 0);
    CHECK(!r.fits && r.drop.empty());

    // the drive filled up by something else: 50 free with 10 kept free, so a 45 save needs 5 dropped
    r = server_shelf_make_room(items_of({4, 4, 4}), 45, 1000, 50, 10);
    CHECK(r.fits && r.drop == std::vector<size_t>({0, 1}));
    // ... and with too little on the shelf to drop, it is skipped and nothing goes
    r = server_shelf_make_room(items_of({4, 4, 4}), 60, 1000, 50, 10);
    CHECK(!r.fits && r.drop.empty());
    // an empty shelf on a full drive
    r = server_shelf_make_room({}, 1, 1000, 10, 10);
    CHECK(!r.fits);
}

// invariants over generated shelves: what is dropped is exactly enough, oldest first, never the kept one
static void test_make_room_generated() {
    std::mt19937 rng(14);
    std::uniform_int_distribution<int> n_items(0, 12);
    std::uniform_int_distribution<uint64_t> size(1, 50);
    std::uniform_int_distribution<int> coin(0, 9);

    for (int iter = 0; iter < 20000; iter++) {
        std::vector<server_shelf_room_item> items;
        const int n = n_items(rng);
        for (int i = 0; i < n; i++) {
            server_shelf_room_item it;
            it.size       = size(rng);
            it.superseded = coin(rng) == 0;
            items.push_back(it);
        }
        if (n > 0 && coin(rng) < 3) {
            items[std::uniform_int_distribution<int>(0, n - 1)(rng)].keep = true;
        }
        const uint64_t size_new = size(rng) * (coin(rng) == 0 ? 4 : 1);
        const uint64_t limit    = std::uniform_int_distribution<uint64_t>(1, 400)(rng);
        const uint64_t free_now = std::uniform_int_distribution<uint64_t>(0, 400)(rng);
        const uint64_t reserve  = std::uniform_int_distribution<uint64_t>(0, 40)(rng);

        const auto r = server_shelf_make_room(items, size_new, limit, free_now, reserve);

        uint64_t used = 0, droppable = 0;
        for (const auto & it : items) {
            used += it.size;
            droppable += it.keep ? 0 : it.size;
        }
        const auto fits_after = [&](uint64_t freed) {
            return used - freed + size_new <= limit && free_now + freed >= size_new + reserve;
        };

        if (!r.fits) {
            CHECK(r.drop.empty());
            CHECK(!fits_after(droppable)); // not even dropping everything would do
            continue;
        }

        uint64_t freed = 0;
        std::vector<bool> dropped(items.size(), false);
        for (size_t i : r.drop) {
            CHECK(i < items.size() && !dropped[i]);
            dropped[i] = true;
            freed += items[i].size;
        }
        CHECK(fits_after(freed));
        size_t last_plain = 0;
        bool   any_plain  = false;
        for (size_t i = 0; i < items.size(); i++) {
            CHECK(!(items[i].keep && dropped[i]));
            if (items[i].superseded && !items[i].keep) {
                CHECK(dropped[i]);
            }
            if (dropped[i] && !items[i].superseded) {
                last_plain = i;
                any_plain  = true;
            }
        }
        if (any_plain) {
            // oldest first: every droppable plain bookmark older than the last dropped one was dropped too
            for (size_t i = 0; i < last_plain; i++) {
                if (!items[i].keep && !items[i].superseded) {
                    CHECK(dropped[i]);
                }
            }
            // and no more than needed: without the last one it would not fit
            CHECK(!fits_after(freed - items[last_plain].size));
        }
    }
}

static void test_pick() {
    server_shelf shelf;
    // A: a long coding conversation; B: a chat; both start with the same 100-token system prompt
    const llama_tokens a = toks(0, 100, 1000, 5000);
    const llama_tokens b = toks(0, 100, 2000, 800);
    add_entry(shelf, 1, a, 50 * MiB);
    add_entry(shelf, 2, b, 30 * MiB);

    // A returns with a follow-up while the seat holds C (a sub-agent sharing the system prompt): A is picked
    const llama_tokens c = toks(0, 100, 3000, 900);
    CHECK(shelf.pick(st(c), st(extend(a, 4000, 50))) == 1);
    // B returns: B
    CHECK(shelf.pick(st(c), st(extend(b, 4000, 50))) == 2);
    // an empty seat: the best entry
    CHECK(shelf.pick(st({}), st(extend(a, 4000, 50))) == 1);
    // a new conversation sharing only the system prompt: under 25% of either entry, none
    CHECK(shelf.pick(st({}), st(toks(0, 100, 5000, 400))) == 0);
    // the seat already holds A (a continuation): nothing beats it
    CHECK(shelf.pick(st(a), st(extend(a, 4000, 50))) == 0);
    // the seat holds the start of A and then went elsewhere: the shelf's A keeps and covers more
    CHECK(shelf.pick(st(extend(llama_tokens(a.begin(), a.begin() + 3000), 6000, 500)), st(extend(a, 4000, 50))) == 1);
    // a seat holding a start of the request is never replaced: the request carries on from it (the store's rule)
    CHECK(shelf.pick(st(llama_tokens(a.begin(), a.begin() + 3000)), st(extend(a, 4000, 50))) == 0);
    // an empty request picks nothing
    CHECK(shelf.pick(st(c), st({})) == 0);

    // the 25% rule: an entry keeping exactly a quarter of itself is still taken, under it not
    server_shelf s2;
    add_entry(s2, 7, toks(0, 25, 9000, 75), MiB);
    CHECK(s2.pick(st({}), st(toks(0, 25, 7000, 10))) == 7);
    server_shelf s3;
    add_entry(s3, 8, toks(0, 24, 9000, 76), MiB);
    CHECK(s3.pick(st({}), st(toks(0, 24, 7000, 10))) == 0);

    // both shares must beat the seat's: a short entry keeping almost all of itself but covering little of the
    // request loses to a seat covering most of it, though the seat keeps less of itself; on an empty seat it wins
    server_shelf s4;
    add_entry(s4, 9, extend(llama_tokens(a.begin(), a.begin() + 300), 8000, 10), MiB);
    const llama_tokens seat_mostly_a = extend(llama_tokens(a.begin(), a.begin() + 4000), 6000, 3000);
    CHECK(s4.pick(st(seat_mostly_a), st(extend(a, 4000, 50))) == 0);
    CHECK(s4.pick(st({}), st(extend(a, 4000, 50))) == 9);

    // holds: a save of A or of any start of A adds nothing; A grown does
    CHECK(shelf.holds(st(a)));
    CHECK(shelf.holds(st(llama_tokens(a.begin(), a.begin() + 200))));
    CHECK(!shelf.holds(st(extend(a, 4000, 1))));
    CHECK(!shelf.holds(st(c)));
}

static void test_room_for() {
    server_shelf shelf;
    uint64_t drive_free = 100 * GiB;
    shelf.free_bytes = [&](uint64_t & out) { out = drive_free; return true; };
    shelf.lim = server_shelf_cap_limit(10 * GiB, drive_free);

    const llama_tokens a = toks(0, 100, 1000, 500);
    add_entry(shelf, 1, a, 4 * GiB);
    add_entry(shelf, 2, toks(0, 100, 2000, 500), 4 * GiB);

    // A grown supersedes A: A goes, and B stays
    auto r = shelf.room_for(st(extend(a, 4000, 100)), 5 * GiB, 0);
    CHECK(r.fits && shelf.ids_at(r.drop) == std::vector<uint64_t>({1}));
    // a new conversation of 5 GiB: the oldest (A) goes
    r = shelf.room_for(st(toks(0, 100, 3000, 500)), 5 * GiB, 0);
    CHECK(r.fits && shelf.ids_at(r.drop) == std::vector<uint64_t>({1}));
    // ... unless A is about to be loaded: then B goes
    r = shelf.room_for(st(toks(0, 100, 3000, 500)), 5 * GiB, 1);
    CHECK(r.fits && shelf.ids_at(r.drop) == std::vector<uint64_t>({2}));

    // the drive filled up from elsewhere since the shelf opened: 12 GiB free, 10 kept free. A 6 needs 4 freed
    // (the oldest), a 7 both
    drive_free = 12 * GiB;
    r = shelf.room_for(st(toks(0, 100, 3000, 500)), 6 * GiB, 0);
    CHECK(r.fits && shelf.ids_at(r.drop) == std::vector<uint64_t>({1}));
    r = shelf.room_for(st(toks(0, 100, 3000, 500)), 7 * GiB, 0);
    CHECK(r.fits && shelf.ids_at(r.drop) == std::vector<uint64_t>({1, 2}));
    drive_free = 5 * GiB;
    r = shelf.room_for(st(toks(0, 100, 3000, 500)), 2 * GiB, 0);
    CHECK(r.fits && shelf.ids_at(r.drop) == std::vector<uint64_t>({1, 2})); // 5 + 8 >= 2 + 10 only with both gone
    // ... and a 15 cannot fit at all: nothing goes
    r = shelf.room_for(st(toks(0, 100, 3000, 500)), 15 * GiB, 0);
    CHECK(!r.fits && r.drop.empty());

    // a drive whose free space cannot be read: no save
    shelf.free_bytes = [](uint64_t &) { return false; };
    r = shelf.room_for(st(toks(0, 100, 3000, 500)), 1, 0);
    CHECK(!r.fits && r.drop.empty());
}

static bool same(server_shelf_hand h, bool save, bool load) {
    return h.save == save && h.load == load;
}

static void test_hand_on() {
    // a follow-up in the same conversation: neither, whatever else holds
    CHECK(same(server_shelf_hand_on(50000, 50000, true, 0), false, false));
    CHECK(same(server_shelf_hand_on(50000, 50000, false, 7), false, false));
    CHECK(server_shelf_carries_on(50000, 50000) && !server_shelf_carries_on(50000, 49999) && !server_shelf_carries_on(0, 0));
    // B (code, sharing only the system prompt) takes A's seat, nothing of B on the shelf: A is saved
    CHECK(same(server_shelf_hand_on(50000, 4211, true, 0), true, false));
    // A returns to B's seat with A on the shelf: B is saved and A loaded
    CHECK(same(server_shelf_hand_on(30000, 4211, true, 1), true, true));
    // an empty seat: nothing to save, the pick is loaded
    CHECK(same(server_shelf_hand_on(0, 0, true, 1), false, true));
    CHECK(same(server_shelf_hand_on(0, 0, true, 0), false, false));
    // the task keeps more than half of the seat and nothing better is on the shelf: the seat carries on as is
    CHECK(same(server_shelf_hand_on(30000, 17000, false, 0), false, false));
    // ... but when a bookmark replaces it, it is saved first
    CHECK(same(server_shelf_hand_on(30000, 17000, false, 2), true, true));
    // a slot asked for by id: the slot choice did not flag the loss, the shelf still saves a seat keeping under half
    CHECK(same(server_shelf_hand_on(30000, 14999, false, 0), true, false));
    CHECK(same(server_shelf_hand_on(30000, 15000, false, 0), false, false));
}

static bool parse(std::vector<const char *> argv, common_params & params) {
    argv.insert(argv.begin(), "llama-server");
    return common_params_parse((int) argv.size(), const_cast<char **>(argv.data()), params, LLAMA_EXAMPLE_SERVER);
}

static void test_options() {
    {
        common_params p;
        CHECK(parse({}, p));
        CHECK(p.cache_disk_dir.empty()); // off
        CHECK(p.cache_disk_gib == 100);
    }
    {
        common_params p;
        CHECK(parse({"--cache-disk", "/Volumes/annex/shelf", "--cache-disk-gib", "300"}, p));
        CHECK(p.cache_disk_dir == "/Volumes/annex/shelf");
        CHECK(p.cache_disk_gib == 300);
    }
    {
        common_params p;
        CHECK(!parse({"--cache-disk-gib", "0"}, p));
    }
    {
        setenv("LLAMA_ARG_CACHE_DISK", "/tmp/shelf", 1);
        setenv("LLAMA_ARG_CACHE_DISK_GIB", "20", 1);
        common_params p;
        CHECK(parse({}, p));
        CHECK(p.cache_disk_dir == "/tmp/shelf");
        CHECK(p.cache_disk_gib == 20);
        unsetenv("LLAMA_ARG_CACHE_DISK");
        unsetenv("LLAMA_ARG_CACHE_DISK_GIB");
    }
}

int main() {
    test_cap_limit();
    test_make_room_examples();
    test_make_room_generated();
    test_pick();
    test_room_for();
    test_hand_on();
    test_options();
    test_files();
    test_record();
    test_folder();

    if (n_fail > 0) {
        std::fprintf(stderr, "test-server-shelf: %d failures\n", n_fail);
        return 1;
    }
    std::printf("test-server-shelf: all passed\n");
    return 0;
}

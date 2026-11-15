// The same conversation, judged by where its reply began (test-server-shelf's fourth file; tools/server/server-shelf.h
// reply_from, study SHELF-BORROW-PLAN.md step 5), without a model:
//
// - the anchor: the final assistant span's position (the generation prompt); none, or a final user or tool span:
//   the task's last 4 tokens; always within the task
// - the load (pick_take): a return shares at least the old reply's start and takes the file off, even though the
//   template re-rendered the reply (Tom's 40K prompt + 10K thinking; the borrow probe's A2, sharing exactly up to
//   it); another conversation parts before it and keeps the file (a new chat at 34% of a short one, an edited last
//   message, a sub-agent sharing a 20K system prompt); a whole continuation is a return; an entry without an anchor
//   is judged as before (only a request continuing all of it returns)
// - the save (room_for): an entry the save shares its reply's start with is superseded, one it parts from before
//   that is not; the entry about to be loaded is never dropped; holds() stays exact
// - invariants over generated shelves: a whole continuation never keeps, a request parting before the anchor always
//   keeps, and the save supersedes exactly the entries a load would take off

#include "test-server-shelf.h"

#include <algorithm>
#include <random>

static const uint64_t MiB = 1024ull * 1024ull;
static const uint64_t GiB = 1024ull * 1024ull * 1024ull;

static common_chat_msg_spans spans_of(std::initializer_list<common_chat_msg_span> s) {
    common_chat_msg_spans res;
    res.spans = s;
    return res;
}

// a bookmark of tokens t whose reply began at reply_from (-1: not known), as a save adds it
static void add_reply(server_shelf & shelf, uint64_t id, const llama_tokens & t, int64_t reply_from,
        uint64_t size = 100 * MiB) {
    shelf.add(id, st(t), "", size, {}, true, reply_from);
}

// the seat when the shelf is asked: code, parting from chat at token 3 (so no seat beats an entry)
static server_tokens code_seat() {
    return st(toks(0, 3, 300000, 2000));
}

static void test_anchor() {
    // the generation prompt: the final span is the assistant's, its position is where the reply begins (here 6 tokens
    // before the end, an opened thinking block, so not the last-4 fallback's answer)
    CHECK(server_shelf_reply_from(spans_of({ { COMMON_CHAT_ROLE_SYSTEM, 0, 4200 }, { COMMON_CHAT_ROLE_USER, 4200, 300 },
            { COMMON_CHAT_ROLE_ASSISTANT, 4500, 6 } }), 4506) == 4500);
    // an assistant span earlier in the chat does not count when a user or tool message comes last: the last 4 tokens
    CHECK(server_shelf_reply_from(spans_of({ { COMMON_CHAT_ROLE_ASSISTANT, 100, 50 },
            { COMMON_CHAT_ROLE_USER, 150, 50 } }), 200) == 196);
    CHECK(server_shelf_reply_from(spans_of({ { COMMON_CHAT_ROLE_ASSISTANT, 100, 50 },
            { COMMON_CHAT_ROLE_TOOL, 150, 50 } }), 200) == 196);
    // no spans (a client sending no message delimiters): the last 4 tokens, never below 0
    CHECK(server_shelf_reply_from({}, 48362) == 48358);
    CHECK(server_shelf_reply_from({}, 4) == 0);
    CHECK(server_shelf_reply_from({}, 3) == 0);
    CHECK(server_shelf_reply_from({}, 0) == 0);
    CHECK(server_shelf_reply_from({}, -7) == 0);
    // a span position past the task is held to the task
    CHECK(server_shelf_reply_from(spans_of({ { COMMON_CHAT_ROLE_ASSISTANT, 900, 4 } }), 500) == 500);

    // an entry's reply_start: its anchor, or the whole bookmark when not known or past its end
    server_shelf shelf;
    add_reply(shelf, 1, toks(0, 1000, 0, 0), 600);
    add_reply(shelf, 2, toks(0, 1000, 0, 0), -1);
    add_reply(shelf, 3, toks(0, 1000, 0, 0), 5000);
    add_entry(shelf, 4, toks(0, 1000, 0, 0), MiB); // an entry made without one
    CHECK(shelf.find(1)->reply_start() == 600);
    CHECK(shelf.find(2)->reply_start() == 1000);
    CHECK(shelf.find(3)->reply_start() == 1000);
    CHECK(shelf.find(4)->reply_start() == 1000);
    // a seat with no anchor known saves as the whole seat (the save clamps it to [0, n_tokens])
    CHECK(server_shelf_seat().reply_from == INT64_MAX);
}

static void test_pick() {
    // Tom's case: 50K = a 40K prompt and 10K of thinking. The return's prompt re-renders the reply (the thinking is
    // not sent back), so it shares the prompt and the reply's 4-token start, then differs: a return, taken off
    const llama_tokens a = toks(0, 40000, 100000, 10000);
    server_shelf shelf;
    add_reply(shelf, 1, a, 39996);
    const llama_tokens back = extend(toks(0, 40000, 100000, 4), 700000, 900);
    auto t = shelf.pick_take(code_seat(), st(back));
    CHECK(t.id == 1 && t.shared == 40004 && !t.keep);

    // the borrow probe's A2: A1's prompt was 48,362 tokens, its reply 255, and A2 shares exactly up to the reply's
    // start (48,358 of 48,617): a return, taken off
    server_shelf a2;
    add_reply(a2, 1, toks(0, 48358, 100000, 259), 48358);
    t = a2.pick_take(code_seat(), st(extend(toks(0, 48358, 0, 0), 800000, 144)));
    CHECK(t.id == 1 && t.shared == 48358 && !t.keep);
    // ... one token short of it is another conversation
    t = a2.pick_take(code_seat(), st(extend(toks(0, 48357, 0, 0), 800000, 144)));
    CHECK(t.id == 1 && t.shared == 48357 && t.keep);

    // kept: a new chat with the same 4.2K opening, 34% of a short old chat whose reply began at 12,000
    server_shelf s34;
    add_reply(s34, 1, toks(0, 4200, 100000, 8000), 12000);
    t = s34.pick_take(code_seat(), st(toks(0, 4200, 200000, 300)));
    CHECK(t.id == 1 && t.shared == 4200 && t.keep);
    // kept: the last user message edited (it began at 19,000; the reply at 20,000), the edit shares 19,100
    server_shelf edit;
    add_reply(edit, 1, toks(0, 19100, 100000, 10900), 20000);
    t = edit.pick_take(code_seat(), st(toks(0, 19100, 200000, 950)));
    CHECK(t.id == 1 && t.shared == 19100 && t.keep);
    // kept: a coding sub-agent sharing the main agent's 20K system prompt (33% of a 60K bookmark)
    server_shelf sub;
    add_reply(sub, 1, toks(0, 20000, 100000, 40000), 58000);
    t = sub.pick_take(code_seat(), st(toks(0, 20000, 200000, 3000)));
    CHECK(t.id == 1 && t.shared == 20000 && t.keep);

    // a whole continuation is a return, wherever the anchor sits
    t = s34.pick_take(code_seat(), st(extend(toks(0, 4200, 100000, 8000), 900000, 40)));
    CHECK(t.id == 1 && t.shared == 12200 && !t.keep);

    // an entry without an anchor is judged as before: Tom's return keeps it, only all of it shared takes it off
    server_shelf none;
    add_reply(none, 1, a, -1);
    t = none.pick_take(code_seat(), st(back));
    CHECK(t.id == 1 && t.keep);
    t = none.pick_take(code_seat(), st(extend(a, 900000, 40)));
    CHECK(t.id == 1 && !t.keep);
    server_shelf raw;
    add_entry(raw, 1, a, 100 * MiB);
    CHECK(raw.pick_take(code_seat(), st(back)).keep);

    // take() goes through pick_take: the same verdicts, no borrow
    server_prompt seat;
    seat.tokens = code_seat();
    t = shelf.take(seat, st(back), false);
    CHECK(t.id == 1 && t.borrow == -1 && !t.keep);
}

static void test_room() {
    server_shelf shelf;
    shelf.free_bytes = [](uint64_t & out) { out = 1000 * GiB; return true; };
    shelf.lim = server_shelf_cap_limit(100 * GiB, 1000 * GiB);

    const llama_tokens a = toks(0, 40000, 100000, 10000);
    const llama_tokens back = extend(toks(0, 40000, 100000, 4), 700000, 900);
    add_reply(shelf, 1, a, 39996, 2 * GiB);
    add_reply(shelf, 2, toks(0, 4200, 200000, 8000), 12000, 1 * GiB);

    // the seat saved after A's return (A's conversation further on) supersedes A's old file, though not all of it
    // is in the save; the other conversation stays
    auto r = shelf.room_for(st(back), 3 * GiB, 0);
    CHECK(r.fits && shelf.ids_at(r.drop) == std::vector<uint64_t>({ 1 }));
    // ... and up to exactly the reply's start counts
    r = shelf.room_for(st(extend(toks(0, 39996, 0, 0), 600000, 10)), 3 * GiB, 0);
    CHECK(r.fits && shelf.ids_at(r.drop) == std::vector<uint64_t>({ 1 }));
    // a save parting before the reply began (a new chat sharing the opening, an edit one token short): nothing goes
    r = shelf.room_for(st(toks(0, 4200, 300000, 500)), 3 * GiB, 0);
    CHECK(r.fits && r.drop.empty());
    r = shelf.room_for(st(extend(toks(0, 39995, 0, 0), 600000, 10)), 3 * GiB, 0);
    CHECK(r.fits && r.drop.empty());
    // the entry about to be loaded is never dropped, superseded or not
    r = shelf.room_for(st(back), 3 * GiB, 1);
    CHECK(r.fits && r.drop.empty());
    // holds() stays exact: A's return is not on the shelf, so it is saved
    CHECK(!shelf.holds(st(back)));
    CHECK(shelf.holds(st(toks(0, 40000, 100000, 10000))));

    // an entry without an anchor: superseded only when all of it is in the save, as before
    server_shelf old;
    old.free_bytes = shelf.free_bytes;
    old.lim = shelf.lim;
    add_reply(old, 1, a, -1, 2 * GiB);
    r = old.room_for(st(back), 3 * GiB, 0);
    CHECK(r.fits && r.drop.empty());
    r = old.room_for(st(extend(a, 600000, 10)), 3 * GiB, 0);
    CHECK(r.fits && old.ids_at(r.drop) == std::vector<uint64_t>({ 1 }));
}

// invariants over generated shelves: whatever the anchor, a whole continuation is a return and a request parting
// before the anchor is another conversation; and with room to spare a save supersedes exactly what a load of the
// same request would take off (the two judge the same conversation the same way)
static void test_generated() {
    std::mt19937 rng(55);
    const auto uni = [&](int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng); };

    int n_return = 0;
    int n_other  = 0;
    for (int iter = 0; iter < 3000; iter++) {
        server_shelf shelf;
        shelf.free_bytes = [](uint64_t & out) { out = 1000 * GiB; return true; };
        shelf.lim = server_shelf_cap_limit(500 * GiB, 1000 * GiB);

        const int n_entry = (int) uni(1, 5000);
        const llama_tokens e = toks(0, n_entry, 0, 0);
        const int64_t anchor = uni(0, 9) == 0 ? -1 : uni(0, n_entry + 50); // sometimes unknown or past the end
        add_reply(shelf, 1, e, anchor, MiB);
        const int64_t start = shelf.find(1)->reply_start();
        CHECK(start >= 0 && start <= n_entry);

        // the task shares a random part of the entry (or all of it), then goes its own way
        const int n_shared = (int) uni(0, n_entry);
        const llama_tokens task = extend(toks(0, n_shared, 0, 0), 5000000, (int) uni(1, 300));
        const auto t = shelf.pick_take(code_seat(), st(task));
        if (t.id == 0) {
            continue; // under the 25% rule: nothing to judge
        }
        CHECK(t.shared == n_shared);
        if (n_shared == n_entry) {
            CHECK(!t.keep);
        }
        if (n_shared < start) {
            CHECK(t.keep);
        }
        (t.keep ? n_other : n_return)++;

        const auto r = shelf.room_for(st(task), MiB, 0);
        CHECK(r.fits);
        CHECK(r.drop.empty() == t.keep);
        // the entry about to be loaded stays, whatever the verdict
        CHECK(shelf.room_for(st(task), MiB, 1).drop.empty());
    }
    std::printf("test-server-shelf: the same conversation, %d returns and %d others among generated picks\n",
            n_return, n_other);
    CHECK(n_return > 100 && n_other > 100); // the generator reaches both verdicts
}

void test_reply() {
    test_anchor();
    test_pick();
    test_room();
    test_generated();
}

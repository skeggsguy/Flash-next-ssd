// The opening copy's rules (--cache-ram-opening, tools/server/server-ckpt-pin.h, study CHAT-RETURN-PLAN.md step 1),
// without a model. Part of test-server-ckpt-pin (main in test-server-ckpt-pin.cpp).
//
// - rule 1, chat only: the store saves a seat whose last task was chat, never one whose last task was code, and only
//   a chat task takes from it; a chat seat handed to a chat (a follow-up, a retry, or a new chat, which keeps the
//   opening from the seat) neither saves nor loads, so the save fires only when coding takes the seat; an empty chat
//   seat looks in the store; with the switch off every hand-on saves and loads, as upstream
// - rule 2, opening only: the step back keeps the pinned copy alone, and nothing without a pin
// - one copy of the opening: the store's save sees none of the seat's copies (the entry carries none), and the seat
//   gets them all back, at the same addresses, whether the save succeeds, fails or throws
// - rule 3, a first chat pinned: with no pin yet, a chat's copy at its first user message takes the pin; and over
//   generated long first chats the pin stays there however many copies roll past, so the step back finds it,
//   where with the switch off the opening's copy rolls out and the step back finds nothing
//
// Sidekick's opening is 4,211 tokens in every measured chat (CHAT-RETURN step 2): the system block and the tools,
// then the first user message (its <memory> block first).

#include "test-server-ckpt-pin.h"

#include <algorithm>
#include <random>
#include <stdexcept>

static const int64_t OPENING = 4211;

// a first turn: system 0..4211, the first user message, the generation prompt's assistant opening
static common_chat_msg_spans first_turn(int64_t n_user) {
    return spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, (size_t) OPENING},
                     {COMMON_CHAT_ROLE_ASSISTANT, (size_t) (OPENING + n_user)}}, (size_t) (OPENING + n_user + 10));
}

// a follow-up: the first turn, its reply, a second user message and the generation prompt
static common_chat_msg_spans follow_up(int64_t reply_at, int64_t user2_at, int64_t n) {
    return spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, (size_t) OPENING},
                     {COMMON_CHAT_ROLE_ASSISTANT, (size_t) reply_at}, {COMMON_CHAT_ROLE_USER, (size_t) user2_at},
                     {COMMON_CHAT_ROLE_ASSISTANT, (size_t) (n - 10)}}, (size_t) n);
}

static bool same(server_ckpt_hand_on h, bool save, bool load) {
    return h.save == save && h.load == load;
}

static void test_opening_hand_on() {
    // the switch off: every hand-on saves the whole seat and loads, whatever the lanes (upstream)
    for (bool seat_chat : {false, true}) {
        for (bool task_chat : {false, true}) {
            for (int64_t n_seat : {0, 4821, 21500}) {
                CHECK(same(server_ckpt_opening_hand_on(false, seat_chat, task_chat, n_seat), true, true));
            }
        }
    }

    // scenario 1: a long chat A (21K) leaves the seat for code (~40K): saved, and code takes nothing from the store
    CHECK(same(server_ckpt_opening_hand_on(true, true,  false, 21000), true,  false));
    // ... then new chat B on the code seat: code is never saved, chat B looks in the store
    CHECK(same(server_ckpt_opening_hand_on(true, false, true,  40010), false, true));
    // scenario 5: code then code: nothing saved, nothing taken
    CHECK(same(server_ckpt_opening_hand_on(true, false, false, 40010), false, false));

    // scenario 4: a follow-up in the same chat: neither, the seat keeps its whole bookmark
    CHECK(same(server_ckpt_opening_hand_on(true, true, true, 21000), false, false));
    // scenario 6 (the fix round): a new chat right after a long chat (A 21.5K, B shares only the opening, f_keep < 0.5)
    // neither saves nor loads: the seat holds the pinned opening and the read-in's restore steps back to it. Saving
    // here put a second copy of the opening in the store beside the seat's own
    CHECK(same(server_ckpt_opening_hand_on(true, true, true, 21500), false, false));
    // an empty chat seat holds no opening: it looks in the store (and has nothing to save)
    CHECK(same(server_ckpt_opening_hand_on(true, true, true, 0), false, true));
    // an empty code seat: the lanes alone decide
    CHECK(same(server_ckpt_opening_hand_on(true, false, true, 0), false, true));
}

// generated: with the switch on, the lanes and the seat's size alone decide: a chat seat is saved only for a code task,
// a chat task takes from the store only from a code seat or an empty one, a code task never takes from it
static void test_opening_hand_on_generated() {
    std::mt19937 rng(20261012);
    int n_empty = 0;
    for (int i = 0; i < 20000; i++) {
        const bool seat_chat = rng() % 2;
        const bool task_chat = rng() % 2;
        const int64_t n_seat = rng() % 5 == 0 ? 0 : 1 + (int64_t) (rng() % 200000);
        n_empty += n_seat == 0;

        const auto h = server_ckpt_opening_hand_on(true, seat_chat, task_chat, n_seat);
        CHECK(h.save == (seat_chat && !task_chat));
        CHECK(h.load == (task_chat && (!seat_chat || n_seat == 0)));
        CHECK(!(h.save && h.load)); // one copy: never saved and taken back in the same hand-on
        CHECK(same(server_ckpt_opening_hand_on(false, seat_chat, task_chat, n_seat), true, true));
    }
    CHECK(n_empty > 1000 && n_empty < 19000); // both kinds of seat generated
}

static void test_opening_keep_pin() {
    // a long chat's list: the pin at the opening and two rolling copies
    server_ckpt_list ckpts;
    add(ckpts, true, 3, 8192, 1, OPENING, true);
    add(ckpts, true, 3, 8192, 5, 18000);
    add(ckpts, true, 3, 8192, 5, 20000);
    server_ckpt_list keep = ckpts;
    auto * pin = server_ckpt_keep_pin(keep, 21000);
    CHECK(pin != nullptr && pin->n_tokens == OPENING && pin->pinned);
    CHECK(keep.size() == 1 && &keep.front() == pin);

    // the pin in the middle or at the end of the list
    server_ckpt_list mid;
    add(mid, true, 5, 8192, 1, 3000);
    add(mid, true, 5, 8192, 1, OPENING, true);
    add(mid, true, 5, 8192, 2, 20000);
    pin = server_ckpt_keep_pin(mid, 21000);
    CHECK(pin != nullptr && pin->n_tokens == OPENING && mid.size() == 1);
    server_ckpt_list last;
    add(last, true, 5, 8192, 1, 3000);
    add(last, true, 5, 8192, 1, OPENING, true);
    pin = server_ckpt_keep_pin(last, OPENING);
    CHECK(pin != nullptr && pin->n_tokens == OPENING && last.size() == 1); // a seat holding just the opening

    // no pin: nothing, the list untouched
    server_ckpt_list none;
    add(none, true, 3, 8192, 1, OPENING);
    add(none, true, 3, 8192, 1, 20000);
    CHECK(server_ckpt_keep_pin(none, 21000) == nullptr && none.size() == 2);
    server_ckpt_list empty;
    CHECK(server_ckpt_keep_pin(empty, 21000) == nullptr);

    // a pin past the seat's tokens (never in a consistent seat): nothing, untouched
    server_ckpt_list past = ckpts;
    CHECK(server_ckpt_keep_pin(past, OPENING - 1) == nullptr && past.size() == 3);
}

static void test_opening_save_without_copies() {
    // a long chat stepped back to its pin (one copy left), and a seat still holding its pin and two rolling copies
    for (bool stepped_back : {true, false}) {
        for (int outcome : {0, 1, 2}) { // the save succeeds, fails (over the limit, already stored), or throws
            server_ckpt_list ckpts;
            add(ckpts, true, 3, 8192, 1, OPENING, true);
            add(ckpts, true, 3, 8192, 5, 18000);
            add(ckpts, true, 3, 8192, 5, 20000);
            const common_prompt_checkpoint * pin = stepped_back ? server_ckpt_keep_pin(ckpts, 21000) : &ckpts.front();
            const size_t n_before = ckpts.size();
            std::vector<const common_prompt_checkpoint *> before;
            for (const auto & cur : ckpts) {
                before.push_back(&cur);
            }

            server_ckpt_list entry; // what the store's alloc copies from the seat's prompt
            bool threw = false;
            bool got   = false;
            try {
                got = server_ckpt_save_without_copies(ckpts, [&]() {
                    entry = ckpts;
                    if (outcome == 2) {
                        throw std::runtime_error("bad_alloc in clone");
                    }
                    return outcome == 0;
                });
            } catch (const std::runtime_error &) {
                threw = true;
            }
            CHECK(threw == (outcome == 2));
            CHECK(got == (outcome == 0));
            CHECK(entry.empty()); // the entry carries no copies
            // the seat keeps every copy, in order, at the same addresses (the step back's pin pointer stays good)
            CHECK(ckpts.size() == n_before);
            size_t i = 0;
            for (const auto & cur : ckpts) {
                CHECK(i < before.size() && &cur == before[i]);
                i++;
            }
            CHECK(n_pinned(ckpts) == 1 && find_at(ckpts, OPENING) == pin && pin->pinned);
        }
    }

    // a seat with no copies: the save still runs, and nothing appears
    server_ckpt_list none;
    int n_calls = 0;
    CHECK(server_ckpt_save_without_copies(none, [&]() { n_calls++; return true; }));
    CHECK(n_calls == 1 && none.empty());
}

static void test_opening_first_pin() {
    const auto a = first_turn(800);
    const server_ckpt_list empty;

    // chat A's first turn on an empty seat: its copy at the first user message's start takes the pin
    CHECK( server_ckpt_pin_first_user(true, true, 3, true, empty, a, OPENING));
    // not at any other copy: a tail copy, the system prompt, the start
    CHECK(!server_ckpt_pin_first_user(true, true, 3, true, empty, a, OPENING + 1));
    CHECK(!server_ckpt_pin_first_user(true, true, 3, true, empty, a, OPENING + 800 - 4));
    CHECK(!server_ckpt_pin_first_user(true, true, 3, true, empty, a, 100));
    CHECK(!server_ckpt_pin_first_user(true, true, 3, true, empty, a, 0));
    // a follow-up's second user message is not the first one
    const auto fu = follow_up(4900, 9000, 9500);
    CHECK(!server_ckpt_pin_first_user(true, true, 3, true, empty, fu, 9000));
    CHECK( server_ckpt_pin_first_user(true, true, 3, true, empty, fu, OPENING));

    // only with the switch, the pin, room for it and a chat task
    CHECK(!server_ckpt_pin_first_user(false, true,  3, true,  empty, a, OPENING));
    CHECK(!server_ckpt_pin_first_user(true,  false, 3, true,  empty, a, OPENING));
    CHECK(!server_ckpt_pin_first_user(true,  true,  1, true,  empty, a, OPENING));
    CHECK(!server_ckpt_pin_first_user(true,  true,  3, false, empty, a, OPENING));

    // only while no copy holds the pin: the split rule moves an existing one
    server_ckpt_list pinned;
    add(pinned, true, 3, 8192, 1, 3000, true);
    CHECK(!server_ckpt_pin_first_user(true, true, 3, true, pinned, a, OPENING));
    server_ckpt_list rolling;
    add(rolling, true, 3, 8192, 1, 3000);
    CHECK( server_ckpt_pin_first_user(true, true, 3, true, rolling, a, OPENING));

    // no user message (a raw completion), or one at 0 (a copy at 0 holds nothing)
    CHECK(!server_ckpt_pin_first_user(true, true, 3, true, empty, common_chat_msg_spans{}, OPENING));
    const auto user0 = spans_of({{COMMON_CHAT_ROLE_USER, 0}, {COMMON_CHAT_ROLE_ASSISTANT, 500}}, 510);
    CHECK(!server_ckpt_pin_first_user(true, true, 3, true, empty, user0, 0));
}

// One long first chat walked through its copies the way the read-in saves them: the first turn's copies at the first
// user message's start and its tail offsets, each follow-up's at its last user message and its tail offsets, each
// copy offered rule 3 first (as server-context.cpp's create_checkpoint call does). Returns the list after the chat.
static server_ckpt_list long_first_chat(bool opening_on, bool chat, int n_max, int min_step, int n_follow,
        std::mt19937 & rng) {
    server_ckpt_list ckpts;
    auto copy = [&](int id_task, const common_chat_msg_spans & spans, int64_t at) {
        const bool pin = server_ckpt_pin_first_user(opening_on, true, n_max, chat, ckpts, spans, at);
        add(ckpts, true, n_max, min_step, id_task, at, pin);
    };
    const int64_t n_user = 300 + (int64_t) (rng() % 1500);
    const auto first = first_turn(n_user);
    const int64_t n_first = OPENING + n_user + 10;
    copy(1, first, OPENING);
    copy(1, first, n_first - 4);

    int64_t end = n_first + 200 + (int64_t) (rng() % 3000); // the first reply
    for (int task = 2; task < 2 + n_follow; task++) {
        const int64_t user_at = end;
        const int64_t n       = user_at + 50 + (int64_t) (rng() % 6000);
        const auto spans = follow_up(OPENING + n_user, user_at, n);
        copy(task, spans, user_at);
        if (n - 4100 > user_at) {
            copy(task, spans, n - 4100);
        }
        copy(task, spans, n - 4);
        end = n + 100 + (int64_t) (rng() % 4000);
    }
    return ckpts;
}

static void test_opening_first_chat_generated() {
    std::mt19937 rng(4211);
    int n_long          = 0; // chats with at least -ctxcp follow-ups
    int n_rolled_on_off = 0; // ... whose opening copy rolled out with the switch off
    for (int i = 0; i < 2000; i++) {
        const int n_max    = 2 + (int) (rng() % 4);           // -ctxcp 2..5 (the everyday library runs 3)
        const int min_step = (rng() % 2) ? 8192 : 256;
        const int n_follow = 1 + (int) (rng() % 25);          // up to ~150K tokens of chat

        // the switch on: one pin, at the opening, however many copies rolled past; the step back finds it
        std::mt19937 rng_on = rng;
        auto on = long_first_chat(true, true, n_max, min_step, n_follow, rng_on);
        CHECK(on.size() <= (size_t) n_max);
        CHECK(n_pinned(on) == 1);
        const auto * at = find_at(on, OPENING);
        CHECK(at != nullptr && at->pinned);
        auto * pin = server_ckpt_keep_pin(on, 1000000);
        CHECK(pin != nullptr && pin->n_tokens == OPENING && on.size() == 1);

        // the switch off (the stock pin needs two chats): no pin, so the step back would find nothing; and once
        // copies have rolled, the opening's copy is nearly always gone (upstream keeps the oldest of copies closer
        // than min-step, so a run of short follow-ups can leave it)
        std::mt19937 rng_off = rng;
        auto off = long_first_chat(false, true, n_max, min_step, n_follow, rng_off);
        CHECK(n_pinned(off) == 0);
        CHECK(server_ckpt_keep_pin(off, 1000000) == nullptr);
        if (n_follow >= n_max) {
            n_rolled_on_off += find_at(off, OPENING) == nullptr ? 1 : 0;
            n_long++;
        }

        // a code conversation never takes the first pin (it is never saved, and keeps all its rolling copies)
        std::mt19937 rng_code = rng;
        auto code = long_first_chat(true, false, n_max, min_step, n_follow, rng_code);
        CHECK(n_pinned(code) == 0);

        rng.discard(1);
    }
    CHECK(n_long > 1000 && n_rolled_on_off * 10 > n_long * 9);
    std::printf("opening copy: %d of %d long first chats lose the opening's copy without rule 3\n", n_rolled_on_off, n_long);
}

void test_opening() {
    test_opening_hand_on();
    test_opening_hand_on_generated();
    test_opening_keep_pin();
    test_opening_save_without_copies();
    test_opening_first_pin();
    test_opening_first_chat_generated();
}

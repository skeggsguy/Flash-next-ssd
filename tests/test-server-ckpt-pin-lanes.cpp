// The bookmark pin learns lanes and the opening copy refuses a pin inside the system prompt (study PLAN #31, decisions
// 2026-10-14), no model. Part of test-server-ckpt-pin (main in test-server-ckpt-pin.cpp).
//
// The day it fixes (dogfood 2026-10-14, after a restart, coding had the seat first): chat A's prompt parts from code's
// at token 3 (the template's effort line), the split rule pinned at 3, inside the system prompt, and that pin blocked
// the first user message's pin. When coding took the seat the store saved "3 of 5864 tokens", and chats B and C after
// code read in 5-7K tokens each instead of reading on from the opening (~5,050 tokens).
//
// - part 1, lanes: the split rule pins only when the seat's previous task was in this task's lane; a chat after code
//   takes off any pin code left on the seat (a pin at 3 even survives the read-in's invalidation), so the first user
//   message takes it; code after code keeps its split pin; code after chat leaves chat's pin alone; within a lane
//   nothing changes (an effort change between two chats still pins at its split); over generated inputs
// - part 2, the floor: the step back refuses a pin below the first user message's start, leaving the list untouched,
//   and saves a pin at or past it as before; over generated lists
// - the day walked through (code, code, chat A, code, chat B, code, chat C) with the real rules: the old rules save 3
//   and B and C read on from 3; the new ones save the opening and B and C read on from it
//
// play() walks one seat through a request the way server-context.cpp does (the hand-on, the read-in's pin target,
// restore, invalidation, the pin check and the copies), modelling only the store's pick and the read-in's batch
// breaks (a copy at the first user message's start, the pin target and 4 before the end).

#include "test-server-ckpt-pin.h"

#include <algorithm>
#include <random>

namespace {

constexpr int64_t CHAT_OPENING = 5050;  // chat's first user message's start (sidekick, 2026-10-14)
constexpr int64_t CODE_OPENING = 10589; // code's first user message's start (OpenCode, 2026-10-14)
constexpr int64_t CHAT_CODE    = 3;     // chat's and code's prompts part at token 3 (the effort line)
constexpr int64_t CHAT_CHAT    = 5077;  // two new chats part inside the first user message (the memory block)
constexpr int64_t CODE_CODE    = 10592; // two code requests part inside the first user message

// a first turn: system 0, the first user message at `opening`, the generation prompt's assistant opening
common_chat_msg_spans turn(int64_t opening, int64_t n) {
    return spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, (size_t) opening},
                     {COMMON_CHAT_ROLE_ASSISTANT, (size_t) (n - 10)}}, (size_t) n);
}

std::vector<std::pair<int64_t, bool>> shape(const server_ckpt_list & ckpts) {
    std::vector<std::pair<int64_t, bool>> out;
    for (const auto & cur : ckpts) {
        out.push_back({cur.n_tokens, cur.pinned});
    }
    return out;
}

} // namespace

static void test_lanes_rule() {
    const auto chat_a = turn(CHAT_OPENING, 5864);
    const auto code   = turn(CODE_OPENING, 68453);

    // 10:45: chat A on a code seat (37,612 tokens, code's pin at 10,589): the old rule pinned at 3
    CHECK(server_ckpt_pin_target(true, 3, chat_a, CHAT_CODE, 37612, 5864) == CHAT_CODE);
    server_ckpt_list seat;
    add(seat, true, 3, 8192, 1, CODE_OPENING, true);
    add(seat, true, 3, 8192, 1, 37608);
    auto r = server_ckpt_pin_target_lanes(true, 3, true, false, seat, chat_a, CHAT_CODE, 37612, 5864);
    CHECK(r.target == -1 && r.skipped == CHAT_CODE && r.unpinned == CODE_OPENING);
    CHECK(n_pinned(seat) == 0 && seat.size() == 2); // taken off, not thrown out

    // a code pin at 3 left on the seat (the old engine's leftover) survives the read-in's invalidation, since its
    // words are all in the chat's prompt: taken off, the first user message's copy takes the pin
    server_ckpt_list at3;
    add(at3, true, 3, 8192, 1, CHAT_CODE, true);
    add(at3, true, 3, 8192, 1, 37608);
    CHECK(!server_ckpt_pin_first_user(true, true, 3, true, at3, chat_a, CHAT_OPENING)); // blocked, as on the day
    r = server_ckpt_pin_target_lanes(true, 3, true, false, at3, chat_a, CHAT_CODE, 37612, 5864);
    CHECK(r.target == -1 && r.unpinned == CHAT_CODE);
    server_ckpt_invalidate(at3, (llama_pos) CHAT_CODE, no_log);
    CHECK(at3.size() == 1 && find_at(at3, CHAT_CODE) != nullptr && !find_at(at3, CHAT_CODE)->pinned);
    CHECK(server_ckpt_pin_first_user(true, true, 3, true, at3, chat_a, CHAT_OPENING));
    add(at3, true, 3, 8192, 2, CHAT_OPENING, true);
    CHECK(server_ckpt_pinned_at(at3) == CHAT_OPENING && n_pinned(at3) == 1);
    CHECK(server_ckpt_keep_pin(at3, 5864, CHAT_OPENING) != nullptr && at3.front().n_tokens == CHAT_OPENING);

    // code after code keeps its split pin (82.11: split at 10,592, pinned at 10,589), and at 9 (20.04)
    server_ckpt_list code_seat;
    add(code_seat, true, 3, 8192, 1, 9, true);
    r = server_ckpt_pin_target_lanes(true, 3, true, true, code_seat, code, CODE_CODE, 167632, 68453);
    CHECK(r.target == CODE_OPENING && r.skipped == -1 && r.unpinned == -1 && server_ckpt_pinned_at(code_seat) == 9);
    r = server_ckpt_pin_target_lanes(true, 3, true, true, code_seat, code, 9, 171300, 68453);
    CHECK(r.target == 9 && r.skipped == -1 && r.unpinned == -1);

    // code after chat (128.23: split at 3, pinned at 3): no pin, and chat's pin stays where it is
    server_ckpt_list chat_seat;
    add(chat_seat, true, 3, 8192, 1, CHAT_OPENING, true);
    add(chat_seat, true, 3, 8192, 1, 5660);
    r = server_ckpt_pin_target_lanes(true, 3, false, true, chat_seat, code, CHAT_CODE, 5671, 68453);
    CHECK(r.target == -1 && r.skipped == CHAT_CODE && r.unpinned == -1 && server_ckpt_pinned_at(chat_seat) == CHAT_OPENING);

    // within chat nothing changes: a new chat pins at its first user message, an effort change at its split
    // (2026-09-28), a follow-up nowhere
    r = server_ckpt_pin_target_lanes(true, 3, false, false, chat_seat, chat_a, CHAT_CHAT, 5671, 5864);
    CHECK(r.target == CHAT_OPENING && r.skipped == -1 && r.unpinned == -1);
    r = server_ckpt_pin_target_lanes(true, 3, false, false, chat_seat, chat_a, CHAT_CODE, 5671, 5864);
    CHECK(r.target == CHAT_CODE && r.skipped == -1);
    const auto follow_up = spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, CHAT_OPENING},
            {COMMON_CHAT_ROLE_ASSISTANT, 5854}, {COMMON_CHAT_ROLE_USER, 6400}, {COMMON_CHAT_ROLE_ASSISTANT, 6500}}, 6510);
    r = server_ckpt_pin_target_lanes(true, 3, false, false, chat_seat, follow_up, 6000, 6300, 6510);
    CHECK(r.target == -1 && r.skipped == -1 && r.unpinned == -1);
    CHECK(server_ckpt_pinned_at(chat_seat) == CHAT_OPENING);

    // LLAMA_CKPT_PIN=0: nothing pins and nothing is taken off
    server_ckpt_list off;
    add(off, true, 3, 8192, 1, CHAT_CODE, true);
    r = server_ckpt_pin_target_lanes(false, 3, true, false, off, chat_a, CHAT_CODE, 37612, 5864);
    CHECK(r.target == -1 && r.skipped == -1 && r.unpinned == -1 && n_pinned(off) == 1);
}

// generated: within a lane the split rule's pick, untouched; across lanes no pick, and only a chat after code takes
// the seat's pin off (never throwing a copy out)
static void test_lanes_generated() {
    std::mt19937 rng(20261014);
    int n_cross = 0;
    int n_taken = 0;
    for (int i = 0; i < 20000; i++) {
        const bool pin_on    = rng() % 8 != 0;
        const int  n_max     = 1 + (int) (rng() % 5);
        const bool prev_code = rng() % 2;
        const bool task_code = rng() % 2;
        const int64_t opening = 1 + (int64_t) (rng() % 12000);
        const int64_t n_new   = opening + 20 + (int64_t) (rng() % 4000);
        const int64_t n_old   = (int64_t) (rng() % 40000);
        const int64_t n_past  = (int64_t) (rng() % (std::min(n_old, n_new) + 1));
        const auto spans = turn(opening, n_new);

        server_ckpt_list ckpts;
        const int n_copies = (int) (rng() % 4);
        const int pinned   = (int) (rng() % 5) - 1; // which copy holds the pin, or none
        for (int k = 0; k < n_copies; k++) {
            add(ckpts, true, 8, 0, 1, 1 + (int64_t) (rng() % 40000), k == pinned);
        }
        const auto before = shape(ckpts);
        const int64_t pin_before = server_ckpt_pinned_at(ckpts);

        const auto r = server_ckpt_pin_target_lanes(pin_on, n_max, prev_code, task_code, ckpts, spans, n_past, n_old, n_new);
        const int64_t split = server_ckpt_pin_target(pin_on, n_max, spans, n_past, n_old, n_new);
        CHECK(ckpts.size() == before.size());
        if (prev_code == task_code) {
            CHECK(r.target == split && r.skipped == -1 && r.unpinned == -1);
            CHECK(shape(ckpts) == before);
            continue;
        }
        n_cross++;
        CHECK(r.target == -1 && r.skipped == split);
        if (pin_on && prev_code) {
            CHECK(r.unpinned == pin_before && n_pinned(ckpts) == 0);
            n_taken += pin_before >= 0;
        } else {
            CHECK(r.unpinned == -1 && shape(ckpts) == before); // code after chat: chat's pin stays
        }
    }
    CHECK(n_cross > 5000 && n_taken > 500);
}

static void test_floor() {
    const auto chat_a = turn(CHAT_OPENING, 5864);
    CHECK(server_ckpt_first_user_start(chat_a) == CHAT_OPENING);
    CHECK(server_ckpt_first_user_start(common_chat_msg_spans{}) == -1);
    CHECK(server_ckpt_first_user_start(spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}}, 100)) == -1);
    CHECK(server_ckpt_first_user_start(spans_of({{COMMON_CHAT_ROLE_USER, 0}, {COMMON_CHAT_ROLE_ASSISTANT, 50}}, 60)) == 0);

    // 92.42 (saved 3 of 5864): a pin at 3 below the floor is refused and the list stays as it was, at the same
    // addresses (no step back of the seat)
    server_ckpt_list seat;
    add(seat, true, 3, 8192, 1, CHAT_CODE, true);
    add(seat, true, 3, 8192, 1, CHAT_OPENING);
    add(seat, true, 3, 8192, 1, 5860);
    const auto before = shape(seat);
    const common_prompt_checkpoint * first = &seat.front();
    CHECK(server_ckpt_keep_pin(seat, 5864, CHAT_OPENING) == nullptr);
    CHECK(shape(seat) == before && &seat.front() == first);
    // without a floor (unknown, or a prompt with no user message), as before: saved
    server_ckpt_list stock = seat;
    CHECK(server_ckpt_keep_pin(stock, 5864, 0) != nullptr && stock.size() == 1 && stock.front().n_tokens == CHAT_CODE);
    server_ckpt_list stock_default = seat;
    CHECK(server_ckpt_keep_pin(stock_default, 5864) != nullptr && stock_default.front().n_tokens == CHAT_CODE);

    // a pin at the floor (the first user message's start), or past it (a split pin inside the message), saves
    for (int64_t at : {CHAT_OPENING, CHAT_OPENING + 27}) {
        server_ckpt_list ok;
        add(ok, true, 3, 8192, 1, 1000);
        add(ok, true, 3, 8192, 1, at, true);
        add(ok, true, 3, 8192, 1, 5860);
        auto * pin = server_ckpt_keep_pin(ok, 5864, CHAT_OPENING);
        CHECK(pin != nullptr && pin->n_tokens == at && pin->pinned && ok.size() == 1);
    }
    // one token under the floor is inside the system prompt
    server_ckpt_list under;
    add(under, true, 3, 8192, 1, CHAT_OPENING - 1, true);
    CHECK(server_ckpt_keep_pin(under, 5864, CHAT_OPENING) == nullptr && n_pinned(under) == 1);
}

// generated: the step back keeps the pin exactly when it lies in the seat and at or past the floor; otherwise the list
// is untouched
static void test_floor_generated() {
    std::mt19937 rng(5050);
    int n_refused_floor = 0;
    int n_kept          = 0;
    for (int i = 0; i < 20000; i++) {
        server_ckpt_list ckpts;
        const int n_copies = (int) (rng() % 4);
        const int pinned   = (int) (rng() % 5) - 1;
        for (int k = 0; k < n_copies; k++) {
            add(ckpts, true, 8, 0, 1, 1 + (int64_t) (rng() % 12000), k == pinned);
        }
        const int64_t seat  = (int64_t) (rng() % 13000);
        const int64_t floor = rng() % 4 == 0 ? 0 : (int64_t) (rng() % 12000);
        const auto before = shape(ckpts);
        const int64_t at  = server_ckpt_pinned_at(ckpts);

        const auto * pin = server_ckpt_keep_pin(ckpts, seat, floor);
        const bool keep = at > 0 && at <= seat && at >= floor;
        CHECK((pin != nullptr) == keep);
        if (keep) {
            CHECK(ckpts.size() == 1 && pin == &ckpts.front() && pin->n_tokens == at && pin->pinned);
            n_kept++;
        } else {
            CHECK(shape(ckpts) == before);
            n_refused_floor += at > 0 && at <= seat && at < floor;
        }
    }
    CHECK(n_kept > 1000 && n_refused_floor > 1000);
}

namespace {

struct rules {
    bool lanes; // part 1
    bool floor; // part 2
};

struct seat_model {
    server_ckpt_list ckpts;
    int64_t n = 0;        // tokens on the seat
    bool code = false;    // whose tokens they are
    bool last_code = false, prev_code = false;
    int64_t floor = 0;    // the seat's opening_floor
    int task = 0;
};

struct store_model {
    bool has = false;
    int64_t n = 0; // the saved chat opening's tokens
};

// how much two prompts share: within chat the opening and a little memory, within code its opening, across lanes 3
int64_t shares(bool a_code, bool b_code, int64_t n_a) {
    return std::min(n_a, a_code != b_code ? CHAT_CODE : a_code ? CODE_CODE : CHAT_CHAT);
}

// one request on the seat; returns where its read-in starts (the bookmark). saves gets each save's n_tokens.
int64_t play(seat_model & s, store_model & st, std::vector<int64_t> & saves, const rules & ru, bool code, int64_t n) {
    // the hand-on (get_available_slot): saved when coding takes a chat seat, taken when a chat takes a code seat
    const auto hand = server_ckpt_opening_hand_on(true, !s.last_code, !code, s.n);
    if (hand.save) {
        const auto * pin = server_ckpt_keep_pin(s.ckpts, s.n, ru.floor ? s.floor : 0);
        if (pin != nullptr) {
            s.n = pin->n_tokens; // stepped back
            saves.push_back(s.n);
            if (!(st.has && s.n <= st.n)) { // the store refuses a prompt it already holds (a prefix of its entry)
                st = {true, s.n};
            }
        }
    }
    // the store hands an entry back when the new chat shares a quarter of it and more than the seat does
    if (hand.load && st.has && shares(false, code, st.n) * 4 >= st.n && shares(false, code, st.n) > shares(s.code, code, s.n)) {
        s.ckpts.clear(); // the entry carries no copies
        s.n = st.n;
        s.code = false;
    }

    // the read-in (launch_slot_with_task, then the prompt's pin target, restore, invalidation and pin check)
    s.task++;
    s.prev_code = s.last_code;
    s.last_code = code;
    const auto spans = turn(code ? CODE_OPENING : CHAT_OPENING, n);
    const int64_t split = shares(s.code, code, s.n);
    int64_t target = ru.lanes
        ? server_ckpt_pin_target_lanes(true, 3, s.prev_code, code, s.ckpts, spans, split, s.n, n).target
        : server_ckpt_pin_target(true, 3, spans, split, s.n, n);
    int64_t n_past = split;
    if (split < s.n) {
        const auto * it = server_ckpt_restore_pick(s.ckpts, (llama_pos) split, (llama_pos) split);
        n_past = it ? it->n_tokens : 0;
    }
    server_ckpt_invalidate(s.ckpts, (llama_pos) n_past, no_log);
    int64_t was = -1;
    if (target >= 0 && server_ckpt_pin_existing(s.ckpts, target, &was)) {
        s.floor = std::max<int64_t>(0, server_ckpt_first_user_start(spans));
        target = -1;
    } else if (n_past > target) {
        target = -1;
    }
    std::vector<int64_t> at = {target, server_ckpt_first_user_start(spans), n - 4};
    std::sort(at.begin(), at.end());
    at.erase(std::unique(at.begin(), at.end()), at.end());
    for (int64_t a : at) {
        if (a <= 0 || a < n_past) {
            continue;
        }
        const bool pin = a == target || server_ckpt_pin_first_user(true, true, 3, !code, s.ckpts, spans, a);
        add(s.ckpts, true, 3, 8192, s.task, a, pin);
        if (pin) {
            s.floor = std::max<int64_t>(0, server_ckpt_first_user_start(spans));
        }
    }
    s.n = n + 500; // and its answer
    s.code = code;
    return n_past;
}

struct day {
    std::vector<int64_t> saves;
    int64_t a = -1, b = -1, c = -1;     // where chats A, B and C read on from
    int64_t code2_pin = -1;              // the pin after code after code
    int64_t after_code3_pin = -2;        // the pin after code after chat A
};

// 2026-10-14 after the restart: code, code, chat A, code, chat B, code, chat C (scenario 7, with a second code first)
day play_day(const rules & ru) {
    seat_model s;
    store_model st;
    day d;
    play(s, st, d.saves, ru, true, 37612);
    play(s, st, d.saves, ru, true, 40010);
    d.code2_pin = server_ckpt_pinned_at(s.ckpts);
    d.a = play(s, st, d.saves, ru, false, 5864);
    play(s, st, d.saves, ru, true, 68453);
    d.after_code3_pin = server_ckpt_pinned_at(s.ckpts);
    d.b = play(s, st, d.saves, ru, false, 9373);
    play(s, st, d.saves, ru, true, 14685);
    d.c = play(s, st, d.saves, ru, false, 5671);
    return d;
}

} // namespace

static void test_lanes_day() {
    // the old rules: the store saves 3 tokens and B and C read on from 3, reading in their whole opening
    const day old_day = play_day({false, false});
    CHECK(!old_day.saves.empty() && old_day.saves.front() == CHAT_CODE);
    CHECK(old_day.b == CHAT_CODE && old_day.c == CHAT_CODE);
    CHECK(old_day.after_code3_pin == CHAT_CODE); // the seat stepped back to 3 keeps that pin through code

    // the new rules: one full read-in after the opening (A, the known limit), then every new chat reads on from the
    // saved opening, and every save is the opening
    const day fixed = play_day({true, true});
    CHECK(fixed.code2_pin == CODE_OPENING);      // code after code keeps its split pin
    CHECK(fixed.a == 0);
    CHECK(fixed.b == CHAT_OPENING && fixed.c == CHAT_OPENING);
    CHECK(fixed.saves.size() == 2);
    for (int64_t n : fixed.saves) {
        CHECK(n == CHAT_OPENING);
    }
    CHECK(fixed.after_code3_pin == -1);          // code after chat pins nothing (chat's pin went with its words)

    // the floor alone stops the junk save, but chats after code still read in everything: lanes are the fix
    const day floor_only = play_day({false, true});
    CHECK(floor_only.saves.empty());
    CHECK(floor_only.b == CHAT_CODE && floor_only.c == CHAT_CODE);
    // lanes alone fix this day; the floor guards a same-lane pin inside the system prompt (test_floor)
    const day lanes_only = play_day({true, false});
    CHECK(lanes_only.b == CHAT_OPENING && lanes_only.saves == fixed.saves);
}

void test_lanes() {
    test_lanes_rule();
    test_lanes_generated();
    test_floor();
    test_floor_generated();
    test_lanes_day();
}

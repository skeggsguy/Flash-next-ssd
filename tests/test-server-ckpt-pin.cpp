// The bookmark pin's rules (tools/server/server-ckpt-pin.h, study BOOKMARK-PIN-PLAN.md step 1), without a model.
//
// - the split rule: a new chat's split (inside the system prompt or the first user message) counts; a follow-up's
//   or a tool turn's (at or after an assistant message) and a retry's (no split inside the new prompt) don't
// - in test-server-ckpt-pin-chats.cpp: the pin target (the start of the message holding the split, the amendment),
//   upstream's restore choice, and cleric's hand-back played out as chats
// - the pin survives more -ctxcp new copies than the list holds, moves to each new split, and an old pin becomes
//   an ordinary rolling copy; a copy superseding the pin inherits it
// - a pin beyond the split is dropped by the invalidation (its words are gone)
// - LLAMA_CKPT_PIN=0 (and on, with no pin asked for) gives upstream's picks: compared with upstream's
//   create_checkpoint list logic, copied below as the reference, over generated sequences
// - with the pin on, invariants over generated sequences: never more than -ctxcp copies, at most one pinned,
//   and the pin is never thrown out to make room

#include "test-server-ckpt-pin.h"

#include <cinttypes>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static void test_parse() {
    CHECK(server_ckpt_pin_parse(nullptr));
    CHECK(server_ckpt_pin_parse("1"));
    CHECK(!server_ckpt_pin_parse("0"));
}

static void test_split_rule() {
    // chat B's first turn: system 0..12890, user (memory + question) 12890..14790, generation prompt at 14790
    const auto first_turn = spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, 12890},
                                      {COMMON_CHAT_ROLE_ASSISTANT, 14790}}, 14800);
    // a new chat splits right after the first user message's opening marker
    CHECK( server_ckpt_split_is_new_chat(first_turn, 12893, 21000, 14800));
    CHECK(server_ckpt_pin_target(true, 5, first_turn, 12893, 21000, 14800) == 12890); // the user message's start
    // an effort or tool-list change splits inside the system prompt: a new chat too (the pin moves there)
    CHECK( server_ckpt_split_is_new_chat(first_turn, 40, 21000, 14800));

    // a follow-up in the same chat re-renders the last reply: it splits inside the first assistant message
    const auto follow_up = spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, 12890},
                                     {COMMON_CHAT_ROLE_ASSISTANT, 14790}, {COMMON_CHAT_ROLE_USER, 15600},
                                     {COMMON_CHAT_ROLE_ASSISTANT, 15700}}, 15710);
    CHECK(!server_ckpt_split_is_new_chat(follow_up, 15400, 15650, 15710));
    CHECK(!server_ckpt_split_is_new_chat(follow_up, 14790, 15650, 15710)); // exactly at the reply's start
    CHECK(server_ckpt_pin_target(true, 5, follow_up, 15400, 15650, 15710) == -1);
    // the same chat edited in its first message still splits before any reply
    CHECK( server_ckpt_split_is_new_chat(follow_up, 13000, 15650, 15710));

    // a tool turn splits at or after the assistant message that called the tool
    const auto tool_turn = spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, 12890},
                                     {COMMON_CHAT_ROLE_ASSISTANT, 14790}, {COMMON_CHAT_ROLE_TOOL, 14900},
                                     {COMMON_CHAT_ROLE_ASSISTANT, 16000}}, 16010);
    CHECK(!server_ckpt_split_is_new_chat(tool_turn, 14899, 14950, 16010));

    // a retry of the same prompt: the whole new prompt is the old one's prefix (the old one ends in its answer)
    CHECK(!server_ckpt_split_is_new_chat(first_turn, 14800, 15300, 14800));
    // a prompt that extends the old one entirely: no split inside the old prompt
    CHECK(!server_ckpt_split_is_new_chat(first_turn, 12000, 12000, 14800));
    // nothing shared, or nothing in the slot yet
    CHECK(!server_ckpt_split_is_new_chat(first_turn, 0, 21000, 14800));
    CHECK(!server_ckpt_split_is_new_chat(first_turn, 0, 0, 14800));

    // no delimiters (a raw completion prompt): no reply to re-render, any split inside both counts
    CHECK( server_ckpt_split_is_new_chat(common_chat_msg_spans{}, 100, 200, 300));
    // but a retry is still no split (without an assistant span only the lengths can tell)
    CHECK(!server_ckpt_split_is_new_chat(common_chat_msg_spans{}, 300, 400, 300));
    CHECK(!server_ckpt_split_is_new_chat(common_chat_msg_spans{}, 200, 200, 300));

    // the switch, and a list too short to hold a pin beside a rolling copy
    CHECK(server_ckpt_pin_target(false, 5, first_turn, 12893, 21000, 14800) == -1);
    CHECK(server_ckpt_pin_target(true,  1, first_turn, 12893, 21000, 14800) == -1);
    CHECK(server_ckpt_pin_target(true,  2, first_turn, 12893, 21000, 14800) == 12890);
}

static void upstream_add(server_ckpt_list & ckpts, int n_max, int min_step, int id_task, int64_t n_tokens_new,
        std::vector<int64_t> & erased);
static std::vector<int64_t> positions(const server_ckpt_list & ckpts);

static void test_pin_survives() {
    for (int min_step : {256, 8192}) {
        server_ckpt_list ckpts;
        add(ckpts, true, 5, min_step, 1, 12893, true);
        // a long chat's follow-ups: each task saves copies at its last user message and its tail offsets
        int64_t pos = 12893;
        for (int task = 2; task < 40; task++) {
            pos += 300 + 37 * task;
            add(ckpts, true, 5, min_step, task, pos);
            add(ckpts, true, 5, min_step, task, pos + 2048);
            add(ckpts, true, 5, min_step, task, pos + 2560);
            pos += 2600;
            CHECK(ckpts.size() <= 5);
            CHECK(n_pinned(ckpts) == 1);
            CHECK(find_at(ckpts, 12893) != nullptr && find_at(ckpts, 12893)->pinned);
        }
        CHECK(ckpts.size() == 5);

        // the same with the switch off: the flag is ignored and the copy at the start rolls out (copies spaced
        // wider than min-step, as the dogfood's were: closer ones are pruned first and the oldest kept)
        server_ckpt_list off;
        add(off, false, 5, min_step, 1, 12893, true);
        CHECK(n_pinned(off) == 0);
        for (int task = 2; task < 12; task++) {
            add(off, false, 5, min_step, task, 13000 + 9000 * task);
        }
        CHECK(find_at(off, 12893) == nullptr);

        // a list that holds a pinned copy when the switch is off (none can get one, but the flag is ignored
        // wherever it came from): the copy rolls out exactly as upstream's would
        server_ckpt_list held = ckpts;
        server_ckpt_list ref  = ckpts;
        std::vector<int64_t> er_ref;
        for (int task = 50; task < 60; task++) {
            add(held, false, 5, min_step, task, pos + 9000 * (task - 49));
            upstream_add(ref, 5, min_step, task, pos + 9000 * (task - 49), er_ref);
            CHECK(positions(held) == positions(ref));
        }
        CHECK(find_at(held, 12893) == nullptr);
    }
}

static void test_pin_moves() {
    server_ckpt_list ckpts;
    add(ckpts, true, 3, 8192, 1, 12893, true);
    add(ckpts, true, 3, 8192, 1, 20000);
    // the next chat splits later: the copies past the split go, the pin moves, the old pin becomes a rolling copy
    server_ckpt_invalidate(ckpts, 13100, no_log);
    add(ckpts, true, 3, 8192, 2, 13100, true);
    CHECK(n_pinned(ckpts) == 1 && find_at(ckpts, 13100)->pinned && !find_at(ckpts, 12893)->pinned);
    // so it is the first thrown out when room is needed, and the pin stays
    add(ckpts, true, 3, 8192, 3, 30000);
    add(ckpts, true, 3, 8192, 4, 40000);
    CHECK(find_at(ckpts, 12893) == nullptr && find_at(ckpts, 13100) != nullptr && find_at(ckpts, 13100)->pinned);
    add(ckpts, true, 3, 8192, 5, 50000);
    CHECK(find_at(ckpts, 30000) == nullptr && find_at(ckpts, 13100) != nullptr && find_at(ckpts, 13100)->pinned);

    // a copy saved again at the pin's place supersedes it and keeps the pin
    add(ckpts, true, 3, 8192, 6, 13100);
    CHECK(n_pinned(ckpts) == 1 && find_at(ckpts, 13100)->pinned);

    // a copy already at the split takes the pin as it is
    int64_t was = -2;
    CHECK(server_ckpt_pin_existing(ckpts, 50000, &was) && was == 13100);
    CHECK(n_pinned(ckpts) == 1 && find_at(ckpts, 50000)->pinned);
    // no copy there: nothing changes
    CHECK(!server_ckpt_pin_existing(ckpts, 12345, &was));
    CHECK(n_pinned(ckpts) == 1 && find_at(ckpts, 50000)->pinned);
}

static void test_pin_beyond_split_dropped() {
    server_ckpt_list ckpts;
    add(ckpts, true, 5, 8192, 1, 5000);
    add(ckpts, true, 5, 8192, 1, 12893, true);
    add(ckpts, true, 5, 8192, 1, 20000);
    // the next chat splits before the pin (an effort change): the pin's words are gone
    server_ckpt_invalidate(ckpts, 10000, no_log);
    CHECK(ckpts.size() == 1 && find_at(ckpts, 5000) != nullptr && n_pinned(ckpts) == 0);

    // a split at the pin keeps it (its words, tokens 0..12892, are all still there)
    server_ckpt_list at;
    add(at, true, 5, 8192, 1, 12893, true);
    add(at, true, 5, 8192, 1, 20000);
    server_ckpt_invalidate(at, 12893, no_log);
    CHECK(at.size() == 1 && find_at(at, 12893)->pinned);
}

// upstream's create_checkpoint list logic (engine a2eebf440), the reference for the switch off
static void upstream_add(server_ckpt_list & ckpts, int n_max, int min_step, int id_task, int64_t n_tokens_new,
        std::vector<int64_t> & erased) {
    int64_t last = -1;
    for (auto it = ckpts.begin(); ckpts.size() + 1 >= (size_t) n_max && it != ckpts.end(); ) {
        if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + min_step) {
            erased.push_back(it->n_tokens);
            it = ckpts.erase(it);
            continue;
        }
        last = it->n_tokens;
        ++it;
    }
    while (ckpts.size() >= (size_t) n_max) {
        erased.push_back(ckpts.front().n_tokens);
        ckpts.erase(ckpts.begin());
    }
    for (auto it = ckpts.begin(); it != ckpts.end(); ) {
        if (it->n_tokens == n_tokens_new) {
            erased.push_back(it->n_tokens);
            it = ckpts.erase(it);
        } else {
            ++it;
        }
    }
    auto & cur = ckpts.emplace_back();
    cur.id_task = id_task;
    cur.update_pos(n_tokens_new, (llama_pos) n_tokens_new - 1, (llama_pos) n_tokens_new - 1);
}

static std::vector<int64_t> positions(const server_ckpt_list & ckpts) {
    std::vector<int64_t> out;
    for (const auto & cur : ckpts) {
        out.push_back(cur.n_tokens);
    }
    return out;
}

// a generated sequence of what a server does to one slot's list: chats read in (copies at rising positions),
// new prompts that split somewhere (the invalidation), repeats at the same place
struct op {
    int     kind;     // 0 = add, 1 = invalidate at pos
    int     id_task;
    int64_t pos;
    bool    pin;
};

static std::vector<op> gen_ops(std::mt19937 & rng, int n) {
    std::vector<op> ops;
    int64_t top = 0;
    int task = 1;
    for (int i = 0; i < n; i++) {
        const int r = (int) (rng() % 10);
        if (r < 7) {
            // a copy at or past the top; sometimes at the same place again
            const int64_t step = (rng() % 4 == 0) ? 0 : 1 + (int64_t) (rng() % 9000);
            top += step;
            ops.push_back({0, task, top, rng() % 5 == 0});
            if (rng() % 3 == 0) {
                task++;
            }
        } else {
            // a new prompt splits somewhere below the top
            const int64_t split = top > 0 ? (int64_t) (rng() % (uint64_t) (top + 1)) : 0;
            ops.push_back({1, task, split, false});
            top = split;
            task++;
        }
    }
    return ops;
}

static void test_upstream_picks() {
    std::mt19937 rng(20260928);
    for (int seq = 0; seq < 400; seq++) {
        const int n_max    = 1 + (int) (rng() % 8);
        const int min_step = (int) (rng() % 3) == 0 ? 0 : (int) (rng() % 10000);
        const auto ops     = gen_ops(rng, 60);

        server_ckpt_list ref, off, on_unasked;
        std::vector<int64_t> er_ref, er_off, er_on;
        const auto log_off = [&](const common_prompt_checkpoint & c, server_ckpt_erase_why) { er_off.push_back(c.n_tokens); };
        const auto log_on  = [&](const common_prompt_checkpoint & c, server_ckpt_erase_why) { er_on.push_back(c.n_tokens); };
        for (const auto & o : ops) {
            if (o.kind == 0) {
                upstream_add(ref, n_max, min_step, o.id_task, o.pos, er_ref);
                // switch off: the pin requests the server would make are ignored
                add(off, false, n_max, min_step, o.id_task, o.pos, o.pin, log_off);
                // switch on, but no new chat ever asked for a pin
                add(on_unasked, true, n_max, min_step, o.id_task, o.pos, false, log_on);
            } else {
                for (auto * l : {&ref, &off, &on_unasked}) {
                    server_ckpt_invalidate(*l, (llama_pos) o.pos, no_log);
                }
            }
            CHECK(positions(off) == positions(ref));
            CHECK(positions(on_unasked) == positions(ref));
            CHECK(n_pinned(off) == 0);
        }
        CHECK(er_off == er_ref);
        CHECK(er_on == er_ref);
    }
}

static void test_pin_invariants() {
    std::mt19937 rng(1187);
    for (int seq = 0; seq < 400; seq++) {
        const int n_max    = 2 + (int) (rng() % 7);
        const int min_step = (int) (rng() % 3) == 0 ? 0 : (int) (rng() % 10000);
        const auto ops     = gen_ops(rng, 80);

        server_ckpt_list ckpts;
        int64_t pin_at = -1; // where the pin should be, -1 = none
        bool pin_thrown_out = false;
        const auto log = [&](const common_prompt_checkpoint & c, server_ckpt_erase_why why) {
            if (c.pinned && (why == SERVER_CKPT_ERASE_TOO_CLOSE || why == SERVER_CKPT_ERASE_OLDEST)) {
                pin_thrown_out = true;
            }
        };
        for (const auto & o : ops) {
            if (o.kind == 0) {
                add(ckpts, true, n_max, min_step, o.id_task, o.pos, o.pin, log);
                CHECK(ckpts.size() <= (size_t) n_max);
                if (o.pin) {
                    pin_at = o.pos;
                }
            } else {
                server_ckpt_invalidate(ckpts, (llama_pos) o.pos, no_log);
                if (pin_at >= 0 && pin_at - 1 > o.pos) {
                    pin_at = -1; // its words are gone
                }
            }
            CHECK(n_pinned(ckpts) <= 1);
            CHECK(!pin_thrown_out);
            if (pin_at >= 0) {
                const auto * p = find_at(ckpts, pin_at);
                CHECK(p != nullptr && p->pinned);
            } else {
                CHECK(n_pinned(ckpts) == 0);
            }
        }
    }
}

int n_fail = 0;

int main() {
    test_parse();
    test_split_rule();
    test_pin_target();
    test_restore_pick();
    test_handback_chats();
    test_pin_survives();
    test_pin_moves();
    test_pin_beyond_split_dropped();
    test_upstream_picks();
    test_pin_invariants();

    if (n_fail > 0) {
        std::fprintf(stderr, "test-server-ckpt-pin: %d check(s) failed\n", n_fail);
        return 1;
    }
    std::printf("test-server-ckpt-pin: all checks passed\n");
    return 0;
}

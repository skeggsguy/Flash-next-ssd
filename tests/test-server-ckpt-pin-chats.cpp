// The bookmark pin's target and cleric's hand-back, played out as chats (BOOKMARK-PIN-PLAN.md amendment), no model.
// Part of test-server-ckpt-pin (main and the list rules in test-server-ckpt-pin.cpp).
//
// The hand-back: sidekick's <memory> block opens with fact counts and ~600 tokens of always-on facts, so
// consecutive chats share a varying part of it. Chats A and B share 900 tokens of memory; B's follow-ups roll every
// rolling copy; C shares only 20. A pin at B's exact split (the old rule) is past C's split and C read in from 0;
// pinned at the user message's start, C restores that copy and reads in only its memory and question.
//
// serve() below walks one slot through a request the way server-context.cpp does, with the real rules
// (server_ckpt_pin_target, _restore_pick, _invalidate, _pin_existing, _add). What it models rather than runs is
// the read-in loop's batch breaks: a copy at each batch start the loop saves, i.e. the last user message's start,
// the pin target and the two tail offsets (n_ubatch + 4 and 4 before the end), on hybrid memory (a copy at n
// holds positions up to n - 1; no window).

#include "test-server-ckpt-pin.h"

#include <algorithm>

void test_pin_target() {
    const auto first_turn = spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, 12890},
                                      {COMMON_CHAT_ROLE_ASSISTANT, 14790}}, 14800);
    // which span holds a position
    CHECK(server_ckpt_span_start(first_turn, 0)     == 0);
    CHECK(server_ckpt_span_start(first_turn, 12889) == 0);
    CHECK(server_ckpt_span_start(first_turn, 12890) == 12890);
    CHECK(server_ckpt_span_start(first_turn, 14799) == 14790);
    CHECK(server_ckpt_span_start(first_turn, 14800) == -1);

    // a split inside the first user message (anywhere in the memory block) pins at the message's start
    for (int64_t split : {12891, 12893, 13790, 14789}) {
        CHECK(server_ckpt_pin_target(true, 5, first_turn, split, 21000, 14800) == 12890);
    }
    // a split exactly at the user message's start: that is the start
    CHECK(server_ckpt_pin_target(true, 5, first_turn, 12890, 21000, 14800) == 12890);
    // a split inside the system message (an effort or tool-list change) pins at the split itself
    CHECK(server_ckpt_pin_target(true, 5, first_turn, 40, 21000, 14800) == 40);
    CHECK(server_ckpt_pin_target(true, 5, first_turn, 12889, 21000, 14800) == 12889);
    // also when the system message does not start at 0 (a BOS token before it): never its start
    const auto bos = spans_of({{COMMON_CHAT_ROLE_SYSTEM, 1}, {COMMON_CHAT_ROLE_USER, 12890},
                               {COMMON_CHAT_ROLE_ASSISTANT, 14790}}, 14800);
    CHECK(server_ckpt_pin_target(true, 5, bos, 40, 21000, 14800) == 40);
    CHECK(server_ckpt_pin_target(true, 5, bos, 12893, 21000, 14800) == 12890);

    // no spans (a raw prompt): the split
    CHECK(server_ckpt_pin_target(true, 5, common_chat_msg_spans{}, 100, 200, 300) == 100);
    // a split before every span (a prefix the delimiters do not cover): the split
    const auto late = spans_of({{COMMON_CHAT_ROLE_USER, 50}, {COMMON_CHAT_ROLE_ASSISTANT, 400}}, 410);
    CHECK(server_ckpt_pin_target(true, 5, late, 20, 500, 410) == 20);
    CHECK(server_ckpt_pin_target(true, 5, late, 60, 500, 410) == 50);
    // no system message, the user message at 0: a copy at 0 holds nothing, so the split
    const auto no_system = spans_of({{COMMON_CHAT_ROLE_USER, 0}, {COMMON_CHAT_ROLE_ASSISTANT, 400}}, 410);
    CHECK(server_ckpt_pin_target(true, 5, no_system, 60, 500, 410) == 60);
    // a follow-up still pins nothing, and the switch still rules
    const auto follow_up = spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, 12890},
                                     {COMMON_CHAT_ROLE_ASSISTANT, 14790}, {COMMON_CHAT_ROLE_USER, 15600},
                                     {COMMON_CHAT_ROLE_ASSISTANT, 15700}}, 15710);
    CHECK(server_ckpt_pin_target(true,  5, follow_up, 15400, 15650, 15710) == -1);
    CHECK(server_ckpt_pin_target(false, 5, first_turn, 13790, 21000, 14800) == -1);
}


void test_restore_pick() {
    server_ckpt_list ckpts;
    for (int64_t n : {5000, 12890, 20000}) {
        add(ckpts, true, 5, 8192, 1, n);
    }
    // the newest copy whose words are all in the new prompt (pos_max <= pos_next)
    CHECK(server_ckpt_restore_pick(ckpts, 12913, 12913)->n_tokens == 12890);
    CHECK(server_ckpt_restore_pick(ckpts, 12890, 12890)->n_tokens == 12890); // holds 0..12889: exactly fits
    CHECK(server_ckpt_restore_pick(ckpts, 12888, 12888)->n_tokens == 5000);
    CHECK(server_ckpt_restore_pick(ckpts, 30000, 30000)->n_tokens == 20000);
    CHECK(server_ckpt_restore_pick(ckpts, 4000, 4000) == nullptr);
    // and old enough for the memory's window: pos_min < pos_min_thold (or 0)
    CHECK(server_ckpt_restore_pick(ckpts, 30000, 19999)->n_tokens == 12890);
    // a copy whose words run past the split (pos_max > pos_next) is never restored, however old its start
    // (a windowed memory's copy spans [pos_min, pos_max])
    server_ckpt_list wide;
    server_ckpt_add(wide, true, 5, 8192, 1, 20000, 100, 19999, false, no_log);
    CHECK(server_ckpt_restore_pick(wide, 15000, 15000) == nullptr);
    CHECK(server_ckpt_restore_pick(wide, 20000, 20000) != nullptr);
    // a copy whose window starts at 0 (a plain attention memory, no recurrent part) fits however low the threshold
    // falls; one starting at 1 does not (upstream's `pos_min == 0` alternative, cleric's MD)
    server_ckpt_list plain;
    server_ckpt_add(plain, true, 5, 8192, 1, 100, 0, 99, false, no_log);
    CHECK(server_ckpt_restore_pick(plain, 100, 0) != nullptr);
    server_ckpt_list from_one;
    server_ckpt_add(from_one, true, 5, 8192, 1, 100, 1, 99, false, no_log);
    CHECK(server_ckpt_restore_pick(from_one, 100, 0) == nullptr);
    // looked at newest first
    std::vector<int64_t> seen;
    server_ckpt_restore_pick(ckpts, 12913, 12913, [&](const common_prompt_checkpoint & c) { seen.push_back(c.n_tokens); });
    CHECK((seen == std::vector<int64_t>{20000, 12890}));
}

namespace {

constexpr int64_t SYS = 12890; // the system block: effort line, tools, tool-call wording, sidekick's prompt
constexpr int64_t MEM = 1900;  // the <memory> block, after the user message's 3-token opening
constexpr int64_t Q   = 100;   // the question
constexpr int64_t GEN = 10;    // the generation prompt (<|im_start|>assistant ...)

// a chat's prompt after `follow_ups` earlier turns (reply 500, next message 4,000 each: tool results, as the
// dogfood day's long chat grew; copies closer together than min-step would keep the start copy anyway, upstream's
// prune keeping the oldest); sys_len lets a chat change its system block (an effort change)
struct prompt {
    common_chat_msg_spans spans;
    int64_t n = 0;
    int64_t last_user = 0;
};

prompt chat(int follow_ups, int64_t sys_len = SYS) {
    std::vector<std::pair<common_chat_role, size_t>> starts = {{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, sys_len}};
    int64_t pos = sys_len + 3 + MEM + Q;
    prompt p;
    p.last_user = sys_len;
    for (int i = 0; i < follow_ups; i++) {
        starts.push_back({COMMON_CHAT_ROLE_ASSISTANT, pos});
        pos += 500;
        starts.push_back({COMMON_CHAT_ROLE_USER, pos});
        p.last_user = pos;
        pos += 4000;
    }
    starts.push_back({COMMON_CHAT_ROLE_ASSISTANT, pos});
    p.n = pos + GEN;
    p.spans = spans_of(starts, p.n);
    return p;
}

struct slot_model {
    bool pin_on;
    server_ckpt_list ckpts;
    int64_t n_old = 0; // the tokens the context memory holds: the last prompt and its answer
    int task = 0;
    static constexpr int n_max = 5, min_step = 8192, ub = 4096;
};

struct outcome {
    int64_t target   = -1; // where this prompt pins, before the copy check
    int64_t restored = -1; // n_tokens of the copy restored, -1 = none
    int64_t n_past   = 0;  // where reading in starts
};

outcome serve(slot_model & s, const prompt & p, int64_t split) {
    outcome o;
    s.task++;
    o.target = server_ckpt_pin_target(s.pin_on, s.n_max, p.spans, split, s.n_old, p.n);
    int64_t target = o.target;

    llama_pos pos_next = (llama_pos) split;
    if (split > 0 && split < s.n_old) {
        // the recurrent part holds the old prompt's end, so a copy must be restored
        const auto * it = server_ckpt_restore_pick(s.ckpts, pos_next, pos_next);
        o.restored = it ? it->n_tokens : -1;
        o.n_past   = it ? std::min(split, it->n_tokens) : 0;
        pos_next   = (llama_pos) o.n_past;
    } else {
        o.n_past = split;
    }
    server_ckpt_invalidate(s.ckpts, pos_next, no_log);

    int64_t was = -1;
    if (target >= 0 && (server_ckpt_pin_existing(s.ckpts, target, &was) || o.n_past > target)) {
        target = -1;
    }

    std::vector<int64_t> starts = {p.last_user, target, p.n - 4 - s.ub, p.n - 4};
    std::sort(starts.begin(), starts.end());
    starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
    for (int64_t at : starts) {
        if (at > 0 && at >= o.n_past && at < p.n) {
            add(s.ckpts, s.pin_on, s.n_max, s.min_step, s.task, at, at == target);
        }
        CHECK(s.ckpts.size() <= (size_t) s.n_max);
    }
    s.n_old = p.n + 500; // and its answer
    return o;
}

// chat A, B, C of the hand-back; returns C's outcome
outcome play_handback(slot_model & s) {
    serve(s, chat(0), 0); // A, the first chat after boot
    for (int k = 1; k <= 4; k++) {
        serve(s, chat(k), chat(k - 1).n + 5); // follow-ups split inside A's last reply
    }
    const auto b = serve(s, chat(0), SYS + 3 + 900); // B shares 900 tokens of memory with A
    CHECK(b.target == (s.pin_on ? SYS : -1));
    for (int k = 1; k <= 4; k++) {
        serve(s, chat(k), chat(k - 1).n + 5);
    }
    return serve(s, chat(0), SYS + 3 + 20); // C shares only 20 with B
}

} // namespace

void test_handback_chats() {
    // with the pin: C restores the copy at the user message's start and reads in only its memory and question
    slot_model on{true, {}};
    const auto c = play_handback(on);
    CHECK(c.target == SYS);
    CHECK(c.restored == SYS);
    CHECK(c.n_past == SYS);
    CHECK(find_at(on.ckpts, SYS) != nullptr && find_at(on.ckpts, SYS)->pinned && n_pinned(on.ckpts) == 1);

    // without it (upstream): B's follow-ups rolled the copy at the start out, so C reads in everything
    slot_model off{false, {}};
    const auto c_off = play_handback(off);
    CHECK(c_off.restored == -1);
    CHECK(c_off.n_past == 0);

    // an effort change splits inside the system block: the pin moves to that split and C's copies go
    const auto d = serve(on, chat(0, SYS + 1), 40);
    CHECK(d.target == 40 && d.restored == -1 && d.n_past == 0);
    CHECK(find_at(on.ckpts, 40) != nullptr && find_at(on.ckpts, 40)->pinned);
    // the next ordinary chat moves it back to its user message's start, a copy D's read-in saved
    const auto e = serve(on, chat(0, SYS + 1), SYS + 1 + 3 + 50);
    CHECK(e.target == SYS + 1 && e.restored == SYS + 1 && e.n_past == SYS + 1);
    CHECK(find_at(on.ckpts, SYS + 1) != nullptr && find_at(on.ckpts, SYS + 1)->pinned && n_pinned(on.ckpts) == 1);
}

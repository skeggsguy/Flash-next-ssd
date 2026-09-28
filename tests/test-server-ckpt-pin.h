#pragma once

// helpers shared by test-server-ckpt-pin.cpp and test-server-ckpt-pin-chats.cpp (one test, two files to stay short)

#include "server-ckpt-pin.h"

#include <cstdio>
#include <utility>
#include <vector>

extern int n_fail;

#define CHECK(cond) do { if (!(cond)) { ++n_fail; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

inline const auto no_log = [](const common_prompt_checkpoint &, server_ckpt_erase_why) {};

// a chat as sidekick renders it: the system block, the first user message (its <memory> block first), then
// replies and turns; the last span is the generation prompt's assistant opening
inline common_chat_msg_spans spans_of(const std::vector<std::pair<common_chat_role, size_t>> & starts, size_t n) {
    common_chat_msg_spans spans;
    for (size_t i = 0; i < starts.size(); i++) {
        const size_t end = i + 1 < starts.size() ? starts[i + 1].second : n;
        spans.add(starts[i].first, starts[i].second, end - starts[i].second);
    }
    return spans;
}

inline int n_pinned(const server_ckpt_list & ckpts) {
    int n = 0;
    for (const auto & cur : ckpts) {
        n += cur.pinned ? 1 : 0;
    }
    return n;
}

inline const common_prompt_checkpoint * find_at(const server_ckpt_list & ckpts, int64_t n_tokens) {
    for (const auto & cur : ckpts) {
        if (cur.n_tokens == n_tokens) {
            return &cur;
        }
    }
    return nullptr;
}

// what create_checkpoint does to the list: unpin first when this copy is the pin
inline void add(server_ckpt_list & ckpts, bool pin_on, int n_max, int min_step, int id_task, int64_t n_tokens,
        bool pin = false, const server_ckpt_on_erase & on_erase = no_log) {
    if (pin) {
        server_ckpt_unpin(ckpts);
    }
    server_ckpt_add(ckpts, pin_on, n_max, min_step, id_task, n_tokens, (llama_pos) n_tokens - 1, (llama_pos) n_tokens - 1, pin, on_erase);
}

// in test-server-ckpt-pin-chats.cpp
void test_pin_target();
void test_restore_pick();
void test_handback_chats();

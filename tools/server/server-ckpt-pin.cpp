#include "server-ckpt-pin.h"

#include <cstdlib>

bool server_ckpt_pin_parse(const char * value) {
    return value == nullptr ? true : std::atoi(value) > 0;
}

bool server_ckpt_split_is_new_chat(const common_chat_msg_spans & spans, int64_t n_past, int64_t n_old, int64_t n_new) {
    if (n_past <= 0 || n_past >= n_old || n_past >= n_new) {
        return false;
    }
    // the new prompt's first assistant message: a chat's first turn has one only at its end (the generation
    // prompt), so the split must lie before it; a prompt without one (no delimiters) has no reply to re-render
    for (const auto & span : spans.spans) {
        if (span.role == COMMON_CHAT_ROLE_ASSISTANT) {
            return n_past < (int64_t) span.pos;
        }
    }
    return true;
}

int64_t server_ckpt_pin_target(bool pin_on, int32_t n_ctx_checkpoints, const common_chat_msg_spans & spans,
        int64_t n_past, int64_t n_old, int64_t n_new) {
    if (!pin_on || n_ctx_checkpoints < 2) {
        return -1;
    }
    return server_ckpt_split_is_new_chat(spans, n_past, n_old, n_new) ? n_past : -1;
}

int64_t server_ckpt_unpin(server_ckpt_list & ckpts) {
    int64_t was = -1;
    for (auto & cur : ckpts) {
        if (cur.pinned) {
            was = cur.n_tokens;
            cur.pinned = false;
        }
    }
    return was;
}

bool server_ckpt_pin_existing(server_ckpt_list & ckpts, int64_t n_tokens, int64_t * was) {
    auto it = ckpts.begin();
    while (it != ckpts.end() && it->n_tokens != n_tokens) {
        ++it;
    }
    if (it == ckpts.end()) {
        return false;
    }
    *was = server_ckpt_unpin(ckpts);
    it->pinned = true;
    return true;
}

void server_ckpt_invalidate(server_ckpt_list & ckpts, llama_pos pos_next, const server_ckpt_on_erase & on_erase) {
    for (auto it = ckpts.begin(); it != ckpts.end(); ) {
        if (it->pos_max > pos_next) {
            on_erase(*it, SERVER_CKPT_ERASE_INVALIDATED);
            it = ckpts.erase(it);
        } else {
            ++it;
        }
    }
}

common_prompt_checkpoint & server_ckpt_add(server_ckpt_list & ckpts, bool pin_on, int32_t n_max, int32_t min_step,
        int id_task, int64_t n_tokens_new, llama_pos pos_min, llama_pos pos_max, bool pin,
        const server_ckpt_on_erase & on_erase) {
    const auto is_pin = [pin_on](const common_prompt_checkpoint & cur) { return pin_on && cur.pinned; };

    // evict checkpoints within min-step of a previous checkpoint, unless they were
    // created by the current task
    // only when the list is full, otherwise short prompts keep just the oldest checkpoint
    int64_t last = -1;
    for (auto it = ckpts.begin(); ckpts.size() + 1 >= (size_t) n_max && it != ckpts.end(); ) {
        if (it->id_task != id_task && !is_pin(*it) && last >= 0 && it->n_tokens <= last + min_step) {
            on_erase(*it, SERVER_CKPT_ERASE_TOO_CLOSE);
            it = ckpts.erase(it);
            continue;
        }

        last = it->n_tokens;
        ++it;
    }

    // make room for the new checkpoint, oldest first, passing over the pin
    while (!ckpts.empty() && ckpts.size() >= (size_t) n_max) {
        auto it = ckpts.begin();
        while (it != ckpts.end() && is_pin(*it)) {
            ++it;
        }
        if (it == ckpts.end()) {
            it = ckpts.begin(); // only the pin is left (n_max < 2 never pins, so this is a guard)
        }
        on_erase(*it, SERVER_CKPT_ERASE_OLDEST);
        ckpts.erase(it);
    }

    // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
    bool inherit = false;
    for (auto it = ckpts.begin(); it != ckpts.end(); ) {
        if (it->n_tokens == n_tokens_new) {
            inherit = inherit || is_pin(*it);
            on_erase(*it, SERVER_CKPT_ERASE_SUPERSEDED);
            it = ckpts.erase(it);
        } else {
            ++it;
        }
    }

    auto & cur = ckpts.emplace_back();

    cur.id_task = id_task;
    cur.pinned  = pin_on && (pin || inherit);
    cur.update_pos(n_tokens_new, pos_min, pos_max);

    return cur;
}

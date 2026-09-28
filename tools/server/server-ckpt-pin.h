#pragma once

// The bookmark pin (study BOOKMARK-PIN-PLAN.md, decisions 2026-09-28): one of the slot's copies of the fixed
// summary (context checkpoints) is kept at the start of the message where the latest two chats split. A new chat
// shares the fixed start (system prompt, tools) with the chat before it and differs inside the first user message
// (sidekick puts its <memory> block there), so with a copy at that message's start it reads in only the message.
// Measured on the first dogfood day: once a long chat's follow-ups rolled all 4 copies past the start, a new chat
// that shared 13.8K tokens read in all 14.8K (pen to paper 30 s instead of ~5).
//
// Not at the split itself (amendment, Tom, after cleric's review): the memory block opens with fact counts and
// ~600 tokens of always-on facts, so consecutive chats share a varying part of it (900 tokens on the dogfood day).
// A pin at one pair's split is past the next pair's split whenever they share less, and by then the copy at the
// message start has rolled out: that chat read in from 0. The message start costs a memory re-read (~2 s) a chat.
//
// The rules live here, free of the server, so test-server-ckpt-pin covers them without a model;
// server-context.cpp applies them. LLAMA_CKPT_PIN is on unless set to 0, and 0 gives upstream's picks exactly.

#include "chat.h"
#include "common.h"

#include <cstdint>
#include <functional>
#include <list>

using server_ckpt_list = std::list<common_prompt_checkpoint>;

// called for each copy thrown out, before it is erased
enum server_ckpt_erase_why {
    SERVER_CKPT_ERASE_TOO_CLOSE,   // within min-step of an earlier copy
    SERVER_CKPT_ERASE_OLDEST,      // making room, oldest first
    SERVER_CKPT_ERASE_SUPERSEDED,  // the new copy sits at the same n_tokens
    SERVER_CKPT_ERASE_INVALIDATED, // its words are no longer in the context memory (pos_max > pos_next)
};
using server_ckpt_on_erase = std::function<void(const common_prompt_checkpoint &, server_ckpt_erase_why)>;

// LLAMA_CKPT_PIN: on when unset, off when it parses to 0 or less (the fork's env-switch pattern)
bool server_ckpt_pin_parse(const char * value);

// A split (the common prefix n_past of the old and the new prompt) counts as a new chat only if it lies inside
// both prompts and before the new prompt's first assistant message (inside the system prompt or the first user
// message). A follow-up or a tool turn re-renders the last reply and splits at or after an assistant message;
// a retry of the same prompt has no split inside the new prompt.
bool server_ckpt_split_is_new_chat(const common_chat_msg_spans & spans, int64_t n_past, int64_t n_old, int64_t n_new);

// the start of the message span holding pos (span.pos <= pos < span.pos + len), or -1 if no span holds it
int64_t server_ckpt_span_start(const common_chat_msg_spans & spans, int64_t pos);

// where this prompt's read-in pins a copy, or -1. For a new chat's split with the pin on and room for a rolling
// copy beside it (n_ctx_checkpoints >= 2): the start of the message that holds the split, or the split itself if
// that message is the system message, no span holds it, or the message starts at 0 (a copy at 0 holds nothing)
int64_t server_ckpt_pin_target(bool pin_on, int32_t n_ctx_checkpoints, const common_chat_msg_spans & spans,
        int64_t n_past, int64_t n_old, int64_t n_new);

// takes the pin off whichever copy holds it (it becomes an ordinary rolling copy); returns its n_tokens, or -1
int64_t server_ckpt_unpin(server_ckpt_list & ckpts);

// gives the pin to the copy already at n_tokens, if there is one (no new copy needs saving); *was gets the n_tokens
// of the copy that held the pin before (-1: none). Returns false, changing nothing, if no copy sits there.
bool server_ckpt_pin_existing(server_ckpt_list & ckpts, int64_t n_tokens, int64_t * was);

// upstream's restore choice: the newest copy whose words are all still in the new prompt (pos_max <= pos_next) and
// that is old enough for the memory's window (pos_min < pos_min_thold, or 0); nullptr if none. on_check sees each
// copy looked at, newest first.
common_prompt_checkpoint * server_ckpt_restore_pick(server_ckpt_list & ckpts, llama_pos pos_next, llama_pos pos_min_thold,
        const std::function<void(const common_prompt_checkpoint &)> & on_check = nullptr);

// upstream's invalidation: erases the copies whose words are gone (pos_max > pos_next); a pin beyond the split goes too
void server_ckpt_invalidate(server_ckpt_list & ckpts, llama_pos pos_next, const server_ckpt_on_erase & on_erase);

// Makes room for a new copy at n_tokens_new and appends it (positions set, no state data; the caller saves that).
// Upstream's passes in upstream's order: when the list is nearly full, copies within min_step of an earlier one
// (unless made by this task), then the oldest first down to n_max - 1, then any copy at n_tokens_new. With pin_on
// the pinned copy is never thrown out by the first two (it still counts as the earlier copy for min_step), and a
// copy superseding the pin inherits it. pin marks the new copy as the pin (the caller unpins the old one first).
common_prompt_checkpoint & server_ckpt_add(server_ckpt_list & ckpts, bool pin_on, int32_t n_max, int32_t min_step,
        int id_task, int64_t n_tokens_new, llama_pos pos_min, llama_pos pos_max, bool pin,
        const server_ckpt_on_erase & on_erase);

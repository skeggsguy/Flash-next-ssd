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

// Does a bookmark copy (a prompt checkpoint) also save the apprentice's (the draft context's) record? Only when its
// memory cannot be trimmed back to the restore point (study COPY-FIX-PLAN.md step 1). A PART draft is trimmed, never
// copied: after a restore the server removes both contexts' cells past the restore point (common_memory::seq_rm),
// and the apprentice's cells below it hold what a copy would, since each is written once from the library's own
// state at that position and only ever removed from the end (the speculative path already treats a PART draft this
// way: use_ckpt_dft). Copying it was ~3/4 of each copy (~372 of ~488 MiB at 187K) and grew the heap ~0.7 GiB a long
// follow-up. FULL must be copied; RS rolls back only a bounded number of cells and a retry can rewind any distance,
// so it is copied too; NO means no draft context. load_dft is a no-op on the empty record a skipped copy leaves.
bool server_ckpt_copy_dft(common_context_seq_rm_type dft_type);

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

// The pin learns lanes (study PLAN #31, decisions 2026-10-14). The split rule says "a new chat" only when the seat's
// last prompt was the same kind of work: chat's and code's prompts part at token 3 (the template's effort line), so a
// chat after code split there and pinned at 3, inside the system prompt. That pin blocked the first user message's
// pin (server_ckpt_pin_first_user), and the opening copy saved 3 tokens when coding took the seat (dogfood 2026-10-14:
// three chats read in 5-7K tokens each). So the split rule runs only when the seat's previous task was in this task's
// lane (NONE counts as chat, so with the chat window off nothing changes). A chat after code pins nothing at its split
// and takes off any pin code left on the seat, so the first user message takes it; code after code keeps its split
// pin (code's own ~10.6K pin serves code); code after chat leaves chat's pin alone.
struct server_ckpt_lane_pin {
    int64_t target;   // where this read-in pins a copy (server_ckpt_pin_target), or -1
    int64_t skipped;  // the split rule's target, set aside because the lane changed (for the log), or -1
    int64_t unpinned; // the pin a chat after code took off the seat (now an ordinary rolling copy), or -1
};
server_ckpt_lane_pin server_ckpt_pin_target_lanes(bool pin_on, int32_t n_ctx_checkpoints, bool prev_code,
        bool task_code, server_ckpt_list & ckpts, const common_chat_msg_spans & spans, int64_t n_past, int64_t n_old,
        int64_t n_new);

// the first user message's start, where a chat's opening ends (sidekick's ~4.2-5K tokens), or -1 if it has none
int64_t server_ckpt_first_user_start(const common_chat_msg_spans & spans);

// the n_tokens of the copy holding the pin, or -1
int64_t server_ckpt_pinned_at(const server_ckpt_list & ckpts);

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

// The opening copy (--cache-ram-opening, study CHAT-RETURN-PLAN.md step 1, decisions 2026-10-12): the saved-prompts
// store (--cache-ram) keeps chat's opening only. When a chat leaves the seat, the seat is stepped back to its pinned
// copy (the end of sidekick's opening, ~4.2K tokens) and only that is saved, so a new chat after a coding request
// reads in only its own message. The stock store saves the whole chat (682 MiB at 8K, 1.06 GiB at 21K), and a new
// chat shares too little of a long one for the store to hand it back (it wants 25%). A seat's lane is chat or code
// (server-lane.h); every request is chat while the chat window is off.

// what the store does when the seat is handed to another task (get_available_slot's save point)
struct server_ckpt_hand_on {
    bool save; // save the seat (with the switch: stepped back to its pinned copy first)
    bool load; // look in the store for the new prompt (stock: load a better entry, else clear the seat)
};

// Stock (opening_on false): save and load. With the switch, a chat seat handed to a chat task does neither: the same
// chat going on (a follow-up, a retry) keeps its whole bookmark, and a new chat keeps the opening from the seat itself,
// since the seat already holds the pinned copy and the read-in's own restore steps back to it (the path from before the
// switch). The copy is saved when coding takes the seat. Saving here as well kept a second copy of the opening in the
// store beside the seat's (fix round of step 3's review, 2026-10-12). An empty chat seat (n_seat 0) holds no opening,
// so it still looks in the store. Otherwise only a chat seat is saved and only a chat task takes from the store, so
// coding never takes chat's copy.
server_ckpt_hand_on server_ckpt_opening_hand_on(bool opening_on, bool seat_chat, bool task_chat, int64_t n_seat);

// The step back's list: erases every copy but the pinned one and returns it. nullptr, leaving the list untouched, when
// there is no pin, it lies outside the seat's tokens (0 < n_tokens <= n_tokens_seat), or it sits below opening_floor:
// then nothing is saved. opening_floor (PLAN #31) is the first user message's start of the task that pinned (the
// seat records it): a pin below it sits inside the system prompt (the 3-token pin of 2026-10-14), a copy no new chat
// gains from, so the store keeps whatever it holds and the seat is not stepped back. 0: no floor.
common_prompt_checkpoint * server_ckpt_keep_pin(server_ckpt_list & ckpts, int64_t n_tokens_seat, int64_t opening_floor = 0);

// One copy of the opening (Tom, decisions 2026-10-12; cleric's review of step 1): the saved entry carries no bookmark
// copies. A seat stepped back to its pin holds exactly the opening, so a new chat that takes the entry reads on from
// its end with no restore, and its read-in makes its own copy at its first user message, which rule 3 pins; the
// pinned copy inside the entry (~113 MiB + ~2 KB a token) would never be used. `save` runs with the seat's copies set
// aside, so the store neither copies them nor counts them against its limit, and they are put back afterwards (also
// if save throws), at the same addresses: the seat keeps its pin for the tie, where it serves the new chat itself.
// Returns what save returns.
bool server_ckpt_save_without_copies(server_ckpt_list & ckpts, const std::function<bool()> & save);

// A first chat keeps its opening too: with the switch on (and the pin, with room for it), a chat task's copy made at
// its first user message's start takes the pin when no copy holds one yet. The read-in already makes that copy (it
// breaks there), so no batch changes. The split rule still moves the pin later. A chat after code gets here too:
// server_ckpt_pin_target_lanes takes the pin code left on the seat off first.
bool server_ckpt_pin_first_user(bool opening_on, bool pin_on, int32_t n_ctx_checkpoints, bool task_chat,
        const server_ckpt_list & ckpts, const common_chat_msg_spans & spans, int64_t n_tokens_start);

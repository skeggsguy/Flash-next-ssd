// Borrowing from the bookmark shelf (server-shelf.h; study SHELF-BORROW-PLAN.md): a new conversation reaches into an
// old bookmark for the opening it shares.
//
// A new chat shares the shared opening (sidekick's system prompt and tools, ~4K tokens) with every old one, but that
// is ~8% of a 50K bookmark, so the 25% rule (server_shelf::pick) never hands it one, and with coding on the seat the
// chat read its opening in again (~10 s). Every old chat's bookmark holds a saved copy at the end of that opening
// (the pin, server_ckpt_pin_first_user), so the new chat may load an old bookmark, let the read-in step back to that
// copy as it does for any copy, and read in only its own text. The load costs ~0.4 s at 50K, ~1.3 s at 150K.
//
// A borrow keeps the file: the old conversation still returns from its own bookmark (server_shelf::hand_out).

#include "server-shelf.h"

#include <algorithm>

bool server_shelf_borrow_saves(int64_t n_read, uint64_t size) {
    if (n_read <= 0) {
        return false;
    }
    const double s_read = (double) n_read / SERVER_SHELF_BORROW_READ_TPS;
    const double s_load = (double) size / SERVER_SHELF_BORROW_LOAD_BPS;
    return s_read - s_load >= SERVER_SHELF_BORROW_MARGIN_S;
}

// a copy at p serves a task sharing n_shared tokens with its bookmark when the read-in's restore can land on it
// (server_ckpt_restore_pick: a copy's last position p - 1 at most the split's, and its first below it, as a
// recurrent memory's copy has) and leaves at least one token to read in
static bool server_shelf_copy_serves(int64_t p, int64_t n_shared, int64_t n_task) {
    return p > 0 && p <= n_shared && p < n_task;
}

int64_t server_shelf_seat_reach(const server_prompt & seat, const server_tokens & task, bool cuts_anywhere) {
    if (seat.tokens.size() == 0 || task.size() == 0) {
        return 0;
    }
    const int64_t n_shared = (int64_t) seat.tokens.get_common_prefix(task);
    if (cuts_anywhere) {
        return n_shared;
    }
    int64_t reach = 0;
    for (const auto & c : seat.checkpoints) {
        if (server_shelf_copy_serves(c.n_tokens, n_shared, (int64_t) task.size())) {
            reach = std::max<int64_t>(reach, c.n_tokens);
        }
    }
    return reach;
}

server_shelf_take server_shelf::borrow_pick(const server_prompt & seat, const server_tokens & task,
        bool seat_cuts_anywhere) const {
    server_shelf_take best;
    if (task.size() == 0) {
        return best;
    }
    const int64_t n_task = (int64_t) task.size();
    const int64_t reach  = server_shelf_seat_reach(seat, task, seat_cuts_anywhere);

    uint64_t best_size = 0;
    for (const auto & e : entries) {
        if (e.tokens.size() == 0 || e.copies.empty() || !e.spec_synced) {
            continue;
        }
        const int64_t n_shared = (int64_t) e.tokens.get_common_prefix(task);
        int64_t p = 0;
        for (const int64_t c : e.copies) {
            if (server_shelf_copy_serves(c, n_shared, n_task)) {
                p = std::max(p, c);
            }
        }
        if (p <= reach || !server_shelf_borrow_saves(p - reach, e.size)) {
            continue;
        }
        if (p > best.borrow || (p == best.borrow && e.size < best_size)) {
            best.id     = e.id;
            best.borrow = p;
            best.shared = n_shared;
            best_size   = e.size;
        }
    }
    return best;
}

server_shelf_take server_shelf::take(const server_prompt & seat, const server_tokens & task,
        bool seat_cuts_anywhere) const {
    server_shelf_take res;
    res.id = pick(seat.tokens, task);
    if (res.id != 0) {
        return res;
    }
    return borrow_pick(seat, task, seat_cuts_anywhere);
}

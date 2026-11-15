// A seat on the bookmark shelf (server-shelf.h): the library's and the apprentice's states streamed straight between
// their contexts and the shelf's file (llama_state_seq_write_stream / _read_stream, a few MiB at a time), and the
// log lines a probe reads: "bookmark shelf: saved <n> tokens, <MiB> MiB, <ms> ms", "... loaded ..." and a borrow's
// "bookmark shelf: borrowed <p> of <n> tokens from a bookmark, <MiB> MiB, <ms> ms (kept on the shelf ...)".

#include "server-shelf.h"

#include "log.h"
#include "speculative.h"

#include <cinttypes>
#include <filesystem>
#include <system_error>

static double shelf_mib(uint64_t n) { return n / (1024.0 * 1024.0); }
static double shelf_gib(uint64_t n) { return n / (1024.0 * 1024.0 * 1024.0); }

// one llama state's part of the file: the callbacks may move at most the part's size
struct shelf_cb_data {
    server_shelf_file * f;
    uint64_t            left;
};

static bool shelf_put_cb(const void * src, size_t size, void * user_data) {
    auto * d = (shelf_cb_data *) user_data;
    if (size > d->left) {
        return false;
    }
    d->left -= size;
    return d->f->write(src, size);
}

static bool shelf_get_cb(void * dst, size_t size, void * user_data) {
    auto * d = (shelf_cb_data *) user_data;
    if (size > d->left) {
        return false;
    }
    d->left -= size;
    return d->f->read(dst, size);
}

static server_shelf_part shelf_put_state(llama_context * ctx, llama_seq_id id) {
    return [ctx, id](server_shelf_file & f, uint64_t size) {
        if (ctx == nullptr) {
            return false;
        }
        shelf_cb_data d = { &f, size };
        return llama_state_seq_write_stream(ctx, id, LLAMA_STATE_SEQ_FLAGS_NONE, shelf_put_cb, &d) == size;
    };
}

static server_shelf_part shelf_get_state(llama_context * ctx, llama_seq_id id) {
    return [ctx, id](server_shelf_file & f, uint64_t size) {
        if (ctx == nullptr) {
            return false; // a file with an apprentice's state, a server without one
        }
        shelf_cb_data d = { &f, size };
        return llama_state_seq_read_stream(ctx, id, LLAMA_STATE_SEQ_FLAGS_NONE, shelf_get_cb, &d) == size;
    };
}

static uint64_t shelf_copies_size(const std::list<common_prompt_checkpoint> & ckpts) {
    uint64_t n = 0;
    for (const auto & c : ckpts) {
        n += c.size();
    }
    return n;
}

bool server_shelf_save_seat(server_shelf & shelf, const server_shelf_seat & seat, const server_prompt & prompt,
        bool spec_synced, uint64_t keep_id) {
    const int n_tokens = prompt.n_tokens();
    if (n_tokens == 0) {
        return false;
    }
    if (shelf.holds(prompt.tokens)) {
        SRV_WRN("bookmark shelf: not saved, %d tokens already on the shelf\n", n_tokens);
        return false;
    }

    const int64_t t_start = ggml_time_us();

    const uint64_t size_tgt = llama_state_seq_get_size_ext(seat.ctx_tgt, seat.id, LLAMA_STATE_SEQ_FLAGS_NONE);
    const uint64_t size_dft = seat.ctx_dft ? llama_state_seq_get_size_ext(seat.ctx_dft, seat.id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    std::vector<uint8_t> data_spec;
    common_speculative_get_state(seat.spec, seat.id, data_spec);

    const uint64_t size_new = server_shelf_record_size(size_tgt, size_dft, data_spec.size(), prompt.checkpoints);

    const server_shelf_room room = shelf.room_for(prompt.tokens, size_new, keep_id);
    if (!room.fits) {
        uint64_t free_now = 0;
        const bool known = shelf.free_bytes && shelf.free_bytes(free_now);
        SRV_WRN("bookmark shelf: not saved, %d tokens, %.1f MiB do not fit (limit %.1f GiB, on the shelf %.2f GiB, "
                "the drive's free space %s%.1f GiB, kept free %.1f GiB)\n", n_tokens, shelf_mib(size_new),
                shelf_gib(shelf.lim.limit), shelf_gib(shelf.size()), known ? "" : "unknown, ", shelf_gib(free_now),
                shelf_gib(shelf.lim.reserve));
        return false;
    }
    for (const uint64_t id : shelf.ids_at(room.drop)) {
        const auto * e = shelf.find(id);
        const bool superseded = e->tokens.get_common_prefix(prompt.tokens) == e->tokens.size();
        SRV_WRN("bookmark shelf: dropped %zu tokens, %.1f MiB (%s)\n", e->tokens.size(), shelf_mib(e->size),
                superseded ? "all of it is in the new save" : "the oldest, to make room");
        shelf.drop(id);
    }

    const uint64_t id   = shelf.next_id++;
    const std::string path = shelf.path_for(id);

    server_shelf_head head;
    head.n_tokens    = (uint64_t) n_tokens;
    head.spec_synced = spec_synced;

    server_shelf_file f;
    bool ok = f.open_write(path) &&
              server_shelf_write_record(f, head, size_tgt, shelf_put_state(seat.ctx_tgt, seat.id),
                      size_dft, shelf_put_state(seat.ctx_dft, seat.id), data_spec, prompt.checkpoints);
    ok = f.close() && ok;
    if (!ok || f.n_bytes() != size_new) {
        SRV_ERR("bookmark shelf: save failed, %d tokens: %s\n", n_tokens,
                f.error().empty() ? "a state did not stream whole" : f.error().c_str());
        std::error_code ec;
        std::filesystem::remove(path, ec);
        return false;
    }

    shelf.add(id, prompt.tokens.clone(), path, size_new, prompt.checkpoints, spec_synced);

    const double ms = (ggml_time_us() - t_start) / 1000.0;
    SRV_WRN("bookmark shelf: saved %d tokens, %.1f MiB, %.0f ms (apprentice %.1f MiB, %zu copies %.1f MiB%s; "
            "shelf: %zu bookmarks, %.2f of %.1f GiB)\n", n_tokens, shelf_mib(size_new), ms, shelf_mib(size_dft),
            prompt.checkpoints.size(), shelf_mib(shelf_copies_size(prompt.checkpoints)),
            f.nocache() ? "" : ", through the file cache", shelf.entries.size(), shelf_gib(shelf.size()),
            shelf_gib(shelf.lim.limit));
    return true;
}

bool server_shelf_load_seat(server_shelf & shelf, const server_shelf_seat & seat, uint64_t id,
        server_prompt & prompt, bool & spec_synced, int64_t borrow_at) {
    auto * e = shelf.find(id);
    if (e == nullptr) {
        return false;
    }

    const int64_t t_start = ggml_time_us();
    const int     n_tokens = (int) e->tokens.size();
    const uint64_t size    = e->size;

    server_shelf_head head;
    uint64_t size_dft = 0;
    std::vector<uint8_t> data_spec;
    std::list<common_prompt_checkpoint> ckpts;

    server_shelf_file f;
    bool ok = f.open_read(e->path) &&
              server_shelf_read_record(f, head, shelf_get_state(seat.ctx_tgt, seat.id),
                      shelf_get_state(seat.ctx_dft, seat.id), size_dft, data_spec, ckpts);
    // the same server wrote it, so the apprentice's state is there whenever the seat has an apprentice's context
    ok = ok && head.n_tokens == (uint64_t) n_tokens && (seat.ctx_dft == nullptr || size_dft > 0);
    f.close();

    if (!ok) {
        SRV_ERR("bookmark shelf: load failed, %d tokens: %s; the seat reads in from the start\n", n_tokens,
                f.error().empty() ? "the file does not match" : f.error().c_str());
        shelf.drop(id);
        return false;
    }

    // the apprentice: its memory came with the file; its carry state is the file's, from a clean start, so the
    // first draft after a return is not cold (as a copy's restore does, common_speculative_set_state)
    common_speculative_reset(seat.spec, seat.id);
    if (!data_spec.empty()) {
        common_speculative_set_state(seat.spec, seat.id, data_spec);
    }

    const bool borrow = borrow_at >= 0;

    prompt.tokens      = shelf.hand_out(id, borrow); // e is not used past here
    prompt.checkpoints = std::move(ckpts);
    spec_synced        = head.spec_synced;

    const double ms = (ggml_time_us() - t_start) / 1000.0;
    if (borrow) {
        SRV_WRN("bookmark shelf: borrowed %" PRId64 " of %d tokens from a bookmark, %.1f MiB, %.0f ms (kept on the "
                "shelf, now the newest; apprentice %.1f MiB, %zu copies; shelf: %zu bookmarks, %.2f GiB)\n", borrow_at,
                n_tokens, shelf_mib(size), ms, shelf_mib(size_dft), prompt.checkpoints.size(), shelf.entries.size(),
                shelf_gib(shelf.size()));
        return true;
    }
    SRV_WRN("bookmark shelf: loaded %d tokens, %.1f MiB, %.0f ms (apprentice %.1f MiB%s, %zu copies; "
            "shelf: %zu bookmarks, %.2f GiB)\n", n_tokens, shelf_mib(size), ms, shelf_mib(size_dft),
            head.spec_synced ? "" : ", behind the library: rebuilt at the next draft", prompt.checkpoints.size(),
            shelf.entries.size(), shelf_gib(shelf.size()));
    return true;
}

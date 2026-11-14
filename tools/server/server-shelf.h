#pragma once

// The bookmark shelf (--cache-disk DIR, --cache-disk-gib N; study BOOKMARK-SHELF-PLAN.md step 2): the prompt cache's
// entries kept on an SSD, internal or external, instead of in RAM. When another conversation takes the seat, the
// outgoing conversation's whole bookmark (the library's context memory, the apprentice's (the draft's) memory, the
// saved copies (context checkpoints) and the apprentice's carry state) goes to one file in DIR; RAM keeps only its
// tokens, for the store's choice. When the conversation comes back, the file is read straight into the seat and
// the request reads in only its new text: ~1-2 s for a 200K conversation instead of minutes of reading in.
//
// The store's rules are kept (server_prompt_cache in server-task.cpp): the choice by shared prefix with the 25% rule,
// oldest-first eviction, a loaded entry leaving the store. What differs: the limit is the shelf's GiB, capped at open
// to what the drive can spare (its free space less a 10% reserve) and re-checked against the drive before each save;
// the shelf's files are deleted at open (a bookmark is only valid for the engine and model that wrote it); a save
// that does not fit is skipped, never half-written. Off unless --cache-disk is given.
//
// Rules and the file format live here, free of a model, so test-server-shelf covers them; the llama side (a seat's
// states streamed in and out) is in server-shelf-seat.cpp, and server-context.cpp decides when to save and load.

#include "server-shelf-io.h"
#include "server-task.h"

#include <cstdint>
#include <functional>
#include <list>
#include <string>
#include <vector>

//
// rules
//

// the limit the shelf works to: the asked GiB, capped to the drive's free space less a reserve of 10% of it
struct server_shelf_limit {
    uint64_t limit   = 0; // bytes the shelf may hold
    uint64_t reserve = 0; // bytes the shelf always leaves free on the drive
    bool     capped  = false;
};

server_shelf_limit server_shelf_cap_limit(uint64_t asked, uint64_t free_bytes);

// a saved bookmark, as the room rule sees it (oldest first)
struct server_shelf_room_item {
    uint64_t size       = 0;
    bool     keep       = false; // the entry about to be loaded: never dropped
    bool     superseded = false; // all of its tokens are in the save: dropped whenever the save goes ahead
};

struct server_shelf_room {
    bool                fits = false;
    std::vector<size_t> drop;        // indices into the items, superseded ones first, then oldest first
};

// what a save of size_new bytes drops: the superseded entries, then the oldest until the shelf stays within limit
// and the drive keeps reserve bytes free (free_now: the drive's free bytes now, which each drop adds to); a save
// that cannot fit even with every droppable entry gone drops nothing
server_shelf_room server_shelf_make_room(const std::vector<server_shelf_room_item> & items, uint64_t size_new,
        uint64_t limit, uint64_t free_now, uint64_t reserve);

// the hand-over of a seat to a task (server-context.cpp, get_available_slot): with n_seat tokens on the seat, lcp
// of them shared with the task, losing (the slot choice's own "about to lose a large part": update_cache) and pick
// (the shelf's choice for the task, server_shelf::pick, 0 = none):
// - a task carrying on the seat's conversation (all of the seat shared) neither saves nor loads, and nothing on the
//   shelf can beat the seat then (the caller need not pick)
// - the seat is saved when the task keeps under half of it (losing, or f_keep < 0.5 for a slot asked for by id), or
//   when a bookmark is about to replace it; an empty seat has nothing to save
// - a picked bookmark is loaded
struct server_shelf_hand {
    bool save = false;
    bool load = false;
};

inline bool server_shelf_carries_on(size_t n_seat, size_t lcp) {
    return n_seat > 0 && lcp == n_seat;
}

server_shelf_hand server_shelf_hand_on(size_t n_seat, size_t lcp, bool losing, uint64_t pick);

//
// the shelf
//

struct server_shelf_entry {
    uint64_t      id = 0;
    server_tokens tokens;   // the conversation, for the choice; everything else is in the file
    std::string   path;
    uint64_t      size = 0; // the file's bytes
};

struct server_shelf {
    std::string dir;
    uint64_t    limit_asked  = 0;
    uint64_t    free_at_open = 0;
    server_shelf_limit lim;

    std::list<server_shelf_entry> entries; // oldest first
    uint64_t next_id = 1;

    // the drive's free bytes (server_shelf_free_bytes on dir); tests set their own
    std::function<bool(uint64_t &)> free_bytes;

    // checks dir, deletes its old bookmark files, reads the free space and sets the limit; "" or why the shelf
    // cannot open
    std::string open(const std::string & dir, uint64_t limit_bytes);

    uint64_t size() const;
    server_shelf_entry * find(uint64_t id);

    // the store's choice (server_prompt_cache::load) for a request on a seat holding seat: the entry sharing the
    // most with the request, if it beats the seat both on the share of itself kept (f_keep) and of the request
    // covered (f_sim), never one keeping under 25% of itself; its id, or 0
    uint64_t pick(const server_tokens & seat, const server_tokens & task) const;

    // an entry already holds all of tokens: a save would add nothing
    bool holds(const server_tokens & tokens) const;

    // the room a save of size_new bytes of tokens needs (server_shelf_make_room on the drive's free space now);
    // keep_id is never dropped. Nothing is dropped here: drop() what it lists
    server_shelf_room room_for(const server_tokens & tokens, uint64_t size_new, uint64_t keep_id) const;

    // the ids at room.drop's indices
    std::vector<uint64_t> ids_at(const std::vector<size_t> & idx) const;

    std::string path_for(uint64_t id) const;

    // the entry leaves the shelf and its file is deleted
    void drop(uint64_t id);
};

//
// the file: a header, the library's state, the apprentice's state, the carry state, the copies, a trailer
//

struct server_shelf_head {
    uint64_t n_tokens    = 0;
    bool     spec_synced = true; // the apprentice's memory covers the whole conversation (spec_context_synced)
};

// a part of the record whose bytes come from elsewhere (a llama state): passes exactly size bytes through the file
using server_shelf_part = std::function<bool(server_shelf_file & f, uint64_t size)>;

// the record's exact size in bytes
uint64_t server_shelf_record_size(uint64_t size_tgt, uint64_t size_dft, uint64_t size_spec,
        const std::list<common_prompt_checkpoint> & ckpts);

bool server_shelf_write_record(server_shelf_file & f, const server_shelf_head & head,
        uint64_t size_tgt, const server_shelf_part & put_tgt, uint64_t size_dft, const server_shelf_part & put_dft,
        const std::vector<uint8_t> & data_spec, const std::list<common_prompt_checkpoint> & ckpts);

// get_tgt / get_dft are called with each part's size (get_dft not at all when it is 0); size_dft_out tells the
// caller whether the apprentice's state was in the file
bool server_shelf_read_record(server_shelf_file & f, server_shelf_head & head,
        const server_shelf_part & get_tgt, const server_shelf_part & get_dft, uint64_t & size_dft_out,
        std::vector<uint8_t> & data_spec, std::list<common_prompt_checkpoint> & ckpts);

//
// a seat on the shelf (server-shelf-seat.cpp): the llama side
//

struct common_speculative;

struct server_shelf_seat {
    llama_context      * ctx_tgt = nullptr;
    llama_context      * ctx_dft = nullptr; // the apprentice's context, if it has one
    common_speculative * spec    = nullptr;
    llama_seq_id         id      = 0;
};

// saves the seat's prompt with its states to a new file, keep_id (the entry about to be loaded) never dropped to make
// room; false if nothing was saved (already on the shelf, no room, or a write failed: logged)
bool server_shelf_save_seat(server_shelf & shelf, const server_shelf_seat & seat, const server_prompt & prompt,
        bool spec_synced, uint64_t keep_id);

// loads entry id into the seat: its states, prompt (tokens and copies) and the apprentice's carry state, reset
// first; the entry leaves the shelf either way. false: the seat's memory is half-written, the caller clears it
bool server_shelf_load_seat(server_shelf & shelf, const server_shelf_seat & seat, uint64_t id,
        server_prompt & prompt, bool & spec_synced);

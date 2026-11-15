#include "server-shelf.h"

#include <cstring>
#include <filesystem>
#include <system_error>

//
// rules
//

server_shelf_limit server_shelf_cap_limit(uint64_t asked, uint64_t free_bytes) {
    server_shelf_limit res;
    res.reserve = free_bytes / 10;
    const uint64_t usable = free_bytes - res.reserve;
    res.capped = asked > usable;
    res.limit  = res.capped ? usable : asked;
    return res;
}

server_shelf_room server_shelf_make_room(const std::vector<server_shelf_room_item> & items, uint64_t size_new,
        uint64_t limit, uint64_t free_now, uint64_t reserve) {
    server_shelf_room res;

    uint64_t used = 0;
    for (const auto & it : items) {
        used += it.size;
    }

    uint64_t freed = 0;
    const auto fits = [&]() {
        return used - freed + size_new <= limit && free_now + freed >= size_new + reserve;
    };

    for (size_t i = 0; i < items.size(); i++) {
        if (items[i].superseded && !items[i].keep) {
            res.drop.push_back(i);
            freed += items[i].size;
        }
    }
    for (size_t i = 0; i < items.size() && !fits(); i++) {
        if (!items[i].superseded && !items[i].keep) {
            res.drop.push_back(i);
            freed += items[i].size;
        }
    }

    res.fits = fits();
    if (!res.fits) {
        res.drop.clear();
    }
    return res;
}

server_shelf_hand server_shelf_hand_on(size_t n_seat, size_t lcp, bool losing, uint64_t pick) {
    server_shelf_hand res;
    if (server_shelf_carries_on(n_seat, lcp)) {
        return res;
    }
    losing   = losing || 2 * lcp < n_seat;
    res.save = n_seat > 0 && (losing || pick != 0);
    res.load = pick != 0;
    return res;
}

//
// the shelf
//

std::string server_shelf::open(const std::string & d, uint64_t limit_bytes) {
    dir         = d;
    limit_asked = limit_bytes;
    entries.clear();

    const std::string why = server_shelf_check_dir(dir);
    if (!why.empty()) {
        return why;
    }
    if (!free_bytes) {
        free_bytes = [this](uint64_t & out) { return server_shelf_free_bytes(dir, out); };
    }

    server_shelf_clear_dir(dir);

    if (!free_bytes(free_at_open)) {
        return "cannot read the drive's free space";
    }
    lim = server_shelf_cap_limit(limit_asked, free_at_open);
    if (lim.limit == 0) {
        return "a limit of 0 bytes (--cache-disk-gib)";
    }
    return "";
}

uint64_t server_shelf::size() const {
    uint64_t res = 0;
    for (const auto & e : entries) {
        res += e.size;
    }
    return res;
}

server_shelf_entry * server_shelf::find(uint64_t id) {
    for (auto & e : entries) {
        if (e.id == id) {
            return &e;
        }
    }
    return nullptr;
}

uint64_t server_shelf::pick(const server_tokens & seat, const server_tokens & task) const {
    if (task.size() == 0) {
        return 0;
    }

    const size_t lcp_seat = seat.get_common_prefix(task);

    float f_keep_best = seat.size() > 0 ? float(lcp_seat) / seat.size() : -1.0f; // an empty seat: any entry wins
    float f_sim_best  = float(lcp_seat) / task.size();

    uint64_t best = 0;
    for (const auto & e : entries) {
        if (e.tokens.size() == 0) {
            continue;
        }
        const size_t lcp = e.tokens.get_common_prefix(task);

        const float f_keep = float(lcp) / e.tokens.size();
        const float f_sim  = float(lcp) / task.size();

        // don't trash large prompts
        if (f_keep < 0.25f) {
            continue;
        }
        if (f_keep_best < f_keep && f_sim_best < f_sim) {
            f_keep_best = f_keep;
            f_sim_best  = f_sim;
            best        = e.id;
        }
    }
    return best;
}

bool server_shelf::holds(const server_tokens & tokens) const {
    for (const auto & e : entries) {
        if (e.tokens.get_common_prefix(tokens) == tokens.size()) {
            return true;
        }
    }
    return false;
}

server_shelf_room server_shelf::room_for(const server_tokens & tokens, uint64_t size_new, uint64_t keep_id) const {
    std::vector<server_shelf_room_item> items;
    items.reserve(entries.size());
    for (const auto & e : entries) {
        server_shelf_room_item it;
        it.size       = e.size;
        it.keep       = e.id == keep_id;
        it.superseded = e.tokens.get_common_prefix(tokens) == e.tokens.size();
        items.push_back(it);
    }

    uint64_t free_now = 0;
    if (!free_bytes || !free_bytes(free_now)) {
        return {}; // the drive cannot be read: no save
    }
    return server_shelf_make_room(items, size_new, lim.limit, free_now, lim.reserve);
}

std::vector<uint64_t> server_shelf::ids_at(const std::vector<size_t> & idx) const {
    std::vector<uint64_t> ids;
    std::vector<uint64_t> all;
    for (const auto & e : entries) {
        all.push_back(e.id);
    }
    for (size_t i : idx) {
        if (i < all.size()) {
            ids.push_back(all[i]);
        }
    }
    return ids;
}

std::string server_shelf::path_for(uint64_t id) const {
    return (std::filesystem::path(dir) / server_shelf_file_name(id)).string();
}

void server_shelf::drop(uint64_t id) {
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->id == id) {
            std::error_code ec;
            std::filesystem::remove(it->path, ec);
            entries.erase(it);
            return;
        }
    }
}

void server_shelf::add(uint64_t id, server_tokens tokens, const std::string & path, uint64_t size,
        const std::list<common_prompt_checkpoint> & ckpts, bool spec_synced) {
    server_shelf_entry e;
    e.id          = id;
    e.tokens      = std::move(tokens);
    e.path        = path;
    e.size        = size;
    e.copies      = server_shelf_copy_positions(ckpts);
    e.spec_synced = spec_synced;
    entries.push_back(std::move(e));
}

server_tokens server_shelf::hand_out(uint64_t id, bool keep) {
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->id != id) {
            continue;
        }
        if (keep) {
            entries.splice(entries.end(), entries, it); // the newest end; the entry itself is not moved in memory
            return it->tokens.clone();
        }
        server_tokens tokens = std::move(it->tokens);
        drop(id);
        return tokens;
    }
    return server_tokens();
}

std::vector<int64_t> server_shelf_copy_positions(const std::list<common_prompt_checkpoint> & ckpts) {
    std::vector<int64_t> res;
    res.reserve(ckpts.size());
    for (const auto & c : ckpts) {
        res.push_back(c.n_tokens);
    }
    return res;
}

//
// the file
//

static const char     SHELF_MAGIC[8]  = { 'B', 'K', 'S', 'H', 'E', 'L', 'F', '1' };
static const char     SHELF_TAIL[8]   = { 'B', 'K', 'S', 'H', 'E', 'N', 'D', '1' };
static const uint32_t SHELF_VERSION   = 1;
static const uint64_t SHELF_HEAD_SIZE = 8 + 4 + 4 + 8 + 4 + 4; // magic, version, flags, n_tokens, n_ckpt, reserved
static const uint64_t SHELF_CKPT_HEAD = 8 + 4 + 4 + 4 + 4;     // n_tokens, id_task, pinned, pos_min, pos_max

template <typename T> static bool put(server_shelf_file & f, const T & v) { return f.write(&v, sizeof(v)); }
template <typename T> static bool get(server_shelf_file & f, T & v)       { return f.read (&v, sizeof(v)); }

static bool put_blob(server_shelf_file & f, const std::vector<uint8_t> & v) {
    return put(f, (uint64_t) v.size()) && (v.empty() || f.write(v.data(), v.size()));
}

static bool get_blob(server_shelf_file & f, std::vector<uint8_t> & v) {
    uint64_t n = 0;
    if (!get(f, n) || n > f.remaining()) {
        return false; // a size past the file's end: never allocate it
    }
    v.resize(n);
    return n == 0 || f.read(v.data(), n);
}

// a part passed through by its owner must move exactly size bytes
static bool pass_part(server_shelf_file & f, const server_shelf_part & part, uint64_t size) {
    const uint64_t before = f.n_bytes();
    return part(f, size) && f.n_bytes() - before == size;
}

uint64_t server_shelf_record_size(uint64_t size_tgt, uint64_t size_dft, uint64_t size_spec,
        const std::list<common_prompt_checkpoint> & ckpts) {
    uint64_t n = SHELF_HEAD_SIZE + 8 + size_tgt + 8 + size_dft + 8 + size_spec;
    for (const auto & c : ckpts) {
        n += SHELF_CKPT_HEAD + 3*8 + c.data_tgt.size() + c.data_dft.size() + c.data_spec.size();
    }
    return n + sizeof(SHELF_TAIL);
}

bool server_shelf_write_record(server_shelf_file & f, const server_shelf_head & head,
        uint64_t size_tgt, const server_shelf_part & put_tgt, uint64_t size_dft, const server_shelf_part & put_dft,
        const std::vector<uint8_t> & data_spec, const std::list<common_prompt_checkpoint> & ckpts) {
    const uint32_t flags = head.spec_synced ? 1u : 0u;
    bool ok = f.write(SHELF_MAGIC, sizeof(SHELF_MAGIC)) && put(f, SHELF_VERSION) && put(f, flags) &&
              put(f, head.n_tokens) && put(f, (uint32_t) ckpts.size()) && put(f, (uint32_t) 0);

    ok = ok && put(f, size_tgt) && pass_part(f, put_tgt, size_tgt);
    ok = ok && put(f, size_dft) && (size_dft == 0 || pass_part(f, put_dft, size_dft));
    ok = ok && put_blob(f, data_spec);

    for (const auto & c : ckpts) {
        if (!ok) {
            break;
        }
        ok = put(f, (int64_t) c.n_tokens) && put(f, (int32_t) c.id_task) && put(f, (uint32_t) (c.pinned ? 1 : 0)) &&
             put(f, (int32_t) c.pos_min) && put(f, (int32_t) c.pos_max) &&
             put_blob(f, c.data_tgt) && put_blob(f, c.data_dft) && put_blob(f, c.data_spec);
    }

    return ok && f.write(SHELF_TAIL, sizeof(SHELF_TAIL));
}

bool server_shelf_read_record(server_shelf_file & f, server_shelf_head & head,
        const server_shelf_part & get_tgt, const server_shelf_part & get_dft, uint64_t & size_dft_out,
        std::vector<uint8_t> & data_spec, std::list<common_prompt_checkpoint> & ckpts) {
    char     magic[8] = {};
    uint32_t version  = 0;
    uint32_t flags    = 0;
    uint32_t n_ckpt   = 0;
    uint32_t reserved = 0;
    if (!f.read(magic, sizeof(magic)) || std::memcmp(magic, SHELF_MAGIC, sizeof(magic)) != 0 ||
        !get(f, version) || version != SHELF_VERSION ||
        !get(f, flags) || !get(f, head.n_tokens) || !get(f, n_ckpt) || !get(f, reserved)) {
        return false;
    }
    head.spec_synced = (flags & 1u) != 0;

    uint64_t size_tgt = 0;
    if (!get(f, size_tgt) || size_tgt == 0 || size_tgt > f.remaining() || !pass_part(f, get_tgt, size_tgt)) {
        return false;
    }
    size_dft_out = 0;
    if (!get(f, size_dft_out) || size_dft_out > f.remaining() ||
        (size_dft_out > 0 && !pass_part(f, get_dft, size_dft_out))) {
        return false;
    }
    if (!get_blob(f, data_spec)) {
        return false;
    }

    ckpts.clear();
    for (uint32_t i = 0; i < n_ckpt; i++) {
        common_prompt_checkpoint c;
        int64_t  n_tokens = 0;
        int32_t  id_task  = -1;
        uint32_t pinned   = 0;
        int32_t  pos_min  = 0;
        int32_t  pos_max  = 0;
        if (!get(f, n_tokens) || !get(f, id_task) || !get(f, pinned) || !get(f, pos_min) || !get(f, pos_max) ||
            !get_blob(f, c.data_tgt) || !get_blob(f, c.data_dft) || !get_blob(f, c.data_spec)) {
            return false;
        }
        c.n_tokens = n_tokens;
        c.id_task  = id_task;
        c.pinned   = pinned != 0;
        c.pos_min  = pos_min;
        c.pos_max  = pos_max;
        ckpts.push_back(std::move(c));
    }

    char tail[8] = {};
    return f.read(tail, sizeof(tail)) && std::memcmp(tail, SHELF_TAIL, sizeof(tail)) == 0;
}

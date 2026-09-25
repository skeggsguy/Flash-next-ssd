// The reading room's stats lines, beside the book manager's (llama-moe-stream-stats.cpp). Stats line 1
// (remaps, waves, miss, stall) keeps its exact format: the room's groups count there as waves and its
// waits as stall, so the study's parser and its reading-in windows work unchanged. What only the room
// knows gets a line of its own, in every window that ran room groups:
//
//   moe stream: room floors F groups G | fetched X MiB | unused U% | waited W ms (P%) | parts ahead A | cancelled C MiB
//
// unused = books the runners brought that no word on their floor read (the price of fetching before the
// router has run); waited = desk and part ops waiting for books, as a share of the window; parts ahead =
// on average, how many parts were already READY behind the one a part op needed (0 = the runners were
// only just keeping up); cancelled = belt bytes of parts dropped because their read-in stopped.

#include "llama-moe-room.h"

#include "llama-impl.h"

#include <cinttypes>

void llama_moe_room::dump_stats_locked(int64_t dt_us) {
    const llama_moe_room_stats & s = stats;
    const llama_moe_room_stats & p = stats_prev;

    const int64_t d_groups = s.n_groups - p.n_groups;
    const int64_t d_bytes  = s.n_bytes  - p.n_bytes;
    if (d_groups > 0 || d_bytes > 0) {
        const int64_t d_books = s.n_books      - p.n_books;
        const int64_t d_idle  = s.n_books_idle - p.n_books_idle;
        const int64_t d_wait  = s.t_wait_us    - p.t_wait_us;
        const int64_t d_ops   = s.n_part_ops   - p.n_part_ops;
        LLAMA_LOG_WARN("%s: moe stream: room floors %4" PRId64 " groups %5" PRId64 " | fetched %8.1f MiB"
                       " | unused %5.1f%% | waited %8.1f ms (%5.1f%%) | parts ahead %4.1f | cancelled %6.1f MiB\n",
                "maybe_dump_stats_locked", s.n_floors - p.n_floors, d_groups, d_bytes/1048576.0,
                d_books > 0 ? 100.0*d_idle/d_books : 0.0,
                d_wait/1000.0, dt_us > 0 ? 100.0*d_wait/dt_us : 0.0,
                d_ops > 0 ? (double) (s.n_ahead - p.n_ahead)/d_ops : 0.0,
                (s.n_bytes_cancelled - p.n_bytes_cancelled)/1048576.0);
    }
    stats_prev = stats;
}

void llama_moe_room::print_stats_locked() const {
    LLAMA_LOG_WARN("%s: moe stream: reading room = %" PRId64 " ubatches, %" PRId64 " floors, %" PRId64 " groups, "
                   "%.2f GiB fetched (%.1f%% of books unused), waited %.2f ms, %.2f GiB cancelled\n",
            "print_stats", stats.n_ubatches, stats.n_floors, stats.n_groups, stats.n_bytes/1073741824.0,
            stats.n_books > 0 ? 100.0*stats.n_books_idle/stats.n_books : 0.0, stats.t_wait_us/1000.0,
            stats.n_bytes_cancelled/1073741824.0);
}

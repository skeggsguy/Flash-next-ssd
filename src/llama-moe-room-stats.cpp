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
//
// The lent belt (llama-moe-room-lend-ops.cpp) has a line of its own, in every window where writing missed
// or the room took the belt back. It starts "moe stream: lent belt:", never "room" or "drives", which the
// study's runner matches (runner/roomstats.py):
//
//   moe stream: lent belt: N books held | X of Y misses from the belt (Z%) | copied in I MiB, out O MiB | T ms a book | not kept K | taken back B
//
// held = complete copies on the belt at the end of the window; Y = the writing remap's misses, X = those
// copied back from the belt instead of read (each still a desk miss in stats line 1); copied in = keeps,
// out = copies back; ms a book = copying one book back, slab by slab; not kept = books put back that the
// call's cap or a busy copy kept off the belt; taken back = read-ins that took the belt back.

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

    const llama_moe_lend_stats & l = lstats;
    const llama_moe_lend_stats & q = lstats_prev;
    if (lend_on && (l.n_miss > q.n_miss || l.n_kept > q.n_kept || l.n_taken_back > q.n_taken_back)) {
        const int64_t d_miss = l.n_miss - q.n_miss;
        const int64_t d_lent = l.n_lent - q.n_lent;
        LLAMA_LOG_WARN("%s: moe stream: lent belt: %5zu books held | %5" PRId64 " of %5" PRId64 " misses from the belt (%5.1f%%)"
                       " | copied in %7.1f MiB, out %7.1f MiB | %6.3f ms a book | not kept %4" PRId64 " | taken back %3" PRId64 "\n",
                "maybe_dump_stats_locked", lend.held(), d_lent, d_miss, d_miss > 0 ? 100.0*d_lent/d_miss : 0.0,
                (l.n_bytes_in - q.n_bytes_in)/1048576.0, (l.n_bytes_out - q.n_bytes_out)/1048576.0,
                d_lent > 0 ? (l.t_restore_us - q.t_restore_us)/1000.0/d_lent : 0.0,
                l.n_not_kept - q.n_not_kept, l.n_taken_back - q.n_taken_back);
    }
    lstats_prev = lstats;
}

void llama_moe_room::print_stats_locked() const {
    LLAMA_LOG_WARN("%s: moe stream: reading room = %" PRId64 " ubatches, %" PRId64 " floors, %" PRId64 " groups, "
                   "%.2f GiB fetched (%.1f%% of books unused), waited %.2f ms, %.2f GiB cancelled\n",
            "print_stats", stats.n_ubatches, stats.n_floors, stats.n_groups, stats.n_bytes/1073741824.0,
            stats.n_books > 0 ? 100.0*stats.n_books_idle/stats.n_books : 0.0, stats.t_wait_us/1000.0,
            stats.n_bytes_cancelled/1073741824.0);
    if (lend_on) {
        const llama_moe_lend_stats & l = lstats;
        LLAMA_LOG_WARN("%s: moe stream: lent belt = %" PRId64 " books kept, %" PRId64 " of %" PRId64 " writing misses from "
                       "the belt (%.1f%%), %.2f GiB copied in, %.2f GiB out, %.3f ms a book, %" PRId64 " not kept, "
                       "taken back %" PRId64 " times (%" PRId64 " copies dropped)\n",
                "print_stats", l.n_kept, l.n_lent, l.n_miss, l.n_miss > 0 ? 100.0*l.n_lent/l.n_miss : 0.0,
                l.n_bytes_in/1073741824.0, l.n_bytes_out/1073741824.0, l.n_lent > 0 ? l.t_restore_us/1000.0/l.n_lent : 0.0,
                l.n_not_kept, l.n_taken_back, l.n_cleared);
    }
}

#include "llama-ple-shelf-stats.h"

#include "llama-impl.h"

#include <algorithm>
#include <cinttypes>

int llama_ple_shelf_kind_of(int64_t n_tokens) {
    return n_tokens <= LLAMA_PLE_SHELF_WRITING_MAX ? 0 : 1;
}

static double pct(int64_t part, int64_t whole) {
    return whole > 0 ? 100.0 * (double) part / (double) whole : 0.0;
}

static std::string rates(const llama_ple_shelf_stats & s) {
    const int64_t asks = s.asks[0] + s.asks[1];
    const int64_t hits = s.hits[0] + s.hits[1];
    return format("asks %" PRId64 ", on the shelf %" PRId64 " (%.1f%%), writing (%.1f%%), reading in (%.1f%%)",
            asks, hits, pct(hits, asks), pct(s.hits[0], s.asks[0]), pct(s.hits[1], s.asks[1]));
}

static llama_ple_shelf_stats minus(const llama_ple_shelf_stats & a, const llama_ple_shelf_stats & b) {
    llama_ple_shelf_stats d;
    d.calls     = a.calls     - b.calls;
    d.passed    = a.passed    - b.passed;
    d.bytes     = a.bytes     - b.bytes;
    d.t_wait_us = a.t_wait_us - b.t_wait_us;
    for (int k = 0; k < 2; ++k) {
        d.asks[k]  = a.asks[k]  - b.asks[k];
        d.hits[k]  = a.hits[k]  - b.hits[k];
        d.reads[k] = a.reads[k] - b.reads[k];
    }
    return d;
}

static std::string body(const llama_ple_shelf_stats & d) {
    const int64_t asks = d.asks[0] + d.asks[1];
    const int64_t hits = d.hits[0] + d.hits[1];
    return format("asks %" PRId64 ", on the shelf %" PRId64 " (%.1f%%) | writing %" PRId64 ", %" PRId64 " (%.1f%%)"
            " | reading in %" PRId64 ", %" PRId64 " (%.1f%%) | read %" PRId64 " rows (file %" PRId64 ", alt %" PRId64 ")"
            ", %.2f MiB, %.1f ms waited",
            asks, hits, pct(hits, asks), d.asks[0], d.hits[0], pct(d.hits[0], d.asks[0]),
            d.asks[1], d.hits[1], pct(d.hits[1], d.asks[1]), d.reads[0] + d.reads[1], d.reads[0], d.reads[1],
            d.bytes / 1048576.0, d.t_wait_us / 1000.0);
}

std::string llama_ple_shelf_window_line(const llama_ple_shelf_stats & now, const llama_ple_shelf_stats & prev,
                                        int64_t held, int64_t capacity) {
    const llama_ple_shelf_stats d = minus(now, prev);
    if (d.asks[0] + d.asks[1] == 0) {
        return "";
    }
    return format("phrasebook: window %s | held %" PRId64 " of %" PRId64 " rows | since start %s",
            body(d).c_str(), held, capacity, rates(now).c_str());
}

std::string llama_ple_shelf_total_line(const llama_ple_shelf_stats & s, int64_t held, int64_t capacity) {
    return format("phrasebook: total %s | held %" PRId64 " of %" PRId64 " rows | %" PRId64 " batches (%" PRId64
            " started ahead), %" PRId64 " rows passed (read, not kept)", body(s).c_str(), held, capacity, s.calls, s.begun,
            s.passed);
}

int64_t llama_ple_shelf_slots(int32_t mib, size_t row_size, int64_t n_rows) {
    if (mib == 0 || row_size == 0 || n_rows <= 0) {
        return 0;
    }
    const int64_t bytes = (int64_t) (mib < 0 ? LLAMA_PLE_SHELF_AUTO_MIB : mib) * 1048576;
    // at least one slot (a shelf that keeps nothing is off, not tiny), at most one slot per row
    return std::min<int64_t>(std::max<int64_t>(1, bytes / (int64_t) row_size), std::min<int64_t>(n_rows, INT32_MAX));
}

std::string llama_ple_shelf_startup_line(int32_t mib, int64_t slots, size_t row_size, int64_t n_rows,
                                         int n_copies, bool nocache, bool locked, const std::string & lock_error) {
    const double size_mib = (double) slots * row_size / 1048576.0;
    std::string s = format("phrasebook shelf: %.1f MiB holds %" PRId64 " of the phrasebook's %" PRId64 " rows (%zu B each)",
            size_mib, slots, n_rows, row_size);
    s += mib < 0
        ? format(" (auto: %d MiB, twice the 64 MiB at which the traced sitting of home and away papers reached its reuse ceiling)", LLAMA_PLE_SHELF_AUTO_MIB)
        : format(" (--ple-shelf %d MiB)", mib);
    s += n_copies > 1 ? "; missing rows are read from both drives' copies" : "; missing rows are read from its one copy";
    s += nocache ? ", past macOS's file cache" : ", through macOS's file cache (no-cache reads were refused)";
    s += locked
        ? "; the shelf is locked in memory, so macOS cannot pack it into the pile"
        : "; the shelf could not be locked in memory (" + lock_error + "), so macOS may pack it into the pile";
    s += " (--ple-shelf 0 reads the phrasebook through the file cache as before)";
    return s;
}

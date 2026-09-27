#include "llama-qsa-picks.h"

#include "llama-impl.h"

#include <algorithm>
#include <cinttypes>

void llama_qsa_pick_count::add(const llama_qsa_pick_count & o) {
    n_rows         += o.n_rows;
    n_rows_future  += o.n_rows_future;
    n_rows_blind   += o.n_rows_blind;
    n_picks        += o.n_picks;
    n_picks_future += o.n_picks_future;
}

void llama_qsa_row_tally::finish(int64_t n_pick, llama_qsa_pick_count & count) const {
    count.n_rows  += 1;
    count.n_picks += n_pick;

    // the 1e9 blocks outrank every other block, so they take the first picks. which of them win
    // when they outnumber the picks is up to their scores (1e9 + score rounds), so the tail may lose
    // its place and the future count is a lower bound
    if (n_future_hi + n_tail_hi >= n_pick) {
        count.n_rows_future += 1;
        if (n_tail_hi == 0) {
            count.n_rows_blind += 1;
        }
    }

    count.n_picks_future += std::max<int64_t>(0, std::min<int64_t>(n_future_hi, n_pick - n_tail_hi));
}

static double pct(int64_t a, int64_t b) {
    return b > 0 ? 100.0*(double) a/(double) b : 0.0;
}

void llama_qsa_pick_log(const llama_qsa_pick_count & c, int64_t n_blocks, int64_t n_pick, bool causal) {
    LLAMA_LOG_WARN("%s: qsa picks: %" PRId64 " rows, %" PRId64 " blocks, %" PRId64 " picked each: "
            "%" PRId64 " rows spent every pick past the token (%" PRId64 " of them see nothing), "
            "%" PRId64 " of %" PRId64 " picks past the token (%.1f%%); %s rule\n", __func__,
            c.n_rows, n_blocks, n_pick, c.n_rows_future, c.n_rows_blind,
            c.n_picks_future, c.n_picks, pct(c.n_picks_future, c.n_picks), causal ? "causal" : "old");
}

void llama_qsa_pick_log_total(const llama_qsa_pick_stats & s, bool causal) {
    if (s.n_ubatch == 0) {
        return;
    }

    LLAMA_LOG_INFO("%s: qsa picks: %" PRIu64 " batches, %" PRId64 " rows: %" PRId64 " spent every pick past the token "
            "(%" PRId64 " see nothing), %" PRId64 " of %" PRId64 " picks past the token (%.1f%%); %s rule\n", __func__,
            s.n_ubatch, s.total.n_rows, s.total.n_rows_future, s.total.n_rows_blind,
            s.total.n_picks_future, s.total.n_picks, pct(s.total.n_picks_future, s.total.n_picks),
            causal ? "causal" : "old");
}

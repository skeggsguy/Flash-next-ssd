#include "llama-moe-stream-split.h"

#include "llama-moe-stream.h"
#include "llama-moe-stream-impl.h"

#include <algorithm>
#include <cstdlib>

int32_t llama_moe_stream_split_env() {
    const char * s = std::getenv("LLAMA_MOE_STREAM_SPLIT");
    if (s == nullptr) {
        return 0;
    }
    return std::clamp<int32_t>(atoi(s), 0, LLAMA_MOE_SPLIT_MAX); // an EMPTY value must mean off, not on
}

int llama_moe_split_slab(int32_t level, size_t file_offs, size_t nb, uint32_t & rr, llama_moe_split_piece out[2]) {
    out[0] = llama_moe_split_piece();
    if (level <= 0) {
        return 1; // today's job: the whole slab, the runner picked at the read
    }
    const int8_t first = (int8_t) (rr++ & 1u);
    out[0].drive = first;
    if (level == 1 || nb < 2) {
        return 1;
    }
    // the cut: the direct-read block boundary nearest below the middle, so neither half reads the other's pages
    const size_t a   = MOE_STREAM_DIRECT_ALIGN;
    const size_t mid = file_offs + nb/2;
    const size_t at  = mid & ~(a - 1);
    const size_t cut = at > file_offs ? at - file_offs : nb/2;
    out[0].len   = (uint32_t) cut;
    out[1].offs  = (uint32_t) cut;
    out[1].len   = (uint32_t) (nb - cut);
    out[1].drive = (int8_t) (1 - first);
    return 2;
}

void llama_moe_stream_queue_demand_locked(llama_moe_stream & mgr, llama_moe_stream_layer & sl, int32_t expert,
        int32_t slot, uint64_t lent, uint64_t save) {
    // a restore copies from RAM: no runner, whole slabs (its pins count slabs, llama-moe-room-lend-ops.cpp); one
    // runner: nothing to split across
    const int32_t level = lent != 0 || mgr.files_alt.empty() ? 0 : mgr.split.level;

    size_t n = 0;
    for (size_t wi = 0; wi < sl.weights.size(); wi++) {
        const llama_moe_stream_weight & wt = sl.weights[wi];
        llama_moe_split_piece p[2];
        const int np = llama_moe_split_slab(level, wt.offs + (size_t) expert*wt.nb_expert, wt.nb_expert, mgr.split.rr, p);
        mgr.split.n_spread += level >= 1 && np == 1;
        mgr.split.n_halved += np == 2;
        for (int k = 0; k < np; k++, n++) {
            llama_moe_stream_work w = { &sl, expert, slot, (int32_t) wi, sl.slot_gen[slot] };
            w.lent       = lent;
            w.save       = save;
            w.piece_offs = p[k].offs;
            w.piece_len  = p[k].len;
            w.drive      = p[k].drive;
            mgr.q_demand.push_back(w);
            mgr.cv_work.notify_one(); // one wakeup per job: the jobs of a book are independent reads
        }
    }
    // the LAST job to land publishes the slot (llama-moe-stream-io.cpp); the count is a uint8_t. Set after the
    // pushes, still under the lock, so no runner has seen a job yet
    GGML_ASSERT(n > 0 && n <= 255);
    sl.slot_pending[slot] = (uint8_t) n;
}

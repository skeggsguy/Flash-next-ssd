#include "llama-moe-stream.h"

#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <vector>

// stable per-wave userdata; grows lazily and records the per-wave expert capacity (set at build)
llama_moe_stream_wave * llama_moe_stream_layer::wave_userdata(int32_t wave, uint32_t capacity) {
    GGML_ASSERT(capacity >= 1 && capacity <= n_slots);
    plan_capacity = capacity;
    while ((size_t) wave >= wave_ud.size()) {
        auto ud = std::make_unique<llama_moe_stream_wave>();
        ud->sl   = this;
        ud->wave = (int32_t) wave_ud.size();
        wave_ud.push_back(std::move(ud));
    }
    return wave_ud[wave].get();
}

bool moe_stream_partition() {
    static const bool v = [] {
        const char * s = std::getenv("LLAMA_MOE_STREAM_PARTITION");
        return s == nullptr || atoi(s) != 0; // on by default; an EMPTY value must mean off, not on
    }();
    return v;
}

// wave 0 of a ubatch: record the distinct touched experts (sl.uniq, first-use order) and split them
// into consecutive groups of plan_capacity, one group per wave (sl.expert_wave[e] = e's wave)
void llama_moe_stream::plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    stats.n_calls++;
    start_workers_locked();
    maybe_dump_stats_locked();

    sl.touched.assign(sl.n_expert, 0);
    sl.uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
        if (!sl.touched[e]) {
            sl.touched[e] = 1;
            sl.uniq.push_back(e);
        }
    }


    GGML_ASSERT(sl.plan_capacity > 0);
    sl.expert_wave.assign(sl.n_expert, 0xff);
    for (size_t i = 0; i < sl.uniq.size(); i++) {
        GGML_ASSERT(i/sl.plan_capacity < 0xff);
        sl.expert_wave[sl.uniq[i]] = (uint8_t) (i/sl.plan_capacity);
    }
    sl.plan_n_waves   = (uint32_t) ((sl.uniq.size() + sl.plan_capacity - 1)/sl.plan_capacity);
    sl.plan_next_wave = 0;

    // Wave slices. The masked path packs uniq into full groups of plan_capacity, which is what it has
    // always done. The partition path CANNOT: the graph sized its pair chunk from the wave count it
    // built, so the planner has to produce exactly that many waves. A ubatch touching few experts
    // otherwise plans fewer, fatter waves than the graph expects, and each then holds more pairs than
    // the chunk allows - a repetitive prompt planned 2 waves against a graph built for 5, putting
    // n_pairs/2 = 1536 pairs into a chunk of 1127. No balancing can fix a wave-count disagreement.
    const size_t n_uniq = sl.uniq.size();
    if (sl.plan_waves_want > 1 && n_uniq >= sl.plan_waves_want) {
        sl.plan_n_waves = sl.plan_waves_want;
    }
    sl.wave_first.assign(sl.plan_n_waves, 0);
    sl.wave_count.assign(sl.plan_n_waves, 0);
    {
        // spread the experts evenly over exactly plan_n_waves slices, never exceeding plan_capacity
        const size_t base = n_uniq/sl.plan_n_waves;
        const size_t rem  = n_uniq%sl.plan_n_waves;
        size_t at = 0;
        for (uint32_t w = 0; w < sl.plan_n_waves; w++) {
            const size_t cnt = std::min<size_t>(base + (w < rem ? 1 : 0), sl.plan_capacity);
            sl.wave_first[w] = (uint32_t) at;
            sl.wave_count[w] = (uint32_t) cnt;
            at += cnt;
        }
        GGML_ASSERT(at == n_uniq); // every touched expert belongs to exactly one wave
        for (uint32_t w = 0; w < sl.plan_n_waves; w++) {
            for (uint32_t i = 0; i < sl.wave_count[w]; i++) {
                sl.expert_wave[sl.uniq[sl.wave_first[w] + i]] = (uint8_t) w;
            }
        }
    }

    // keyed off the chunk the graph set, NOT off the env var: a ubatch too small to partition falls
    // back to the masked path, and planning pairs for it would check against a stale chunk
    if (sl.plan_pair_chunk > 0) {
        plan_pairs_locked(sl, ids, n);
    }
}

// Pair partitioning: give every (token, expert) pair to exactly one wave, so the expert GEMMs cover
// each pair once instead of once per wave. Called after the expert->wave split above, which it
// REORDERS: stage_wave_locked stages uniq[w*cap .. +cap], so keeping the waves as contiguous runs of
// uniq means the staging and preload paths need no changes at all - only the order within uniq moves.
//
// Why reorder: the graph fixes chunk_pairs before the router has run, so an unbalanced split (some
// waves owning far more pairs than others, since routing is skewed) would force chunk_pairs up to the
// worst case and give back the saving. Balancing the pair count across waves bounds it near the mean.
void llama_moe_stream::plan_pairs_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    const uint32_t n_waves = sl.plan_n_waves;
    if (n_waves <= 1) {
        sl.plan_pair.clear();
        return;
    }

    // pairs per expert - the weight each expert contributes to its wave
    sl.pair_count.assign(sl.n_expert, 0);
    for (int64_t i = 0; i < n; i++) {
        sl.pair_count[ids[i]]++;
    }

    // group sizes must match what stage_wave_locked will slice out of uniq: cap for every wave but
    // the last, which takes the remainder
    std::vector<uint32_t> room(sl.wave_count);

    // heaviest expert first into the currently lightest wave that still has room (LPT scheduling):
    // bounds the max wave load near the mean for skewed routing, which is what caps chunk_pairs
    std::vector<int32_t> order(sl.uniq);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return sl.pair_count[a] > sl.pair_count[b];
    });

    // SNAKE order, not greedy-lightest. Each wave must end up with a fixed NUMBER of experts (its
    // staging slice), and that cardinality constraint fights load balance: sending each expert to the
    // lightest wave fills the light waves' expert slots first, after which every remaining expert is
    // forced into whatever wave still has room regardless of its load. With 256 experts over 5 waves
    // of 52 there is almost no spare capacity, so that is forced rather than unlucky - it piled 1667
    // pairs into one wave against a mean of 615, all of them individually small.
    //
    // Sweeping back and forth (rank 0->wave 0, 1->1, .. k-1->k-1, k->k-1, k+1->k-2, ..) pairs each
    // heavy expert with a light one and fills every wave to exactly its room by construction.
    std::vector<std::vector<int32_t>> group(n_waves);
    std::vector<int64_t>              load (n_waves, 0);
    {
        std::vector<uint32_t> left(room);
        std::vector<uint32_t> seq;
        seq.reserve(order.size());

        uint32_t w = 0;
        int      dir = 1;
        while (seq.size() < order.size()) {
            if (left[w] > 0) {
                seq.push_back(w);
                left[w]--;
            }
            if (dir > 0) {
                if (w + 1 < n_waves) { w++; } else { dir = -1; }
            } else {
                if (w > 0) { w--; } else { dir = 1; }
            }
        }
        for (size_t i = 0; i < order.size(); i++) {
            group[seq[i]].push_back(order[i]);
            load [seq[i]] += sl.pair_count[order[i]];
        }
    }

    // Repair pass: swapping a heavy expert out of the worst wave for a lighter one from the best wave
    // preserves both cardinalities, so it can only help. Only runs when a wave is actually over the
    // chunk, which snake ordering already makes rare.
    for (int iter = 0; iter < 64; iter++) {
        uint32_t hi = 0, lo = 0;
        for (uint32_t w = 1; w < n_waves; w++) {
            if (load[w] > load[hi]) { hi = w; }
            if (load[w] < load[lo]) { lo = w; }
        }
        if (load[hi] <= (int64_t) sl.plan_pair_chunk || hi == lo) {
            break;
        }

        // best swap = the one that shrinks the gap most without inverting it
        const int64_t gap = load[hi] - load[lo];
        int64_t best_d = 0;
        size_t  bi = 0, bj = 0;
        for (size_t i = 0; i < group[hi].size(); i++) {
            for (size_t j = 0; j < group[lo].size(); j++) {
                const int64_t d = sl.pair_count[group[hi][i]] - sl.pair_count[group[lo][j]];
                if (d > best_d && 2*d <= gap + best_d) { best_d = d; bi = i; bj = j; }
            }
        }
        if (best_d <= 0) {
            break; // nothing left to trade
        }
        std::swap(group[hi][bi], group[lo][bj]);
        load[hi] -= best_d;
        load[lo] += best_d;
    }

    // rewrite uniq in wave order and record each wave's slice, so the stager needs no change
    sl.uniq.clear();
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.wave_first[w] = (uint32_t) sl.uniq.size();
        sl.wave_count[w] = (uint32_t) group[w].size();
        for (const int32_t e : group[w]) {
            sl.expert_wave[e] = (uint8_t) w;
            sl.uniq.push_back(e);
        }
    }

    // how far the worst wave sits above the mean - what the graph's chunk slack has to cover
    const int64_t mean = (int64_t) n/n_waves;
    for (uint32_t w = 0; w < n_waves; w++) {
        stats.pair_over_max = std::max(stats.pair_over_max, mean > 0 ? (load[w] - mean)*100/mean : 0);
    }

    // CHUNK UTILISATION: the worst wave load as a percentage of the bound that ABORTS when exceeded.
    // This is the number that predicts a crash; imbalance-over-mean does not, because the chunk is
    // floored at n_tokens and so is not a fixed multiple of the mean. 100% means the server died.
    int64_t worst = 0;
    for (uint32_t w = 0; w < n_waves; w++) worst = std::max(worst, load[w]);
    if (sl.plan_pair_chunk > 0) {
        stats.chunk_util_max = std::max(stats.chunk_util_max, worst*100/(int64_t) sl.plan_pair_chunk);
    }

    // flat pair indices (t*n_ids + k) owned by each wave
    sl.plan_pair.assign(n_waves, {});
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.plan_pair[w].reserve((size_t) load[w]);
    }
    for (int64_t i = 0; i < n; i++) {
        sl.plan_pair[sl.expert_wave[ids[i]]].push_back((int32_t) i);
    }

    // ---------------------------------------------------------------------------------------
    // SPLIT PASS: move surplus pairs off any over-full wave, staging that expert in the receiving
    // wave as well.
    //
    // Balancing alone cannot fix this, because all of an expert's pairs go wherever it is staged. A
    // single expert can hold n_tokens pairs (one per token), so even with the chunk floored at
    // n_tokens, that expert PLUS any other in the same wave overflows. Measured: an 800-word
    // repetitive prompt gave "wave 0 holds 1009 pairs but chunk is 1000" - overflowing by the size
    // of the second expert. Repetitive input reaches this trivially, so it is not a corner case.
    //
    // A distribution always exists: total capacity is n_waves*chunk, which exceeds n_pairs by the
    // slack. Only indivisibility stood in the way, and an expert may be staged in more than one wave
    // - it costs a slot there, and a second staging of a resident expert is a cache hit, not I/O.
    // ---------------------------------------------------------------------------------------------
    const size_t chunk = sl.plan_pair_chunk;
    for (uint32_t w = 0; w < n_waves && chunk > 0; w++) {
        while (sl.plan_pair[w].size() > chunk) {
            const size_t surplus = sl.plan_pair[w].size() - chunk;

            // the expert contributing most to this wave is the one worth moving
            std::unordered_map<int32_t, size_t> cnt;
            for (const int32_t idx : sl.plan_pair[w]) cnt[ids[idx]]++;
            int32_t hot = -1; size_t hot_n = 0;
            for (const auto & kv : cnt) if (kv.second > hot_n) { hot = kv.first; hot_n = kv.second; }
            if (hot < 0) break;

            // A receiving wave always needs pair room. It needs a free expert slot only if it does
            // not already stage `hot` - if it does, the pairs land on a slot that wave has already
            // reserved, so the move is free. Preferring those receivers is what the earlier version
            // had backwards: it SKIPPED them, spending a slot where none was needed.
            int32_t dst = -1; size_t room = 0; bool dst_has = false;
            for (uint32_t v = 0; v < n_waves; v++) {
                if (v == w || sl.plan_pair[v].size() >= chunk) continue;

                const bool has = std::find(group[v].begin(), group[v].end(), hot) != group[v].end();
                if (!has && group[v].size() >= sl.plan_capacity) continue;

                // a free move beats a bigger one that costs a slot
                const size_t r = chunk - sl.plan_pair[v].size();
                if (has != dst_has ? has : r > room) { room = r; dst = (int32_t) v; dst_has = has; }
            }
            if (dst < 0) {
                break; // no receiver; the check below reports it rather than truncating silently
            }

            const size_t move = std::min({surplus, room, hot_n});
            std::vector<int32_t> keep; keep.reserve(sl.plan_pair[w].size() - move);
            size_t moved = 0;
            for (const int32_t idx : sl.plan_pair[w]) {
                if (moved < move && ids[idx] == hot) { sl.plan_pair[dst].push_back(idx); moved++; }
                else                                  { keep.push_back(idx); }
            }
            sl.plan_pair[w].swap(keep);
            if (!dst_has) {
                group[dst].push_back(hot);   // stage it in the receiver too
            }
            stats.n_pair_splits++;
            if (moved == 0) break;
        }
    }

    // uniq must reflect the split staging, so rebuild the slices from the (possibly grown) groups.
    // expert_wave is left as the last writer sets it: the partition path keys off plan_pair, not
    // expert_wave, and the masked path never runs when partitioning is active.
    sl.uniq.clear();
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.wave_first[w] = (uint32_t) sl.uniq.size();
        sl.wave_count[w] = (uint32_t) group[w].size();
        for (const int32_t e : group[w]) sl.uniq.push_back(e);
    }

    // still loud if a wave cannot be represented - but this should now be unreachable
    for (uint32_t w = 0; w < n_waves; w++) {
        if (sl.plan_pair[w].size() > sl.plan_pair_chunk) {
            // Report which constraint bound, because the two have different fixes: no pair room
            // means the chunk itself is too small (raise LLAMA_MOE_STREAM_PAIR_SLACK), no slot room
            // means the wave count is too low for the expert count (the cap-1 sizing in
            // llama-graph-moe-stream.cpp did not apply, e.g. the pairs-per-wave floor blocked it).
            size_t free_pairs = 0, free_slots = 0;
            for (uint32_t v = 0; v < n_waves; v++) {
                free_pairs += sl.plan_pair[v].size()   < chunk            ? 1 : 0;
                free_slots += group[v].size() < (size_t) sl.plan_capacity ? 1 : 0;
            }
            GGML_ABORT("MoE expert streaming: wave %u holds %zu pairs but chunk is %u after splitting "
                       "(%u waves, %zu with pair room, %zu with a free expert slot, capacity %u, "
                       "%zu experts staged); report it with the prompt that caused it",
                    w, sl.plan_pair[w].size(), sl.plan_pair_chunk,
                    n_waves, free_pairs, free_slots, sl.plan_capacity, sl.uniq.size());
        }
    }
}

// The fork's expert streaming (--moe-stream) in build_moe_ffn, first half: how many waves a ubatch's
// expert GEMMs run in, the id remap of a ubatch that fits in one wave, and the graph node budget the
// waves need. The GEMMs of a multi-wave ubatch are in llama-graph-moe-waves.cpp. build_moe_ffn calls
// both at the places this code used to sit, so the graph's nodes are created in the same order.

#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-room.h"
#include "llama-moe-stream.h"

#include <algorithm>
#include <cstdlib>

// the wave shape of one ubatch; the two log lines keep build_moe_ffn's name, where they were written
llm_moe_stream_shape llm_moe_stream_graph_shape(const llama_moe_stream_layer * msl,
        int64_t n_expert, int64_t n_expert_used, int64_t n_tokens, bool room_allowed) {
    // A read-in long enough to read nearly every book takes the reading room instead of waves: a mode
    // for the whole floor, so neither the remap nor the waves run beside it (llama-graph-moe-room.cpp).
    // Asked on every streamed floor, so a floor that falls back is recorded as not taking it.
    if (msl && msl->mgr->room && msl->mgr->room->take(msl->il, n_tokens, room_allowed)) {
        static int64_t last_logged_tok = 0;
        if (n_tokens != last_logged_tok) {
            last_logged_tok = n_tokens;
            LLAMA_LOG_WARN("%s: moe stream: n_tokens = %5d -> reading room, %d parts per floor\n",
                    "build_moe_ffn", (int) n_tokens, msl->mgr->room->lay.parts);
        }
        llm_moe_stream_shape shape;
        shape.room = true;
        return shape;
    }

    // a ubatch can touch more distinct experts than the cache holds; the expert GEMMs then run in
    //   waves of at most stream_wave_cap experts, the pairs of the other waves masked to zero and the
    //   wave outputs summed. n_touch_max caps at n_expert, so each expert is loaded at most once per
    //   ubatch regardless of batch size - wave splitting bounds prefill I/O to one sweep of them.
    uint32_t n_stream_waves  = 1;
    uint32_t stream_wave_cap = 0;
    bool     use_partition   = false; // set below; the graph and the runtime planner must agree
    if (msl) {
        // worst-case distinct experts: n_expert_used per token, but never more than n_expert total
        const uint64_t n_touch_max = std::min<uint64_t>((uint64_t) n_expert, (uint64_t) n_tokens*n_expert_used);
        if (n_touch_max > msl->n_slots) {
            // the cache must hold three sets at once: this wave's experts, the next wave's preloaded
            //   experts (so its loads overlap this wave's compute), and n_expert_used parking slots
            //   the masked-out pairs GEMM against (Metal needs a slot at most once per token row).
            //   so cap + cap + n_expert_used = n_slots -> cap = (n_slots - n_expert_used)/2
            stream_wave_cap = msl->n_slots > (uint32_t) n_expert_used ? (msl->n_slots - (uint32_t) n_expert_used)/2 : 0;

            // LLAMA_MOE_STREAM_WAVE_CAP overrides the capacity, i.e. the wave count, WITHOUT touching
            // the preload - the two were conflated in an earlier test and that made its result
            // uninterpretable. Wave count multiplies the expert GEMM (every wave re-runs it over the
            // whole token set and masks the rest away), and prefill is now measured as ~94% GPU, so
            // whether those masked passes are FLOP-expensive or bandwidth-cheap is the open question.
            // Safe at any value because stage_wave_locked bounds its keep-set by the real slack.
            //
            // The override is a MEASUREMENT tool, so it has to win outright: the partition budget
            // below used to recompute stream_wave_cap unconditionally, which silently ran a sweep
            // at a cap other than the one asked for - and only for the smaller ubatches, so it
            // corrupted part of a sweep rather than all of it. cap_forced keeps it honest.
            bool cap_forced = false;
            if (const char * s = getenv("LLAMA_MOE_STREAM_WAVE_CAP")) {
                const uint32_t want = (uint32_t) std::max(1, atoi(s));
                stream_wave_cap = std::clamp(want, (uint32_t) n_expert_used,
                                             msl->n_slots - (uint32_t) n_expert_used);
                cap_forced = true;
            }
            if (stream_wave_cap < (uint32_t) n_expert_used) {
                // a wave must fit at least n_expert_used experts, i.e. n_slots >= 3*n_expert_used
                GGML_ABORT("MoE expert streaming: multi-pass expert GEMMs need an expert cache of at least "
                           "3*n_expert_used slots (have %u, need %u); increase --moe-stream-cache or reduce -ub",
                        msl->n_slots, 3*(uint32_t) n_expert_used);
            }
            n_stream_waves = (uint32_t) ((n_touch_max + stream_wave_cap - 1)/stream_wave_cap); // ceil(n_touch_max/cap)

            // Pair partitioning only: keep enough pairs in a wave for the static chunk bound to hold.
            //
            // The chunk is sized from the MEAN pairs per wave plus slack, but the imbalance it has to
            // absorb comes from individual hot experts, whose pair counts do not shrink when the
            // ubatch does. So a small ubatch spread over the same number of waves is far harder to
            // balance: measured on one prefill split into 1750/1532/512-token ubatches, the worst
            // imbalance was +26% on the big one and +43% on the 512 tail - and at cap 30 the tail
            // came +50.2% over, aborting between two cap values that both worked.
            //
            // Capping the wave count by the pair budget makes the tail use fewer, fatter waves (the
            // regime that measured safe at +20%) while big ubatches keep the wave count that makes
            // them fast. The floor is empirical: mean 614 pairs per wave held at +20%, mean 308 blew
            // past +43%. Wave count costs nothing here, but too FEW pairs per wave cannot be balanced.
            if (moe_stream_partition()) {
                constexpr uint64_t pairs_per_wave_min = 600;

                const uint64_t n_pairs   = (uint64_t) n_tokens*n_expert_used;
                const uint32_t max_waves = (uint32_t) std::max<uint64_t>(1, n_pairs/pairs_per_wave_min);

                if (cap_forced && n_stream_waves > max_waves) {
                    // the sweep asked for this cap explicitly; say so rather than quietly retuning
                    static bool logged = false;
                    if (!logged) {
                        logged = true;
                        LLAMA_LOG_WARN("%s: moe stream: LLAMA_MOE_STREAM_WAVE_CAP=%u forces %u waves, "
                                       "below the %u pairs/wave floor - the static chunk bound may "
                                       "abort; unset it to let the planner pick\n",
                                "build_moe_ffn", stream_wave_cap, n_stream_waves, (unsigned) pairs_per_wave_min);
                    }
                }

                if (!cap_forced && n_stream_waves > max_waves) {
                    // widen the waves to fit the budget, but never past the residency invariant above
                    const uint32_t cap_max = msl->n_slots > (uint32_t) n_expert_used
                        ? (msl->n_slots - (uint32_t) n_expert_used)/2 : 0;
                    const uint32_t want    = (uint32_t) ((n_touch_max + max_waves - 1)/max_waves);

                    stream_wave_cap = std::clamp(want, (uint32_t) n_expert_used, std::max(cap_max, (uint32_t) n_expert_used));
                    n_stream_waves  = (uint32_t) ((n_touch_max + stream_wave_cap - 1)/stream_wave_cap);
                }

                // The widening above is bounded by cap_max, so it CANNOT always reach the floor: a
                // short prompt still touches every expert, needs the same waves to hold them, and has
                // far fewer pairs to spread over them. A ~132-token prompt gives 5 waves of ~158 pairs
                // and aborted on the static chunk bound (264 held vs 237) - the common case of a short
                // chat turn, not an edge case.
                //
                // So the floor is a PRECONDITION for partitioning, not a best effort. Below it, fall
                // back to the masked path, which is correct at any imbalance. Nothing is lost: the
                // whole point of partitioning is to stop recomputing a large expert GEMM per wave, and
                // at these sizes that GEMM is trivial anyway.
                use_partition = (uint64_t) n_tokens*n_expert_used >= (uint64_t) n_stream_waves*pairs_per_wave_min;

                // Leave one spare expert slot per wave, so the planner's split pass always has a
                // receiver. A split moves a hot expert's surplus pairs into another wave, which
                // needs pair room AND a free expert slot there. Pair room always exists (total
                // capacity n_waves*chunk >= n_pairs), but slot room does not: at cap 52 with 256
                // experts the wave count ceil(256/52) = 5 leaves 5*52 - 256 = 4 spare slots for the
                // whole layer, and exhausting them aborts the server mid-request.
                //
                // Sizing the count from cap-1 instead takes that to ceil(256/51) = 6 waves and 56
                // spare slots. It is near-free: the chunk tracks the mean, so n_waves*chunk stays
                // ~1.5*n_pairs whatever the wave count, as long as the n_tokens floor does not bind.
                // Guarded by the same pairs-per-wave precondition, which the extra wave must still
                // satisfy - and where it cannot, n_pairs is small, so n_uniq is small and the slots
                // were never tight to begin with.
                // LLAMA_MOE_WAVE_SLACK=0 disables the cap-1 sizing, for A/B against its cost
                static const bool wave_slack = [] {
                    const char * e = getenv("LLAMA_MOE_WAVE_SLACK");
                    return !e || atoi(e) != 0;
                }();

                if (wave_slack && use_partition && !cap_forced && stream_wave_cap > 1) {
                    const uint32_t waves_slack =
                        (uint32_t) ((n_touch_max + stream_wave_cap - 2)/(stream_wave_cap - 1));

                    if (waves_slack > n_stream_waves &&
                        n_pairs >= (uint64_t) waves_slack*pairs_per_wave_min) {
                        n_stream_waves = waves_slack;
                    }
                }
            }

            // Wave count multiplies the expert-GEMM work (each wave re-runs build_expert_gemms over
            // the whole token set and masks the other waves away), so it is the first number any
            // prefill investigation needs. WARN, not INFO: llama-server filters library INFO, which
            // is why upstream's own "slots per layer" line never appears in a server log.
            static uint32_t last_logged_waves = 0;
            static int64_t  last_logged_tok   = 0;
            if (n_stream_waves != last_logged_waves || n_tokens != last_logged_tok) {
                last_logged_waves = n_stream_waves;
                last_logged_tok   = n_tokens;
                LLAMA_LOG_WARN("%s: moe stream: n_tokens = %5d -> wave cap = %u of %u slots, "
                               "%u waves, partition %s\n",
                        "build_moe_ffn", (int) n_tokens, stream_wave_cap, msl->n_slots, n_stream_waves,
                        use_partition ? "ON" : "OFF (masked)");
            }
        }
    }

    return { n_stream_waves, stream_wave_cap, use_partition };
}

// the ids the expert GEMMs read when the ubatch fits in one wave: slot ids from the CPU remap (with the
// next floor's lookahead) or from the GPU resolve kernel; the router's ids otherwise
ggml_tensor * llm_graph_context::build_moe_stream_ids(llama_moe_stream_layer * msl,
        const llm_moe_stream_shape & shape, ggml_tensor * cur, ggml_tensor * selected_experts, int il) const {
    const int64_t  n_tokens       = cur->ne[1];
    const uint32_t n_stream_waves = shape.n_waves;

    ggml_tensor * ids_gemm = selected_experts;
    if (msl && n_stream_waves == 1 && !shape.room) {
        // ggml_top_k() is contiguous, while a caller-provided selection can still be a view.
        ggml_tensor * ids_cont = ggml_is_contiguous(selected_experts)
            ? selected_experts : ggml_cont(ctx0, selected_experts);

        // The GPU owns residency: no CPU op, so no graph split. The kernel resolves hits from its
        // device table, evicts by least-recent use for misses and hands the slot list to a servicer
        // over a shared event. Only for small ubatches - prefill keeps the CPU path.

        if (msl->mgr->gpu_slot >= 3 && msl->state && n_tokens <= LLAMA_MOE_GPU_SLOT_MAX_TOKENS) {
            ids_gemm = ggml_moe_slot_resolve(ctx0, ids_cont, msl->state, nullptr,
                    msl->n_expert, msl->n_slots, il, msl->mgr->gpu_slot);
            cb(ids_gemm, "ffn_moe_topk_gpu", il);

        } else {
            // One-layer-ahead prefetch: predict the NEXT layer's routing from THIS layer's router
            // input and hand it to the same custom op, so the lookahead costs no extra graph split.
            // The prediction skips the attention and FFN terms between the layers - a lower bound,
            // but the exact version would need layer L+1's attention output, i.e. attention twice
            // per layer.
            if (msl->la && msl->la->sl_next && msl->la_gate_inp) {
                ggml_tensor * la_logits = ggml_mul_mat(ctx0, msl->la_gate_inp, cur);
                ggml_prec_set_acc(la_logits, GGML_PREC_F32);
                cb(la_logits, "ffn_moe_logits_next", il);

                ids_gemm = ggml_map_custom2(ctx0, ids_cont, la_logits, llama_moe_stream_remap_la, 1, msl->la);
            } else {
                ids_gemm = ggml_map_custom1(ctx0, ids_cont, llama_moe_stream_remap, 1, msl);
            }
            cb(ids_gemm, "ffn_moe_topk_stream", il);

            // GPU slot resolution, verify stage: resolve the same ids from the device table and
            // check the answer against the CPU remap's, which is still what reaches the GEMM.
            // Taking the remap output as src[2] is also what orders the kernel after it.
            if (msl->mgr->gpu_slot == 1 && msl->state) {
                ids_gemm = ggml_moe_slot_resolve(ctx0, ids_cont, msl->state, ids_gemm,
                        msl->n_expert, msl->n_slots, il, 1);
                cb(ids_gemm, "ffn_moe_topk_gpu", il);
            }
        }
    }

    return ids_gemm;
}

// graph nodes the waves may add to a ubatch of n_tokens, on top of llama_context::graph_max_nodes
uint32_t llama_moe_stream_graph_nodes_max(const llama_model & model, uint32_t n_tokens) {
    uint32_t res = 0;

    if (const auto * mstream = model.moe_stream()) {
        // multi-pass streamed prefill adds a bounded number of extra nodes per wave per streamed layer
        const uint32_t n_eu = model.hparams.n_expert_used_max();
        uint32_t cap = mstream->n_slots > n_eu ? (mstream->n_slots - n_eu)/2 : 0;
        cap = std::max<uint32_t>(cap, 1);
        const uint32_t n_touch_max = std::min<uint32_t>(model.hparams.n_expert, n_tokens*n_eu);
        uint32_t n_waves = (n_touch_max + cap - 1)/cap;
        if (mstream->room) {
            n_waves = std::max<uint32_t>(n_waves, 1 + (uint32_t) mstream->room->lay.parts); // desk + parts
        }
        res += 24u*n_waves*(uint32_t) mstream->layers.size();
    }

    return res;
}

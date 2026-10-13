// The trained one-floor guess (LLAMA_MOE_STREAM_GUESSER1, #43): what it is, and the rules, are in
// llama-moe-stream-guesser.h. This file reads the switch and gives each floor's lookahead its next floor's guesser;
// the file is read by llama_moe_stream_guesser_load (llama-moe-stream-guesser.cpp) at floor offset 1, the pick is
// llama_moe_stream_lookahead_pick1 (llama-moe-stream-lookahead.cpp), and the graph's swap is
// llama-graph-moe-stream.cpp's.

#include "llama-moe-stream-guesser.h"

#include "llama-impl.h"
#include "llama-moe-stream.h"

#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

const char * llama_moe_stream_guesser1_env() {
    const char * s = std::getenv("LLAMA_MOE_STREAM_GUESSER1");
    return s != nullptr && *s != '\0' && strcmp(s, "0") != 0 ? s : nullptr;
}

void llama_moe_stream_guesser1_attach(llama_moe_stream & ms, const std::vector<const ggml_tensor *> & routers,
        int64_t n_embd, int64_t n_expert, bool no_alloc) {
    const char * path = llama_moe_stream_guesser1_env();
    // The switch is read by every model the server opens with the library's settings, the apprentice's sidecar too
    // (one streamed floor, so no lookahead at all): that one has no guess to serve and is left untouched, set or not.
    bool any_la = false;
    for (const auto & sl : ms.layers) {
        any_la = any_la || (sl && sl->la && sl->la->sl_next && sl->la_gate_inp);
    }
    if (path == nullptr) {
        if (any_la && !no_alloc) {
            LLAMA_LOG_WARN("%s: moe stream: one-floor guess: borrowed, each floor's next floor's own router "
                    "(LLAMA_MOE_STREAM_GUESSER1 unset)\n", __func__);
        }
        return; // today's one-floor guess
    }
    if (ms.gpu_slot != 0) {
        throw std::runtime_error("LLAMA_MOE_STREAM_GUESSER1: the GPU slot servicer (LLAMA_MOE_STREAM_GPU_SLOT) writes "
                "without the lookahead; unset one of them");
    }
    if (no_alloc) {
        return; // sizing only: no bytes to read
    }
    if (!any_la) {
        LLAMA_LOG_WARN("%s: moe stream: one-floor guess: the trained guesser %s is not loaded here: no floor guesses "
                "ahead (one streamed floor, or LLAMA_MOE_STREAM_LOOKAHEAD=0)\n", __func__, path);
        return;
    }
    const int64_t t0 = ggml_time_us();
    std::string err;
    auto g = llama_moe_stream_guesser_load(path, n_embd, n_expert, (int32_t) routers.size(),
            llama_moe_stream_guesser_buft_of(routers), err, 1);
    if (!g) {
        throw std::runtime_error(format("the trained one-floor guesser %s (LLAMA_MOE_STREAM_GUESSER1): %s", path, err.c_str()));
    }

    int32_t n_on = 0, n_off = 0;
    uint32_t top_k = 0;
    for (const auto & sl : ms.layers) {
        if (!sl || !sl->la || !sl->la->sl_next || !sl->la_gate_inp) {
            continue; // no one-floor guess from this floor
        }
        auto * la = sl->la;
        const int32_t n = la->sl_next->il;
        if (n >= g->first && n <= g->last && g->weight[n] != nullptr) {
            sl->la_guess1   = g->weight[n];
            sl->la_guess1_b = g->bias[n];
            la->guessed1    = true;
            la->bias_src    = nullptr; // the guesser's bias is in its scores; floor n's router bias is not
            top_k           = la->top_k;
            n_on++;
        } else {
            n_off++; // keeps the borrowed guess: the lookahead is on in every rung, never turned off by a file
        }
    }
    if (n_on == 0) {
        throw std::runtime_error(format("the trained one-floor guesser %s serves no floor: its floors %d..%d are none "
                "of the library's one-floor guesses", path, g->first, g->last));
    }
    LLAMA_LOG_WARN("%s: moe stream: one-floor guess: trained, the guesser %s on %d floors (%d with none in it keep "
            "the borrowed guess), %.1f MiB beside the desk in %s, read in %.2f s; fetches the top %u by z, the borrowed "
            "guess's width\n", __func__, path, n_on, n_off, g->nbytes()/1048576.0,
            g->bufs.empty() ? "-" : ggml_backend_buffer_name(g->bufs[0].get()), (ggml_time_us() - t0)/1e6, top_k);
    ms.guesser1 = std::move(g);
}

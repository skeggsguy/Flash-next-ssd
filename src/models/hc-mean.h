#pragma once

// The plain mean over the hyper-connection streams (the copies of a floor's notes that deepseek4 and qwen4exp
// keep side by side): [n_embd, hc, n_tokens] -> [n_embd, n_tokens]. Stream 0 plus each next stream in turn, then
// scaled by 1/hc. It is not the hc mixer (that norms and weighs the streams); it is what a draft trained on a
// floor's averaged notes reads (DFlash, EAGLE3: SGLang's hidden_states.unflatten(-1, (hc, hidden)).mean(-2)).
// Shared by both models' floor-input captures (cparams.embeddings_layer_inp), built only when one is asked for.

#include "ggml.h"

#include <cstdint>

static inline ggml_tensor * llm_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

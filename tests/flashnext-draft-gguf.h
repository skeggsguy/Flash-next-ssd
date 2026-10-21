#pragma once

// A tiny draft GGUF for the --flashnext tests (test-flashnext-draft-type, -dflash, -defaults): only what
// common_flashnext_draft_type reads, general.architecture and, for DSpark, a markov_w1.weight tensor. No model loads it.

#include "ggml.h"
#include "gguf.h"

#include <cstring>
#include <string>

// writes `path` with general.architecture = `arch` (none when empty) and, with `markov`, DSpark's Markov head
inline void flashnext_write_draft_gguf(const std::string & path, const std::string & arch, bool markov = false) {
    gguf_context * gguf = gguf_init_empty();
    if (!arch.empty()) {
        gguf_set_val_str(gguf, "general.architecture", arch.c_str());
    }
    ggml_init_params ip = { /* .mem_size = */ 4096, /* .mem_buffer = */ nullptr, /* .no_alloc = */ false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(t, markov ? "markov_w1.weight" : "fc.weight");
    memset(t->data, 0, ggml_nbytes(t));
    gguf_add_tensor(gguf, t);
    gguf_write_to_file(gguf, path.c_str(), /* only_meta = */ false);
    ggml_free(ctx);
    gguf_free(gguf);
}

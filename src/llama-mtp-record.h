#pragma once

// The apprentice records only while reading in (SHARE-PARTS-PLAN.md phase 4).
//
// The apprentice (the MTP draft head, a context of its own) reads in every prompt token and every
// checked batch after the library does, because its one floor keeps a record (K/V) of every token that
// its guesses attend to later. Those batches ask for no outputs: the guesses come from one-token decodes
// afterwards, and the hidden states they start from come from the library's context, not the draft's.
// Yet the draft ran its whole floor for them: dense attention over the whole context, its own book sweep
// (its MoE), and the head, all thrown away. Record-only builds, for a batch that asks for no outputs, the
// combiner, the attention mixer and the K/V projections, writes K and V into the cache exactly as the
// whole floor would, and stops: no attention, no books, no head (the DFlash precedent,
// src/models/dflash.cpp's KV-injection pass). The records are byte-identical, so every later guess is.
//
// The big reserve: a context sizes its working space by reserving the graph of a full batch. The
// draft's full batch asks for one output (n_outputs_max), so it reserved the whole floor for 1,024
// tokens: the attention mask of every cell for every token (host and GPU copies) and the books' scratch.
// With record-only on, that reserve builds the record-only graph instead, so the working space shrinks
// too; the one-token guess graph keeps its own (small) reserve.
//
// LLAMA_MTP_RECORD_ONLY (read when the context is made, so per context; only an MTP context of an arch
// that builds the record-only graph takes it): unset means LLAMA_MTP_RECORD_ONLY_DEFAULT, "0" (or
// anything that is not a positive number) is today's graph exactly, a positive number is on.

#include "llama-arch.h"

#include <cstdint>

struct ggml_tensor;
struct llm_graph_context;
class llm_graph_input_attn_kv;

// off until the A-record rung has measured pen to paper and writing with it (then on)
#define LLAMA_MTP_RECORD_ONLY_DEFAULT false

// LLAMA_MTP_RECORD_ONLY's value (nullptr: unset) as on/off
bool llama_mtp_record_only_parse(const char * value);

// the archs whose MTP graph builds the record-only path (qwen4exp's graph_mtp)
bool llama_mtp_record_only_supported(llm_arch arch);

// the context's setting: on only for an MTP context of a supported arch whose switch is on. An MTP
// context says which it runs in a startup line; other contexts say nothing
bool llama_mtp_record_only_init(const char * value, bool is_mtp_ctx, llm_arch arch);

// the graph: a batch that asks for no outputs only records its K/V
inline bool llama_mtp_records_only(bool enabled, int64_t n_outputs) {
    return enabled && n_outputs == 0;
}

// the big reserve's outputs: none when record-only is on, so it builds (and sizes for) the
// record-only graph; otherwise the caller's count
inline uint32_t llama_mtp_reserve_outputs(bool enabled, uint32_t n_outputs) {
    return enabled ? 0u : n_outputs;
}

// writes K and V into the cache as llm_graph_context::build_attn(llm_graph_input_attn_kv *, ...) does
// before it attends: the cache's rotations, the same expand order (V, then K, so a rope can fuse into
// its K write), then the two writes. Nothing reads the result
void llama_mtp_record_kv(const llm_graph_context & g, llm_graph_input_attn_kv * inp,
        ggml_tensor * k_cur, ggml_tensor * v_cur, int il);

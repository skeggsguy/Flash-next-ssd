#include "llama-moe-stream.h"

#include "llama-impl.h"

#include <cinttypes>
#include <cstdio>
#include <string>

// --- the borrowing log ------------------------------------------------------------------------
//
// One file per llama_moe_stream instance. A model and its MTP draft head each own one and both
// would write LLAMA_MOE_STREAM_TRACE, so the streamed-layer count is spliced into the name the
// same way dump_route_hotness_locked does it: trace.mstr -> trace.48L.mstr / trace.1L.mstr.
void llama_moe_stream::trace_open_locked(uint32_t n_expert_used_in) {
    if (trace_file != nullptr || trace_path.empty()) {
        return;
    }

    std::string path = trace_path;
    const size_t dot = path.find_last_of('.');
    const std::string suffix = "." + std::to_string(n_streamed_layers()) + "L";
    path = (dot == std::string::npos || path.find('/', dot) != std::string::npos)
         ? path + suffix
         : path.substr(0, dot) + suffix + path.substr(dot);

    trace_file = fopen(path.c_str(), "wb");
    if (trace_file == nullptr) {
        LLAMA_LOG_WARN("%s: moe stream: cannot write trace %s - the borrowing log is off\n",
                __func__, path.c_str());
        trace_path.clear(); // do not retry on every remap
        return;
    }
    // one buffered write per record would be a syscall per token per layer
    setvbuf(trace_file, nullptr, _IOFBF, 1024*1024);

    const uint32_t version = 1;
    const uint32_t n_layer = (uint32_t) layers.size();
    fwrite("MSTR", 1, 4, trace_file);
    fwrite(&version,          sizeof(uint32_t), 1, trace_file);
    fwrite(&n_expert_used_in, sizeof(uint32_t), 1, trace_file);
    fwrite(&n_layer,          sizeof(uint32_t), 1, trace_file);

    LLAMA_LOG_WARN("%s: moe stream: borrowing log -> %s (%u ids per slip, %u floors)\n",
            __func__, path.c_str(), n_expert_used_in, n_layer);
}

// Append one call's selected ids. `n` must be n_tokens*n_expert_used: the graph's warmup pass runs
// with n_expert_used = n_expert (llama-graph.cpp, llm_graph_context's initialiser list), which is a
// 512-wide sweep of every expert and not routing at all. Those calls are skipped rather than
// recorded, so every record in the file has the one stride the header names and no warmup traffic
// reaches the calibration. (kind 2 is reserved for a future warmup-tagged record.)
void llama_moe_stream::trace_record_locked(int32_t il, uint32_t n_tokens, uint8_t kind, const int32_t * ids, int64_t n) {
    if (trace_path.empty() && trace_file == nullptr) {
        return;
    }
    if (n_expert_used == 0 || il < 0 || il > 0xff) {
        return;
    }
    if (n != (int64_t) n_tokens * (int64_t) n_expert_used) {
        trace_skipped++; // warmup sweep, or a routing width we cannot describe
        return;
    }

    trace_open_locked(n_expert_used);
    if (trace_file == nullptr) {
        return;
    }

    // i16 is enough for every expert count this engine streams (n_expert < 32768 is asserted at
    // registration below); narrowing here halves the file against the graph's i32 ids
    trace_ids.resize((size_t) n);
    for (int64_t i = 0; i < n; i++) {
        trace_ids[i] = (int16_t) ids[i];
    }

    const uint8_t  il_u8 = (uint8_t) il;
    fwrite(&il_u8,   sizeof(uint8_t),  1, trace_file);
    fwrite(&n_tokens, sizeof(uint32_t), 1, trace_file);
    fwrite(&kind,    sizeof(uint8_t),  1, trace_file);
    fwrite(trace_ids.data(), sizeof(int16_t), (size_t) n, trace_file);

    // flushed periodically so a server killed mid-run still leaves a readable trace, and so the
    // reader's "a truncated tail is a truncated tail" rule has something to be true about
    if (++trace_records >= 4096) {
        trace_flush_locked();
    }
}

void llama_moe_stream::trace_flush_locked() {
    if (trace_file != nullptr) {
        fflush(trace_file);
        trace_records = 0;
    }
}

void llama_moe_stream::trace_close() {
    std::lock_guard<std::mutex> lock(mtx);
    if (trace_file != nullptr) {
        fclose(trace_file);
        trace_file = nullptr;
    }
    if (trace_skipped > 0) {
        LLAMA_LOG_WARN("%s: moe stream: borrowing log skipped %" PRId64 " calls whose routing width "
                       "was not %u (the graph's warmup pass routes to every expert)\n",
                __func__, trace_skipped, n_expert_used);
    }
}

#include "llama-moe-room.h"

#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

// a streamable book weight of one floor: blk.<il>.ffn_{gate,up,down,gate_up}_exps.weight holding every
// book of the floor, the same filter llama_moe_stream_resolve_slots sums the desk's slot bytes over
static int room_floor_of(const std::string & name, const ggml_tensor * t, uint32_t n_expert) {
    static const char * const suffixes[] = {
        ".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight", ".ffn_gate_up_exps.weight",
    };
    int il = -1;
    if (t->ne[2] != (int64_t) n_expert || sscanf(name.c_str(), "blk.%d.", &il) != 1) {
        return -1;
    }
    for (const char * s : suffixes) {
        const size_t n = strlen(s);
        if (name.size() >= n && name.compare(name.size() - n, n, s) == 0) {
            return il;
        }
    }
    return -1;
}

llama_moe_room_layout llama_moe_room_size_model(const llama_model_params & params, llm_arch arch,
        const llama_hparams & hparams, const llama_model_loader & ml, uint32_t n_slots) {
    llama_moe_room_books books;
    books.desk_slots    = n_slots;
    books.n_expert      = hparams.n_expert;
    books.n_expert_used = hparams.n_expert_used_max();

    // the default comes from the arithmetic; the rung may move it (llama-moe-room-size.h says why)
    books.sweep_min_tokens = llama_moe_room_sweep_min_tokens(books.n_expert, books.n_expert_used);
    if (const char * s = std::getenv("LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS")) {
        books.sweep_min_tokens = (uint32_t) std::max(1, atoi(s));
    }

    // one book's bytes on each floor, weight by weight: the desk keeps each weight in its own slot
    // tensor, the belt keeps a book's weights together as one record
    std::map<int, std::vector<size_t>> floors;
    for (const auto & [name, w] : ml.weights_map) {
        const int il = room_floor_of(name, w.tensor, books.n_expert);
        if (il >= 0) {
            floors[il].push_back(ggml_nbytes(w.tensor)/books.n_expert);
        }
    }
    for (const auto & [il, nb] : floors) {
        const uint64_t stride = llama_moe_room_record_stride(nb.data(), nb.size());
        for (const size_t n : nb) {
            books.book_bytes += n;
        }
        books.stride_min = books.stride_min == 0 ? stride : std::min(books.stride_min, stride);
        books.stride_max = std::max(books.stride_max, stride);
    }
    books.budget = params.moe_stream_slots == 0 && params.moe_stream_budget > 0
        ? params.moe_stream_budget : (uint64_t) n_slots*books.book_bytes;

    llama_moe_room_request req;
    req.mode  = params.moe_stream_room_mode;
    req.value = params.moe_stream_room_value;
    req.parts = params.moe_stream_room_parts;

    if (req.mode != LLAMA_MOE_ROOM_OFF) {
        // The room is exact because every row a floor computes is the row a run without streaming computes.
        // Two archs break that: mistral4 rescales each GEMM by its whole input (the F32 amax path), and
        // llama4 weights each book's input before the maths, a shape the room's GEMMs do not take.
        if (arch == LLM_ARCH_MISTRAL4) {
            throw std::runtime_error("--moe-stream-room: mistral4's expert maths scale each row by the whole "
                                     "input, so a floor split between the desk and the belt would not give the "
                                     "same numbers; run it with --moe-stream-room 0");
        }
        if (arch == LLM_ARCH_LLAMA4) {
            throw std::runtime_error("--moe-stream-room: llama4 weights each book's input before the expert maths, "
                                     "which the reading room does not do; run it with --moe-stream-room 0");
        }
    }

    const llama_moe_room_layout lay = llama_moe_room_resolve(req, books);
    if (!lay.error.empty()) {
        throw std::runtime_error(lay.error);
    }

    // WARN, not INFO: llama-server filters library INFO, and this is the line that says what ran
    LLAMA_LOG_WARN("load_tensors: %s\n", llama_moe_room_describe(lay).c_str());
    if (!lay.warning.empty()) {
        LLAMA_LOG_WARN("load_tensors: %s\n", lay.warning.c_str());
    }

    // The release proof rests on Metal's last enqueued command buffer being the last one committed.
    // With more than two command buffers and an abort callback, later buffers may never be enqueued,
    // and a part could be handed back while the GPU still reads it. The server sets no abort callback.
    if (lay.on) {
        const char * ncb = std::getenv("GGML_METAL_NCB");
        if (ncb != nullptr && atoi(ncb) > 2) {
            LLAMA_LOG_WARN("load_tensors: reading room with GGML_METAL_NCB=%s: safe only without an abort "
                           "callback (a part could be handed back while the GPU still reads it); keep it at 1 or 2\n", ncb);
        }
    }

    return lay;
}

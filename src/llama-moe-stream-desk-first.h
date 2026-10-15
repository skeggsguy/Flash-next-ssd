#pragma once

// Desk books first (study #39, LLAMA_MOE_STREAM_DESK_FIRST): the engine's default from the D39 rung (+9.1% chat, +4.9%
// code); "0" (or "") is the old stop that waits for every trip (llama-flashnext.h).
//
// Without it a floor's stop reserves slots for the books not on the desk, sends the runners for them, and waits until
// every one has landed before the GPU does any of the floor's book work. With it the stop does not wait: it writes two
// id planes at once, and the floor's GPU stretch is cut in two at a gate (ggml_moe_gate, ggml.h):
//
//   stop (CPU, graph thread):  read the slip; hits keep their slot, misses get one and their trips go to the runners
//                              (the same code as the remap: llama_moe_stream_classify_locked); write
//                                plane 0, the desk's:     the slot of every pick whose book is on the desk, -1 elsewhere
//                                plane 1, the arrivals':  the slot of every pick whose book is still coming, -1 elsewhere
//                              (a guess still landing counts as an arrival)
//   GPU, first command buffer: the desk's books (plane 0) and the shared expert, sent at once
//   gate (graph thread):       wait until this floor's trips have landed (the keep-awake ping as the stop's wait has),
//                              then hand the runners the lookahead's guesses, as the stop would have after its wait
//   GPU, the rest:             the arrivals' books (plane 1), written into the desk's output (ggml_mul_mat_id_into),
//                              then the weighted sum and the rest of the floor, sent only after the gate returns
//
// Exact by construction: the reading room's pattern (llama-graph-moe-room.cpp). Every pick is on exactly one plane,
// the id kernels skip -1, and each output row is written once by the kernel a run without streaming uses, so the words
// do not depend on which books happened to be on the desk. The slots are final at the stop; only the arrivals' bytes
// land late, and the command-buffer boundary after the gate is what makes them visible to the GPU (gotchas).
//
// Scope: writing and the apprentice's checks (up to LLAMA_MOE_DESK_FIRST_MAX_TOKENS tokens, one wave), the library's
// own context (never the apprentice's), a desk the GPU reads (Metal), and neither the GPU slot servicer nor the state
// log. Anything else builds the graph as before. A floor whose books are all on the desk runs the same graph: the
// arrivals' plane is all -1 and its gate returns at once.

#include "llama-moe-stream-stops.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_backend;
struct ggml_tensor;
struct llama_moe_stream;
struct llama_moe_stream_layer;
struct llama_moe_stream_lookahead;
struct llama_moe_stream_stats;

// the most tokens a batch may carry for its floors to run desk books first (writing, and a check of up to 15 drafts)
static const int64_t LLAMA_MOE_DESK_FIRST_MAX_TOKENS = 16;

// one floor's state between its stop and its gate (graph thread only). The slots the gate waits on are the floor's
// demand_slots, cut down at the stop to its arrivals.
struct llama_moe_stream_desk_first_floor {
    bool armed = false;            // the stop wrote both planes and the gate has not run yet
    std::vector<uint8_t> arriving; // [n_slots] scratch: the stop's slots whose book is not on the desk yet

    // the lookahead's guesses, held for the gate while trips are on their way (with none, they go out at the stop)
    bool                         held = false;
    llama_moe_stream_lookahead * la   = nullptr;
    std::vector<int32_t>         books1, books2;
    bool                         two  = false;

    // the floor-stop log's line for this stop (GGML_METAL_CBLOG), put by the gate with the gate's wait as its wait
    bool                         rec_held = false;
    llama_moe_stream_stops::rec  rec      = {};
};

// LLAMA_MOE_STREAM_DESK_FIRST: on unless set to "" or "0" (llama_flashnext_desk_first)
bool llama_moe_stream_desk_first_env();

// A context of this model runs desk books first: the switch is on, the context is the library's (not the apprentice's,
// mtp), nothing it is off with is on, every desk buffer is a GPU's, and one of the backends is Metal's. Then the gate
// fn is set on that backend (only then: with a fn set, Metal scans every graph for gates) and true is returned, for
// the context's cparams.moe_desk_first, which is what lets its graphs build the gates.
bool llama_moe_stream_desk_first_register(llama_moe_stream & mgr, ggml_backend * const * backends, size_t n_backends,
        bool mtp);

// The floor stop's op (ggml_custom_4d, I32 [k, n_tok, 2]): src[0] the router's ids [k, n_tok]. userdata is the floor.
void llama_moe_stream_desk_first_remap(ggml_tensor * dst, int ith, int nth, void * userdata);
// the same with the lookahead: src[1] the next floors' predicted scores, as llama_moe_stream_remap_la's b. userdata
// is the floor's llama_moe_stream_lookahead
void llama_moe_stream_desk_first_remap_la(ggml_tensor * dst, int ith, int nth, void * userdata);

// A decode that ended before its graph's gates ran (aborted by the abort callback right after a stop, or failed) left
// floors armed, with guesses and a stop line held for gates that will never run: clear them all, so the next decode's
// stops do not trip the armed check (llama_context::graph_compute calls this on any status but success).
void llama_moe_stream_desk_first_reset(llama_moe_stream & mgr);

// The gate fn (ggml_backend_metal_set_moe_gate): gate_userdata is the floor. On the graph thread, after the desk's
// books were sent and before the arrivals' are.
void llama_moe_stream_desk_first_gate(void * user_data, void * gate_userdata);

// the lookahead's part of a floor stop, after the floor's own books (llama-moe-stream-lookahead.cpp): picks the next
// floors' guesses from b and hands them to the runners, or, with hold, keeps them there for the floor's gate
void llama_moe_stream_lookahead_run(llama_moe_stream_lookahead * la, const ggml_tensor * b,
        llama_moe_stream_desk_first_floor * hold);
// the gate's part, after its wait: hands the runners the guesses hold kept (none held: nothing)
void llama_moe_stream_lookahead_send_held(llama_moe_stream_desk_first_floor & df);

// the stats line: the window's (prev = the last dump) or, with prev null, the run's
void llama_moe_stream_desk_first_print(const llama_moe_stream_stats & s, const llama_moe_stream_stats * prev);

#include "llama-moe-room.h"

#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-moe-room-plan.h"
#include "llama-moe-stream.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>
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
        const llama_hparams & hparams, const llama_model_loader & ml, uint32_t n_slots,
        const llama_moe_room_floor_buft & floor_buft) {
    llama_moe_room_request req;
    req.mode  = params.moe_stream_room_mode;
    req.value = params.moe_stream_room_value;
    req.parts = params.moe_stream_room_parts;
    const bool room_wanted = req.mode != LLAMA_MOE_ROOM_OFF;

    llama_moe_room_books books;
    books.desk_slots    = n_slots;
    books.n_expert      = hparams.n_expert;
    books.n_expert_used = hparams.n_expert_used_max();

    // the default is ~20 slips per book, from RR-room; the rung may move it (llama-moe-room-size.h says why)
    books.sweep_min_tokens = llama_moe_room_sweep_min_tokens_env(std::getenv("LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS"),
            books.n_expert, books.n_expert_used);

    // one book's bytes on each floor, weight by weight: the desk keeps each weight in its own slot
    // tensor, the belt keeps a book's weights together as one record
    std::map<int, std::vector<size_t>> floors;
    std::vector<ggml_backend_buffer_type_t> desk_bufts; // where the floors' desks will sit
    for (const auto & [name, w] : ml.weights_map) {
        const int il = room_floor_of(name, w.tensor, books.n_expert);
        if (il >= 0) {
            floors[il].push_back(ggml_nbytes(w.tensor)/books.n_expert);
            const ggml_backend_buffer_type_t buft = room_wanted ? floor_buft(il, w.tensor) : nullptr;
            if (buft != nullptr && std::find(desk_bufts.begin(), desk_bufts.end(), buft) == desk_bufts.end()) {
                desk_bufts.push_back(buft);
            }
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

    // The room is exact because every row a floor computes is the row a run without streaming computes.
    // Two archs break that: mistral4 rescales each GEMM by its whole input (the F32 amax path), and
    // llama4 weights each book's input before the maths, a shape the room's GEMMs do not take. And a
    // floor's belt links write into its desk's tensors, so every desk must sit on one device (a partial
    // -ngl puts the first floors on the CPU); alloc() checks the same again once the desks exist, but by
    // then the desk has been cut for the room, too late for the default room to step back to off.
    if (arch == LLM_ARCH_MISTRAL4) {
        books.refusal = "--moe-stream-room: mistral4's expert maths scale each row by the whole input, so a floor "
                        "split between the desk and the belt would not give the same numbers; run it with "
                        "--moe-stream-room 0";
    } else if (arch == LLM_ARCH_LLAMA4) {
        books.refusal = "--moe-stream-room: llama4 weights each book's input before the expert maths, which the "
                        "reading room does not do; run it with --moe-stream-room 0";
    } else if (desk_bufts.size() > 1) {
        books.refusal = "--moe-stream-room: the desk's floors live on different devices, and the reading room "
                        "needs them on one; run with --moe-stream-room 0";
    }

    const llama_moe_room_layout lay = llama_moe_room_resolve(req, books);
    if (!lay.error.empty()) {
        throw std::runtime_error(lay.error);
    }

    // WARN, not INFO: llama-server filters library INFO, and this is the line that says what ran; any
    // warning comes after it, as the study's runner takes the first "reading room:" line as the room
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

llama_moe_room::llama_moe_room(llama_moe_stream & mgr, const llama_moe_room_layout & lay) : mgr(mgr), lay(lay), belt(lay.room_bytes) {
    floors.resize(mgr.layers.size());
    if (const char * s = std::getenv("LLAMA_MOE_ROOM_POISON")) {
        poison = atoi(s) != 0;
    }
}

void llama_moe_room::alloc(bool no_alloc) {
    // A floor's belt links write into its desk link's tensor, so the belt must sit where the desk does:
    // one buffer type for every streamed floor, and the belt takes it too.
    ggml_backend_buffer_type_t buft = nullptr;
    for (const auto & [b, c] : mgr.ctxs) {
        if (buft != nullptr && b != buft) {
            throw std::runtime_error("--moe-stream-room: the desk's floors live on different devices, and the "
                                     "reading room needs them on one; run with --moe-stream-room 0");
        }
        buft = b;
    }
    if (buft == nullptr) {
        return;
    }

    // the gentle opening: one more step after the desk's chunks, with the same pause before it
    if (!no_alloc && mgr.alloc_pause_ms > 0 && !mgr.bufs.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(mgr.alloc_pause_ms));
    }
    buf = ggml_backend_buft_alloc_buffer(buft, no_alloc ? 0 : lay.room_bytes);
    if (buf == nullptr) {
        throw std::runtime_error(format("--moe-stream-room: unable to allocate the reading room (%.2f GiB) on %s",
                lay.room_bytes/1073741824.0, ggml_backend_buft_name(buft)));
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    mgr.bufs.emplace_back(buf);

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*(mgr.layers.size()*4 + 1),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ctx.reset(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to create the ggml context for the reading room");
    }

    auto place = [&](ggml_tensor * t, size_t offs) {
        if (no_alloc) {
            t->buffer = buf;
        } else if (ggml_backend_tensor_alloc(buf, t, (char *) ggml_backend_buffer_get_base(buf) + offs) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(format("--moe-stream-room: cannot place %s on the belt", t->name));
        }
    };

    whole = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, (int64_t) lay.room_bytes);
    ggml_set_name(whole, "moe_room.belt");
    place(whole, 0);
    host = no_alloc ? nullptr : (uint8_t *) ggml_backend_tensor_get_host_ptr(whole);

    for (size_t il = 0; il < mgr.layers.size(); il++) {
        llama_moe_stream_layer * sl = mgr.layers[il].get();
        if (sl == nullptr) {
            continue;
        }
        llama_moe_room_floor & F = floors[il];
        F.sl = sl;
        std::vector<size_t> nb;
        for (const auto & w : sl->weights) {
            F.w_offs.push_back(llama_moe_room_record_stride(nb.data(), nb.size()));
            nb.push_back(w.nb_expert);
        }
        F.stride    = llama_moe_room_record_stride(nb.data(), nb.size());
        F.n_records = (uint32_t) (lay.room_bytes/F.stride);
        if (F.n_records > LLAMA_MOE_ROOM_RECORDS_MAX) {
            throw std::runtime_error(format("--moe-stream-room: floor %zu's room holds %u books, more than the %u the "
                    "GPU indexes in one step; make the room smaller", il, F.n_records, LLAMA_MOE_ROOM_RECORDS_MAX));
        }

        // each weight's view of the whole belt, typed like its desk slots: record r of the view is the
        // book at belt offset r*stride, so a part's ids are its records' indices. A view that ran past
        // the belt would not fail a check but crash the GPU, so its last byte is checked here by hand.
        for (size_t wi = 0; wi < sl->weights.size(); wi++) {
            const ggml_tensor * cache = sl->weights[wi].cache;
            GGML_ASSERT(ggml_backend_buffer_get_type(cache->buffer) == buft || no_alloc);
            if (F.n_records == 0 ||
                F.w_offs[wi] + (size_t) (F.n_records - 1)*F.stride + sl->weights[wi].nb_expert > lay.room_bytes) {
                throw std::runtime_error(format("--moe-stream-room: floor %zu's books do not fit the belt", il));
            }
            ggml_tensor * t = ggml_new_tensor_3d(ctx.get(), cache->type, cache->ne[0], cache->ne[1], F.n_records);
            t->nb[2] = F.stride;
            t->nb[3] = F.stride*F.n_records;
            ggml_format_name(t, "blk.%zu.room_belt.%zu", il, wi);
            place(t, F.w_offs[wi]);
            F.views.push_back(t);
        }
        for (int32_t g = -1; g < lay.parts; g++) {
            F.ops.push_back(std::make_unique<llama_moe_room_op>(llama_moe_room_op{ sl, g }));
        }
    }

    LLAMA_LOG_INFO("%s: %12s reading room size = %8.2f MiB, %d parts per floor\n", __func__,
            ggml_backend_buffer_name(buf), lay.room_bytes/1048576.0, lay.parts);
}

bool llama_moe_room::take(int32_t il, int64_t n_tokens, bool allowed) {
    llama_moe_room_floor * F = floor(il);
    if (F == nullptr) {
        return false;
    }
    // in_graph records the target's last-built graph, the one its context reuses when the next batch
    // has the same shape. A build that may not take the room - the apprentice's (MTP) context on the
    // same model, run between two of the target's batches - must leave it alone, or the reused graph's
    // desk ops find no floor to begin the ubatch and abort "ran out of order".
    if (!allowed) {
        return false;
    }
    F->in_graph = n_tokens >= (int64_t) lay.sweep_min_tokens;
    return F->in_graph;
}

llama_moe_room_op * llama_moe_room::op_userdata(llama_moe_stream_layer * sl, int32_t group) {
    llama_moe_room_floor * F = floor(sl->il);
    GGML_ASSERT(F != nullptr && group >= -1 && group < lay.parts);
    return F->ops[group + 1].get();
}

void llama_moe_room::begin_ubatch_locked() {
    stats.n_ubatches++;

    // The last read-in is over: the op running this has drained the GPU. Its parts that are READY or in
    // use are handed back; a part still filling is cancelled and its reads still queued are dropped (a
    // read already landing keeps the part's memory until it lands).
    pump_floor = order.size();
    for (const auto & p : belt.parts) {
        if (p.state == LLAMA_MOE_PART_FILLING) {
            stats.n_bytes_cancelled += (int64_t) p.bytes;
        }
    }
    in_use_seq = belt.next_seq - 1;
    release_locked();
    belt.cancel_all();
    mgr.q_room.erase(std::remove_if(mgr.q_room.begin(), mgr.q_room.end(),
            [](const llama_moe_stream_work & w) { return w.slot < 0; }), mgr.q_room.end());

    // a guess still queued has a LOADING desk slot the desk op will wait for: let it land first
    mgr.q_demand.insert(mgr.q_demand.end(), mgr.q_spec.begin(), mgr.q_spec.end());
    mgr.q_spec.clear();

    order.clear();
    std::vector<int32_t> desk;
    std::vector<uint8_t> is_alt;
    for (auto & F : floors) {
        if (F.sl == nullptr || !F.in_graph) {
            continue;
        }
        llama_moe_stream_layer & sl = *F.sl;
        order.push_back(sl.il);

        desk.assign(sl.n_slots, -1);
        for (uint32_t s = 0; s < sl.n_slots; s++) {
            if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_EMPTY) {
                desk[s] = sl.slot_expert[s]; // RESIDENT, or LOADING: on the desk, waited for by the desk op
            }
        }
        is_alt.resize(sl.n_expert);
        for (uint32_t e = 0; e < sl.n_expert; e++) {
            is_alt[e] = mgr.use_alt((int32_t) e, sl.n_expert);
        }
        llama_moe_room_floor_plan plan = llama_moe_room_plan_floor(sl.n_expert, desk, is_alt, lay.parts);
        F.fills = std::move(plan.fills);
        F.parts = std::move(plan.parts);
        F.part_seq.assign(F.parts.size(), 0);
        F.owned = 0;

        for (const auto & [e, s] : F.fills) {
            mgr.reserve_slot_locked(sl, e, s); // EMPTY slots: nothing is evicted
            sl.slot_pending[s] = (uint8_t) sl.weights.size();
        }
        if (mgr.gpu_slot && !F.fills.empty()) {
            sl.publish_state_locked(mgr); // the GPU's copy of the desk must see the new LOADING slots
        }
    }

    pump_floor = 0;
    pump_part  = -1;
    op_floor   = 0;
    op_group   = -1;
    belt.reclaim();
    pump_locked();
}

void llama_moe_room::pump_locked() {
    bool queued = false;
    while (pump_floor < order.size()) {
        llama_moe_room_floor &   F  = floors[order[pump_floor]];
        llama_moe_stream_layer * sl = F.sl;
        const size_t nw = sl->weights.size();

        if (pump_part < 0) {
            for (const auto & [e, s] : F.fills) {
                for (size_t wi = 0; wi < nw; wi++) {
                    mgr.q_room.push_back({ sl, e, s, (int32_t) wi, sl->slot_gen[s] });
                }
            }
            queued   = queued || !F.fills.empty();
            pump_part = 0;
        }
        for (; pump_part < (int32_t) F.parts.size(); pump_part++) {
            const std::vector<int32_t> & books = F.parts[pump_part];
            if (books.empty()) {
                continue;
            }
            llama_moe_part * p = belt.push(sl->il, pump_part, (uint32_t) books.size(), F.stride, (int32_t) (books.size()*nw));
            if (p == nullptr) {
                break; // the belt is full: the next hand-back pumps again
            }
            F.part_seq[pump_part] = p->seq;
            for (size_t j = 0; j < books.size(); j++) {
                for (size_t wi = 0; wi < nw; wi++) {
                    llama_moe_stream_work w = { sl, books[j], -1, (int32_t) wi, p->seq };
                    w.ring_offs = p->offs + j*F.stride + F.w_offs[wi];
                    mgr.q_room.push_back(w);
                }
            }
            stats.n_books += (int64_t) books.size();
            queued = true;
        }
        if (pump_part < (int32_t) F.parts.size()) {
            break;
        }
        pump_floor++;
        pump_part = -1;
    }
    if (queued) {
        mgr.cv_work.notify_all();
    }
}

void llama_moe_room::release_locked() {
    // In POISON mode a handed-back part is overwritten at once, so a GEMM still reading it would read
    // 0xFF books and change the logits: the test that the hand-back is never early.
    if (poison) {
        for (const auto & p : belt.parts) {
            if (p.seq <= in_use_seq && (p.state == LLAMA_MOE_PART_READY || p.state == LLAMA_MOE_PART_IN_USE)) {
                if (host != nullptr) {
                    memset(host + p.offs, 0xFF, p.bytes);
                } else {
                    ggml_backend_tensor_memset(whole, 0xFF, p.offs, p.bytes);
                }
            }
        }
    }
    belt.release_through(in_use_seq);
    belt.reclaim();
    pump_locked();
}

bool llama_moe_room::worker_begin_locked(const llama_moe_stream_work & w) {
    return belt.begin_read(w.gen) != nullptr;
}

void llama_moe_room::worker_end_locked(const llama_moe_stream_work & w, bool ok) {
    if (ok) {
        stats.n_bytes += (int64_t) w.sl->weights[w.widx].nb_expert;
    }
    belt.end_read(w.gen, ok); // READY on the part's last read; never on a failed one
    belt.reclaim();           // a cancelled part whose last read landed
    pump_locked();
    cv.notify_all();
}

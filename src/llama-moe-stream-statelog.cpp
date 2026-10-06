#include "llama-moe-stream-statelog.h"

#include "llama-moe-stream.h"

#include "llama-impl.h"

#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <utility>

// The state log (format on llama_moe_stream_statelog, llama-moe-stream-statelog.h).

const char * llama_moe_stream_state_log_env() {
    const char * s = std::getenv("LLAMA_MOE_STREAM_STATE_LOG");
    return s != nullptr && *s != '\0' ? s : nullptr;
}

llama_moe_stream_statelog::llama_moe_stream_statelog(std::string path, uint32_t n_floor)
    : path(std::move(path)), n_floor(n_floor) {}

llama_moe_stream_statelog::~llama_moe_stream_statelog() {
    std::lock_guard<std::mutex> lock(mtx);
    if (file != nullptr) {
        fclose(file);
        file = nullptr;
        LLAMA_LOG_WARN("%s: moe stream: state log %s: %" PRId64 " records in %" PRId64 " calls, %" PRId64
                       " stops skipped (the warmup sweep, or shapes that do not fit)\n",
                __func__, path.c_str(), n_records, n_calls, n_skipped);
    }
}

size_t llama_moe_stream_statelog::record_bytes(uint32_t n_slip, uint32_t n_cur, uint32_t n_scores, uint32_t n_shexp) {
    return RECORD_HEAD + sizeof(int16_t)*n_slip + sizeof(ggml_fp16_t)*((size_t) n_cur + n_scores + n_shexp);
}

// on the first record, once the widths are known (as the borrowing log opens on its first slip)
bool llama_moe_stream_statelog::open_locked(uint32_t n_slip_in, uint32_t n_cur_in, uint32_t n_scores_in,
        uint32_t n_shexp_in) {
    if (file != nullptr) {
        return true;
    }
    if (failed) {
        return false;
    }
    file = fopen(path.c_str(), "wb");
    if (file == nullptr) {
        failed = true;
        LLAMA_LOG_WARN("%s: moe stream: cannot write the state log %s - it is off\n", __func__, path.c_str());
        return false;
    }
    setvbuf(file, nullptr, _IOFBF, 4*1024*1024); // records are ~11 KB; a syscall each would be 48 a word

    n_slip   = n_slip_in;
    n_cur    = n_cur_in;
    n_scores = n_scores_in;
    n_shexp  = n_shexp_in;
    const uint32_t head[9] = {
        VERSION, HEADER_BYTES, (uint32_t) record_bytes(n_slip, n_cur, n_scores, n_shexp),
        n_floor, n_slip, n_cur, n_scores, n_shexp, /* value_type f16 = */ 1,
    };
    static_assert(4 + sizeof(head) == HEADER_BYTES, "the header's size is in the file");
    fwrite("MSSL", 1, 4, file);
    fwrite(head, sizeof(head), 1, file);

    LLAMA_LOG_WARN("%s: moe stream: state log -> %s (%u books a slip, router input %u, %u scores, shared expert "
                   "%u, f16; %zu bytes a word a floor)\n",
            __func__, path.c_str(), n_slip, n_cur, n_scores, n_shexp, record_bytes(n_slip, n_cur, n_scores, n_shexp));
    return true;
}

// a row-contiguous F32 matrix of `width` columns and `rows` rows, or not
static bool statelog_fits(const ggml_tensor * t, int64_t width, int64_t rows) {
    return t != nullptr && t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) &&
           t->ne[0] == width && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1;
}

void llama_moe_stream_statelog::put(int32_t il, uint32_t n_slip_want, const ggml_tensor * ids,
        const ggml_tensor * scores, const ggml_tensor * cur, const ggml_tensor * shexp) {
    std::lock_guard<std::mutex> lock(mtx);
    if (failed) {
        return;
    }
    const int64_t n_rows = ids ? ids->ne[1] : 0;
    const bool ok = ids != nullptr && ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids) &&
            n_slip_want > 0 && ids->ne[0] == (int64_t) n_slip_want && n_rows > 0 && n_rows <= UINT16_MAX + 1 &&
            il >= 0 && il <= 0xff &&
            scores && cur && shexp && statelog_fits(scores, scores->ne[0], n_rows) &&
            statelog_fits(cur, cur->ne[0], n_rows) && statelog_fits(shexp, shexp->ne[0], n_rows) &&
            (file == nullptr || (n_slip_want == n_slip && scores->ne[0] == n_scores && cur->ne[0] == n_cur &&
                                 shexp->ne[0] == n_shexp));
    if (!ok) {
        n_skipped++; // the warmup's slip of every book, or a stop this log cannot describe
        return;
    }
    if (!open_locked(n_slip_want, (uint32_t) cur->ne[0], (uint32_t) scores->ne[0], (uint32_t) shexp->ne[0])) {
        return;
    }

    // floors run upward within a call, so a floor at or below the last one starts the next call
    if (il <= last_il) {
        call++;
        n_calls++;
    }
    last_il = il;

    rec.resize(record_bytes(n_slip, n_cur, n_scores, n_shexp));
    const int32_t * slip = (const int32_t *) ids->data;
    for (int64_t r = 0; r < n_rows; r++) {
        uint8_t * p = rec.data();
        const uint32_t call32 = (uint32_t) call;
        const uint32_t ntok   = (uint32_t) n_rows;
        const uint16_t row    = (uint16_t) r;
        memcpy(p, &call32, 4); p += 4;
        memcpy(p, &ntok,   4); p += 4;
        memcpy(p, &row,    2); p += 2;
        *p++ = (uint8_t) il;
        *p++ = 0;
        for (uint32_t k = 0; k < n_slip; k++, p += 2) {
            const int16_t e = (int16_t) slip[r*n_slip + k];
            memcpy(p, &e, 2);
        }
        for (const ggml_tensor * t : { cur, scores, shexp }) {
            const float * src = (const float *) ((const char *) t->data + r*t->nb[1]);
            ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) p, t->ne[0]); // p is 2-aligned: the head is 12 bytes
            p += sizeof(ggml_fp16_t)*t->ne[0];
        }
        fwrite(rec.data(), 1, rec.size(), file);
        n_records++;
    }
    // flushed now and then, so a server killed mid-run leaves a readable file
    if ((since_flush += n_rows) >= 1024) {
        fflush(file);
        since_flush = 0;
    }
}

void llama_moe_stream_remap_log(ggml_tensor * dst, int ith, int nth, void * userdata) {
    auto * sl = (llama_moe_stream_layer *) userdata;
    const ggml_tensor * ids = dst->src[0];

    // the record first: "the router chose this", as in the borrowing log, before the stop waits on any trip
    if (ith == 0 && sl->mgr->state_log) {
        sl->mgr->state_log->put(sl->il, sl->mgr->n_expert_used, ids, dst->src[1], dst->src[2], dst->src[3]);
    }
    if (dst->src[4] != nullptr && sl->la != nullptr) {
        llama_moe_stream_remap_la(dst, ids, dst->src[4], ith, nth, sl->la);
    } else {
        llama_moe_stream_remap(dst, ids, ith, nth, sl);
    }
}

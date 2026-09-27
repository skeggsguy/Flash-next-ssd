#include "llama-ple-trace.h"

#include "llama-graph.h"
#include "llama-impl.h"

#include "ggml.h"

#include <cinttypes>
#include <cstdlib>
#include <cstring>

std::unique_ptr<llama_ple_trace> llama_ple_trace::from_env() {
    const char * path = getenv("LLAMA_PLE_TRACE");
    if (path == nullptr || path[0] == '\0' || strcmp(path, "0") == 0) {
        return nullptr;
    }
    return std::make_unique<llama_ple_trace>(path);
}

llama_ple_trace::llama_ple_trace(std::string path) : fname(std::move(path)) {}

llama_ple_trace::~llama_ple_trace() {
    std::lock_guard<std::mutex> lock(mtx);
    if (file != nullptr) {
        fclose(file);
        file = nullptr;
        LLAMA_LOG_WARN("%s: phrasebook trace: %" PRId64 " records, %" PRId64 " tokens -> %s\n",
                __func__, records, tokens, fname.c_str());
    }
    if (skipped > 0) {
        LLAMA_LOG_WARN("%s: phrasebook trace: %" PRId64 " calls not written (no table described, or rows "
                       "that were not a whole number of tokens)\n", __func__, skipped);
    }
}

void llama_ple_trace::describe(uint32_t n_heads_in, uint32_t n_gram_in, uint64_t n_rows_in, uint32_t row_bytes_in) {
    std::lock_guard<std::mutex> lock(mtx);
    if (n_heads == 0) {
        n_heads   = n_heads_in;
        n_gram    = n_gram_in;
        n_rows    = n_rows_in;
        row_bytes = row_bytes_in;
        return;
    }
    if (n_heads != n_heads_in || n_gram != n_gram_in || n_rows != n_rows_in || row_bytes != row_bytes_in) {
        stop_locked("a second table was described to it");
    }
}

int64_t llama_ple_trace::n_records() const {
    std::lock_guard<std::mutex> lock(mtx);
    return records;
}

int64_t llama_ple_trace::n_skipped() const {
    std::lock_guard<std::mutex> lock(mtx);
    return skipped;
}

int64_t llama_ple_trace::now_us() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
}

void llama_ple_trace::stop_locked(const char * why) {
    if (!failed) {
        LLAMA_LOG_WARN("%s: phrasebook trace %s stopped: %s\n", __func__, fname.c_str(), why);
    }
    failed = true;
    if (file != nullptr) {
        fclose(file); // what was written so far stays readable
        file = nullptr;
    }
}

template <typename T>
static void put(std::vector<uint8_t> & buf, size_t & at, T v) {
    memcpy(buf.data() + at, &v, sizeof(v)); // this engine only builds on little-endian hosts
    at += sizeof(v);
}

bool llama_ple_trace::open_locked() {
    if (file != nullptr) {
        return true;
    }
    if (failed) {
        return false;
    }

    file = fopen(fname.c_str(), "wb");
    if (file == nullptr) {
        stop_locked("cannot open the file");
        return false;
    }
    // records are ~64 B a written token and ~256 KiB a 4,096-token batch
    setvbuf(file, nullptr, _IOFBF, 1024*1024);

    t0 = std::chrono::steady_clock::now();
    const uint64_t t0_unix_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    std::vector<uint8_t> head(header_bytes);
    size_t at = 0;
    memcpy(head.data(), "PLTR", 4);
    at += 4;
    put(head, at, version);
    put(head, at, n_heads);
    put(head, at, n_gram);
    put(head, at, n_rows);
    put(head, at, row_bytes);
    put(head, at, t0_unix_us);
    GGML_ASSERT(at == header_bytes);

    if (fwrite(head.data(), 1, head.size(), file) != head.size()) {
        stop_locked("the header could not be written");
        return false;
    }

    LLAMA_LOG_WARN("%s: phrasebook trace -> %s (%u heads, %" PRIu64 " rows of %u B, n-gram %u; row ids only)\n",
            __func__, fname.c_str(), n_heads, n_rows, row_bytes, n_gram);
    last_flush_us = 0;
    return true;
}

void llama_ple_trace::record(const int32_t * rows, int64_t n, bool warmup) {
    std::lock_guard<std::mutex> lock(mtx);
    if (failed) {
        return;
    }
    if (n_heads == 0 || n <= 0 || n % n_heads != 0) {
        skipped++;
        return;
    }
    if (!open_locked()) {
        return;
    }

    const int64_t  t_us     = now_us();
    const uint32_t n_tokens = (uint32_t) (n / n_heads);
    const uint8_t  kind     = warmup ? kind_warmup : n_tokens == 1 ? kind_token : kind_batch;

    buf.resize(record_bytes + (size_t) n*sizeof(uint32_t));
    size_t at = 0;
    put(buf, at, (uint64_t) t_us);
    put(buf, at, n_tokens);
    put(buf, at, kind);
    for (int64_t i = 0; i < n; i++) {
        put(buf, at, (uint32_t) rows[i]);
    }

    if (fwrite(buf.data(), 1, buf.size(), file) != buf.size()) {
        stop_locked("a record could not be written (disk full?)");
        return;
    }
    records++;
    tokens += n_tokens;

    // flushed often, so a server killed mid-run loses at most its last 64 records (or second of work);
    // a partial last record is what the reader's truncated-tail rule reports
    if (++since_flush >= flush_records || t_us - last_flush_us >= flush_us) {
        flush_locked(t_us);
    }
}

void llama_ple_trace::flush_locked(int64_t now) {
    if (file != nullptr) {
        fflush(file);
    }
    since_flush   = 0;
    last_flush_us = now;
}

void llama_ple_trace_attach(llm_graph_lazy_rows & rows, llama_ple_trace * trace, const ggml_tensor * table,
                            uint32_t n_heads, uint32_t n_gram, bool warmup) {
    rows.trace        = trace;
    rows.trace_warmup = warmup;
    if (trace != nullptr && table != nullptr) {
        trace->describe(n_heads, n_gram, (uint64_t) table->ne[1],
                (uint32_t) ggml_row_size(table->type, table->ne[0]));
    }
}

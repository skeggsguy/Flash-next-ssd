#pragma once

// LLAMA_PLE_TRACE=<path>: the phrasebook trace. Every row of the phrasebook (per_layer_tok_embd, the
// n-gram table qwen4exp reads 16 rows of per token) each batch asks for, in order, so that
// ~/dev/ai/sim/ple_sim.py can size a shelf of rows against a replacement rule before one is built
// (SHARE-PARTS-PLAN.md phase 2). Row ids only: the token ids are never written. The rows are hashes of
// the last few tokens, though, so a trace of a private conversation is private too (sim/README.md).
//
// Little-endian, packed, no alignment padding. A header once, written with the first record:
//     char magic[4] = "PLTR", u32 version = 1, u32 n_heads, u32 n_gram, u64 n_rows, u32 row_bytes,
//     u64 t0_unix_us (the wall clock when the file was opened)
// then one record per ubatch (one llm_graph_lazy_rows::set_rows call):
//     u64 t_us (steady clock, since t0), u32 n_tokens, u8 kind, u32 rows[n_tokens*n_heads]
// kind: 0 one token, 1 a batch of several (reading in, or the apprentice's check batch), 2 the warm-up
// (cparams.warmup, whatever its size). Token t's rows are rows[t*n_heads, (t + 1)*n_heads), heads in
// order. Repeats within a call are kept: the reader drops them, as llama_lazy_reader::gather does.
//
// One writer per context: the qwen4exp memory makes it when LLAMA_PLE_TRACE is set (so one process can
// hold a context with the trace and one without), the graph hands it the rows (llama_ple_trace_attach),
// and the file is opened on the first record, so a context that never reads a row (the apprentice's)
// never creates it. Flushed every 64 records or second, closed with the memory.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct ggml_tensor;
class llm_graph_lazy_rows;

class llama_ple_trace {
public:
    static constexpr uint32_t version       = 1;
    static constexpr size_t   header_bytes  = 36;
    static constexpr size_t   record_bytes  = 13; // before the rows

    static constexpr uint8_t  kind_token    = 0;
    static constexpr uint8_t  kind_batch    = 1;
    static constexpr uint8_t  kind_warmup   = 2;

    static constexpr int64_t  flush_records = 64;
    static constexpr int64_t  flush_us      = 1000000;

    // null unless LLAMA_PLE_TRACE names a file ("" and "0" are off)
    static std::unique_ptr<llama_ple_trace> from_env();

    explicit llama_ple_trace(std::string path);
    ~llama_ple_trace();

    llama_ple_trace(const llama_ple_trace &) = delete;
    llama_ple_trace & operator=(const llama_ple_trace &) = delete;

    // the table the rows index; the first description is the header's, a later one that disagrees
    // stops the trace (one file, one table)
    void describe(uint32_t n_heads, uint32_t n_gram, uint64_t n_rows, uint32_t row_bytes);

    // one set_rows call: n = n_tokens*n_heads row ids
    void record(const int32_t * rows, int64_t n, bool warmup);

    const std::string & path() const { return fname; }

    int64_t n_records() const; // written so far
    int64_t n_skipped() const; // calls that could not be written (no description, or a partial token)

private:
    bool open_locked();
    void flush_locked(int64_t now_us);
    void stop_locked(const char * why);
    int64_t now_us() const;

    mutable std::mutex mtx;

    std::string fname;
    FILE *      file   = nullptr;
    bool        failed = false; // cannot write: stop trying

    uint32_t n_heads   = 0;
    uint32_t n_gram    = 0;
    uint64_t n_rows    = 0;
    uint32_t row_bytes = 0;

    std::chrono::steady_clock::time_point t0;

    int64_t records       = 0;
    int64_t tokens        = 0;
    int64_t skipped       = 0;
    int64_t since_flush   = 0;
    int64_t last_flush_us = 0;

    std::vector<uint8_t> buf; // one record, written with one fwrite
};

// the graph's side (called from the model's build_inp_ple): point rows at the context's trace (null: off),
// say whether this graph is the warm-up's, and describe the table to the trace
void llama_ple_trace_attach(llm_graph_lazy_rows & rows, llama_ple_trace * trace, const ggml_tensor * table,
                            uint32_t n_heads, uint32_t n_gram, bool warmup);

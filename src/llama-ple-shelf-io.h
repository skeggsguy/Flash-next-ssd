#pragma once

// The phrasebook shelf (SHARE-PARTS-PLAN.md phase 5): --ple-shelf <MiB|GiB|auto|0>. A fixed buffer of raw
// phrasebook rows inside the share, filled by reads that bypass macOS's file cache, behind
// llm_graph_lazy_rows (the reader llama_model::lazy_reader hands the graph), so the phrasebook no longer
// grows the pile with 16 KB pages for 90 B rows.
//
// One gather per batch: sort the batch's rows and drop repeats (llama_lazy_reader::gather's pattern), plan
// them on the shelf (llama-ple-shelf.h, the simulator's CLOCK with cold entry), then a pool of reader
// threads started for the call (the gather's pattern again, not the book readers' queue: those are few,
// started lazily, and can wait on the graph thread) reads the missing rows into their slots with plain
// preads on F_NOCACHE handles, alternating the model's own copy of the table and the other drive's
// (--moe-stream-alt-path, two runners), and turns every row into numbers with the dequantizer
// ggml_get_rows uses on the CPU (the same to_float as the direct reader), so the rows are identical by
// construction. begin() starts that in the background at the top of set_inputs, so it overlaps the
// other inputs' filling; wait() (in the phrasebook input's set_input) finishes it.
//
// Thread-safe: a call holds the shelf from its plan to its last conversion, so two contexts of one model
// (the library's and a second one) take turns. A read that fails empties the shelf before the error is
// rethrown, so no slot is left claiming a row it never received.

#include "ggml.h"
#include "llama-ple-shelf-stats.h"

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct llama_file;
struct llama_model_loader;
class llama_ple_shelf;

struct llama_ple_shelf_config {
    std::vector<std::string> paths;        // the table's byte-identical copies: the model's own first
    size_t        offs           = 0;      // file offset of row 0, the same in every copy
    enum ggml_type type          = GGML_TYPE_F32;
    int64_t       row_elems      = 0;
    int64_t       n_rows         = 0;
    int32_t       mib            = 0;      // --ple-shelf: -1 auto, else MiB (0 is never made)
    int           n_readers      = 1;
    int64_t       rows_per_token = 1;      // for the writing / reading-in split of the stats
    int64_t       slots          = 0;      // tests: exactly this many slots (at most n_rows), whatever mib says
};

class llama_ple_shelf_io {
public:
    // throws if a copy cannot be opened or differs in size from the first
    explicit llama_ple_shelf_io(const llama_ple_shelf_config & cfg);
    ~llama_ple_shelf_io();

    llama_ple_shelf_io(const llama_ple_shelf_io &) = delete;
    llama_ple_shelf_io & operator=(const llama_ple_shelf_io &) = delete;

    // fill dst with the n rows (repeats allowed) as F32
    void gather(const int32_t * rows, int64_t n, float * dst);

    // the same in the background; dst must stay valid until wait() (the rows are copied)
    std::future<void> begin(const int32_t * rows, int64_t n, float * dst);
    void wait(std::future<void> & pending);

    const llama_ple_shelf_config & config() const { return cfg; }
    int64_t n_rows()    const { return cfg.n_rows; }
    int64_t row_elems() const { return cfg.row_elems; }
    size_t  row_size()  const { return rsize; }
    int64_t capacity()  const;
    int64_t held()      const;
    int     n_copies()  const { return n_copy; }
    bool    nocache()   const { return all_nocache; }
    bool    locked()    const { return is_locked; }

    llama_ple_shelf_stats stats() const;
    std::string startup_line() const;

private:
    void run(std::vector<int32_t> rows, float * dst);
    void maybe_print_locked();
    llama_ple_shelf_stats stats_locked() const;

    const llama_ple_shelf_config cfg;
    const size_t rsize;
    ggml_to_float_t to_float = nullptr;    // null for F32 rows

    std::vector<std::unique_ptr<llama_file>> files; // reader w reads through files[w], copy w % n_copy
    int  n_copy      = 1;
    bool all_nocache = true;

    uint8_t *   buf       = nullptr;       // capacity * rsize, page aligned
    size_t      buf_bytes = 0;
    bool        is_locked = false;
    std::string lock_error;

    mutable std::mutex mtx;                // the shelf, its buffer's slots and the stats
    std::unique_ptr<llama_ple_shelf> shelf;
    llama_ple_shelf_stats st;
    llama_ple_shelf_stats st_prev;
    std::atomic<int64_t> t_wait_us{0};
    std::atomic<int64_t> n_begun{0};
    int64_t stats_us      = 0;             // window length, 0 = total line only
    int64_t t_last_print  = 0;
};

// --ple-shelf for this table: the shelf, with its startup line. Throws where it cannot be made (a copy
// missing or of another size, no memory): asked for, a shelf that cannot be made refuses to load rather
// than let a run measure the old reads under the new name
std::unique_ptr<llama_ple_shelf_io> llama_ple_shelf_make(const llama_ple_shelf_config & cfg);

// the lazily read table t's config: its shard, and the same shard of the other drive's copy when alt_path
// (--moe-stream-alt-path, the copy's first shard) names one; rows_per_token 0 counts one row a token
llama_ple_shelf_config llama_ple_shelf_table(const llama_model_loader & ml, const ggml_tensor * t,
                                             const char * alt_path, int32_t mib, uint32_t rows_per_token);

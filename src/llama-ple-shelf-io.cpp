#include "llama-ple-shelf-io.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model-loader.h"
#include "llama-moe-stream.h"
#include "llama-ple-shelf.h"

#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

llama_ple_shelf_io::llama_ple_shelf_io(const llama_ple_shelf_config & c) :
    cfg(c),
    rsize(ggml_row_size(c.type, c.row_elems)),
    to_float(c.type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(c.type)->to_float) {
    if (cfg.type != GGML_TYPE_F32 && to_float == nullptr) {
        throw std::runtime_error(format("%s rows have no F32 conversion", ggml_type_name(cfg.type)));
    }
    GGML_ASSERT(!cfg.paths.empty() && cfg.row_elems > 0 && cfg.n_rows > 0 && cfg.n_readers > 0 && cfg.mib != 0);

    // the copies: the model's own first; a copy that is the same file (the phrasebook's shard linked onto
    // the other drive) is dropped, and one of another size is refused
    std::vector<std::string> copies;
    std::vector<std::pair<dev_t, ino_t>> ids;
    off_t size0 = 0;
    for (const auto & path : cfg.paths) {
        struct stat sb {};
        if (stat(path.c_str(), &sb) != 0) {
            throw std::runtime_error(format("cannot stat %s: %s", path.c_str(), strerror(errno)));
        }
        if (copies.empty()) {
            size0 = sb.st_size;
        } else if (sb.st_size != size0) {
            throw std::runtime_error(format("%s is %lld bytes, the model's own copy %lld", path.c_str(),
                    (long long) sb.st_size, (long long) size0));
        }
        if (std::find(ids.begin(), ids.end(), std::make_pair(sb.st_dev, sb.st_ino)) == ids.end()) {
            ids.emplace_back(sb.st_dev, sb.st_ino);
            copies.push_back(path);
        }
    }
    if (cfg.offs + (size_t) cfg.n_rows * rsize > (size_t) size0) {
        throw std::runtime_error(format("%s is too short for %lld rows at offset %zu", copies[0].c_str(),
                (long long) cfg.n_rows, cfg.offs));
    }
    n_copy = (int) copies.size();

#if defined(__APPLE__)
    const bool direct = true;  // F_NOCACHE: no copy in the file cache, and no alignment rule (read_at_nocache)
#else
    const bool direct = false; // O_DIRECT would need aligned reads of whole blocks for a 90 B row
#endif
    files.reserve(cfg.n_readers);
    for (int w = 0; w < cfg.n_readers; ++w) {
        files.emplace_back(std::make_unique<llama_file>(copies[w % n_copy].c_str(), "rb", direct));
        all_nocache = all_nocache && files.back()->has_direct_io();
    }

    const int64_t slots = cfg.slots > 0 ? std::min(cfg.slots, cfg.n_rows) : llama_ple_shelf_slots(cfg.mib, rsize, cfg.n_rows);
    shelf = std::make_unique<llama_ple_shelf>(slots);

    const size_t page = 16384;
    buf_bytes = ((size_t) slots * rsize + page - 1) / page * page;
    void * p = nullptr;
    if (posix_memalign(&p, page, buf_bytes) != 0 || p == nullptr) {
        throw std::runtime_error(format("cannot allocate %zu bytes", buf_bytes));
    }
    buf = (uint8_t *) p;
    if (mlock(buf, buf_bytes) == 0) {
        is_locked = true;
    } else {
        lock_error = strerror(errno);
    }

    // a window line on the book manager's stats clock
    const char * ms = getenv("LLAMA_MOE_STREAM_STATS_MS");
    stats_us = ms ? std::max<int64_t>(0, (int64_t) atoll(ms)) * 1000 : 0;
    t_last_print = now_us();
}

llama_ple_shelf_io::~llama_ple_shelf_io() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        const llama_ple_shelf_stats s = stats_locked();
        if (s.calls > 0) {
            LLAMA_LOG_WARN("%s: %s\n", __func__, llama_ple_shelf_total_line(s, shelf->held(), shelf->capacity()).c_str());
        }
    }
    if (buf) {
        if (is_locked) {
            munlock(buf, buf_bytes);
        }
        free(buf);
    }
}

int64_t llama_ple_shelf_io::capacity() const {
    std::lock_guard<std::mutex> lock(mtx);
    return shelf->capacity();
}

int64_t llama_ple_shelf_io::held() const {
    std::lock_guard<std::mutex> lock(mtx);
    return shelf->held();
}

llama_ple_shelf_stats llama_ple_shelf_io::stats_locked() const {
    llama_ple_shelf_stats s = st;
    s.t_wait_us = t_wait_us.load();
    s.begun     = n_begun.load();
    return s;
}

llama_ple_shelf_stats llama_ple_shelf_io::stats() const {
    std::lock_guard<std::mutex> lock(mtx);
    return stats_locked();
}

std::string llama_ple_shelf_io::startup_line() const {
    return llama_ple_shelf_startup_line(cfg.mib, shelf->capacity(), rsize, cfg.n_rows, n_copy, all_nocache,
            is_locked, lock_error);
}

void llama_ple_shelf_io::gather(const int32_t * rows, int64_t n, float * dst) {
    run(std::vector<int32_t>(rows, rows + n), dst);
}

std::future<void> llama_ple_shelf_io::begin(const int32_t * rows, int64_t n, float * dst) {
    n_begun++;
    return std::async(std::launch::async, [this, r = std::vector<int32_t>(rows, rows + n), dst]() mutable {
        run(std::move(r), dst);
    });
}

void llama_ple_shelf_io::wait(std::future<void> & pending) {
    const int64_t t0 = now_us();
    try {
        pending.get();
    } catch (...) {
        t_wait_us += now_us() - t0;
        throw;
    }
    t_wait_us += now_us() - t0;
}

void llama_ple_shelf_io::run(std::vector<int32_t> rows, float * dst) {
    const int64_t n = (int64_t) rows.size();
    const int64_t relems = cfg.row_elems;

    // the batch's rows sorted, and each distinct row's run of positions
    std::vector<std::pair<int32_t, int32_t>> pairs(n);
    for (int64_t i = 0; i < n; ++i) {
        GGML_ASSERT(rows[i] >= 0 && (int64_t) rows[i] < cfg.n_rows);
        pairs[i] = { rows[i], (int32_t) i };
    }
    std::sort(pairs.begin(), pairs.end());

    std::vector<int32_t> uniq;
    std::vector<int64_t> first;
    for (int64_t i = 0; i < n; ++i) {
        if (i == 0 || pairs[i].first != pairs[i - 1].first) {
            uniq.push_back(pairs[i].first);
            first.push_back(i);
        }
    }
    const int64_t n_u = (int64_t) uniq.size();
    first.push_back(n);

    std::lock_guard<std::mutex> lock(mtx);

    std::vector<llama_ple_shelf_pick> picks(n_u);
    shelf->plan(uniq.data(), n_u, picks.data());

    std::vector<int64_t> to_read, to_convert;
    std::vector<int64_t> scratch_at(n_u, -1);
    int64_t n_pass = 0;
    for (int64_t u = 0; u < n_u; ++u) {
        if (picks[u].kind == LLAMA_PLE_SHELF_HIT) {
            to_convert.push_back(u);
            continue;
        }
        if (picks[u].kind == LLAMA_PLE_SHELF_PASS) {
            scratch_at[u] = n_pass++;
        }
        to_read.push_back(u);
    }
    std::vector<uint8_t> scratch((size_t) n_pass * rsize);

    const int64_t n_read = (int64_t) to_read.size();
    const int64_t n_hit  = (int64_t) to_convert.size();
    // reads are latency, not work: about two a thread; conversions about 4,096 rows a thread
    const int n_w = (int) std::min<int64_t>((int64_t) files.size(),
            std::max<int64_t>(1, std::max((n_read + 1) / 2, (n_u + 4095) / 4096)));
    const size_t rot = (size_t) st.calls; // the first reader's handle turns with each batch, so both copies share the reads

    auto convert = [&](int64_t u, const uint8_t * src) {
        float * out = dst + (size_t) pairs[first[u]].second * relems;
        if (to_float) {
            to_float(src, out, relems);
        } else {
            memcpy(out, src, (size_t) relems * sizeof(float));
        }
        for (int64_t j = first[u] + 1; j < first[u + 1]; ++j) {
            memcpy(dst + (size_t) pairs[j].second * relems, out, (size_t) relems * sizeof(float));
        }
    };

    std::vector<int64_t> reads_by(n_w, 0);
    std::vector<size_t>  file_of(n_w, 0);
    auto work = [&](int w) {
        file_of[w] = (rot + w) % files.size();
        llama_file & f = *files[file_of[w]];
        for (int64_t k = n_read * w / n_w; k < n_read * (w + 1) / n_w; ++k) {
            const int64_t u = to_read[k];
            uint8_t * slot = picks[u].kind == LLAMA_PLE_SHELF_PASS
                ? scratch.data() + (size_t) scratch_at[u] * rsize
                : buf + (size_t) picks[u].slot * rsize;
            f.read_at_nocache(cfg.offs + (size_t) uniq[u] * rsize, slot, rsize);
            reads_by[w]++;
            convert(u, slot);
        }
        for (int64_t k = n_hit * w / n_w; k < n_hit * (w + 1) / n_w; ++k) {
            const int64_t u = to_convert[k];
            convert(u, buf + (size_t) picks[u].slot * rsize);
        }
    };

    std::vector<std::exception_ptr> errs(n_w);
    auto run_one = [&](int w) {
        try {
            work(w);
        } catch (...) {
            errs[w] = std::current_exception();
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(n_w - 1);
    try {
        for (int w = 1; w < n_w; ++w) {
            workers.emplace_back(run_one, w);
        }
    } catch (...) {
        errs[0] = std::current_exception(); // could not start a reader: fail the call, below
    }
    if (!errs[0]) {
        run_one(0);
    }
    for (auto & t : workers) {
        t.join();
    }

    for (const auto & err : errs) {
        if (err) {
            // slots the plan gave out may never have been read: forget every row rather than serve one
            shelf = std::make_unique<llama_ple_shelf>(shelf->capacity());
            std::rethrow_exception(err);
        }
    }

    const int kind = llama_ple_shelf_kind_of(n / std::max<int64_t>(1, cfg.rows_per_token));
    st.calls++;
    st.asks[kind] += n_u;
    st.hits[kind] += n_hit;
    st.passed     += n_pass;
    st.bytes      += n_read * (int64_t) rsize;
    for (int w = 0; w < n_w; ++w) {
        st.reads[file_of[w] % n_copy == 0 ? 0 : 1] += reads_by[w]; // files[i] reads copy i % n_copy
    }
    maybe_print_locked();
}

void llama_ple_shelf_io::maybe_print_locked() {
    if (stats_us <= 0) {
        return;
    }
    const int64_t t = now_us();
    if (t - t_last_print < stats_us) {
        return;
    }
    const llama_ple_shelf_stats s = stats_locked();
    const std::string line = llama_ple_shelf_window_line(s, st_prev, shelf->held(), shelf->capacity());
    if (!line.empty()) {
        LLAMA_LOG_WARN("%s: %s\n", __func__, line.c_str());
    }
    st_prev      = s;
    t_last_print = t;
}

std::unique_ptr<llama_ple_shelf_io> llama_ple_shelf_make(const llama_ple_shelf_config & cfg) {
    auto io = std::make_unique<llama_ple_shelf_io>(cfg);
    LLAMA_LOG_WARN("%s: %s\n", __func__, io->startup_line().c_str());
    return io;
}

llama_ple_shelf_config llama_ple_shelf_table(const llama_model_loader & ml, const ggml_tensor * t,
                                             const char * alt_path, int32_t mib, uint32_t rows_per_token) {
    const auto * w = ml.get_weight(ggml_get_name(t));
    GGML_ASSERT(w != nullptr && ggml_is_matrix(t));

    llama_ple_shelf_config cfg;
    cfg.paths.push_back(ml.files[w->idx]->name());
    if (alt_path != nullptr && *alt_path) {
        cfg.paths.push_back(llama_moe_stream_alt_paths(alt_path, ml.files.size()).at(w->idx));
    }
    cfg.offs           = w->offs;
    cfg.type           = t->type;
    cfg.row_elems      = t->ne[0];
    cfg.n_rows         = t->ne[1];
    cfg.mib            = mib;
    // in-flight reads are queue depth rather than compute, as the direct reader's
    cfg.n_readers      = 2*(int) std::max(1u, std::thread::hardware_concurrency());
    cfg.rows_per_token = std::max<uint32_t>(1, rows_per_token);
    return cfg;
}

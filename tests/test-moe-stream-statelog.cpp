// The state log written behind LLAMA_MOE_STREAM_STATE_LOG (src/llama-moe-stream-statelog.h): does the file on disk
// have the layout its readers are written against, and does the switch do nothing when unset?
//
// The bytes are read back here by hand rather than by anything in llama-moe-stream-statelog.cpp, so this file is a
// second, independent statement of the format (as test-moe-stream-trace.cpp is for the borrowing log). What a real
// floor stop hands the log, and that the words stay byte-identical, is test-moe-stream-statelog-model.cpp's job.

#include "testing.h"

#include "llama.h"

#include "../src/llama-moe-stream.h"
#include "../src/llama-moe-stream-statelog.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

static std::string tmp_path(const std::string & name) {
    const char * d = std::getenv("TMPDIR");
    const std::string dir = d && *d ? std::string(d) : std::string("/tmp/");
    return dir + "/moe-statelog-" + name + "-" + std::to_string(getpid()) + ".mssl";
}

static bool exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f) {
        fclose(f);
    }
    return f != nullptr;
}

// a whole file, and a cursor
struct reader {
    std::vector<uint8_t> bytes;
    size_t at = 0;

    bool load(const std::string & path) {
        FILE * f = fopen(path.c_str(), "rb");
        if (f == nullptr) {
            return false;
        }
        uint8_t buf[4096];
        size_t  n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
            bytes.insert(bytes.end(), buf, buf + n);
        }
        fclose(f);
        return true;
    }
    template <typename T> T get() {
        T v{};
        memcpy(&v, bytes.data() + at, sizeof(T));
        at += sizeof(T);
        return v;
    }
    std::string str(size_t n) {
        std::string s((const char *) bytes.data() + at, n);
        at += n;
        return s;
    }
    float f16() { return ggml_fp16_to_fp32(get<ggml_fp16_t>()); }
};

// tensors on the host, as a floor stop's CPU op sees them
struct tensors {
    ggml_context * ctx = nullptr;
    tensors() {
        ggml_init_params p = { 16*1024*1024, nullptr, false };
        ctx = ggml_init(p);
    }
    ~tensors() { ggml_free(ctx); }

    ggml_tensor * ids(int64_t n_slip, const std::vector<int32_t> & v) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_slip, (int64_t) v.size()/n_slip);
        memcpy(t->data, v.data(), v.size()*sizeof(int32_t));
        return t;
    }
    // [width, rows], value(c, r) = base + c*0.5 + r*100 (every value exact in f16)
    ggml_tensor * f32(int64_t width, int64_t rows, float base) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < width; c++) {
                ((float *) t->data)[r*width + c] = base + 0.5f*(float) c + 100.0f*(float) r;
            }
        }
        return t;
    }
};

int main(int argc, char ** argv) {
    testing t;
    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    t.verbose = verbose && std::string(verbose) == "1";
    if (!t.verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("the_switch", [&](testing & t) {
        unsetenv("LLAMA_MOE_STREAM_STATE_LOG");
        auto off = std::make_unique<llama_moe_stream>(6, 8, 1, false);
        t.assert_true("unset: no log, so the graph builds no logging op", off->state_log == nullptr);

        setenv("LLAMA_MOE_STREAM_STATE_LOG", "", 1);
        auto empty = std::make_unique<llama_moe_stream>(6, 8, 1, false);
        t.assert_true("empty: no log", empty->state_log == nullptr);

        const std::string path = tmp_path("switch");
        setenv("LLAMA_MOE_STREAM_STATE_LOG", path.c_str(), 1);
        auto on = std::make_unique<llama_moe_stream>(6, 8, 1, false);
        unsetenv("LLAMA_MOE_STREAM_STATE_LOG");
        t.assert_true("set: a log", on->state_log != nullptr);
        if (on->state_log) {
            t.assert_equal("at the path asked", path, on->state_log->path);
            t.assert_equal("the manager's floors", 6u, on->state_log->n_floor);
        }
        on.reset();
        t.assert_true("nothing logged, so no file", !exists(path));
    });

    t.test("header_and_records", [&](testing & t) {
        const std::string path = tmp_path("records");
        remove(path.c_str());
        auto log = std::make_unique<llama_moe_stream_statelog>(path, /*n_floor =*/ 48);
        tensors T;

        // a call of two words on floors 0 and 1 (floor 1's router input a strided view), then one word on floor 0
        ggml_tensor * ids2   = T.ids(2, { 5, 7, 63, 0 });
        ggml_tensor * wide   = T.f32(6, 2, -1.0f);
        ggml_tensor * cur_v  = ggml_view_2d(T.ctx, wide, 4, 2, wide->nb[1], 0);
        log->put(0, 2, ids2, T.f32(3, 2, 2.0f), T.f32(4, 2, 1.0f), T.f32(4, 2, -3.0f));
        log->put(1, 2, ids2, T.f32(3, 2, 2.5f), cur_v,             T.f32(4, 2, -4.0f));
        log->put(0, 2, T.ids(2, { 9, 1 }), T.f32(3, 1, 7.0f), T.f32(4, 1, 8.0f), T.f32(4, 1, 9.0f));
        t.assert_equal("records", (int64_t) 5, log->n_records);
        t.assert_equal("calls: floor 0 after floor 1 starts the next", (int64_t) 2, log->n_calls);
        log.reset();

        reader r;
        t.assert_true("written", r.load(path));
        const uint32_t rec = 12 + 2*2 + 2*(4 + 3 + 4);
        t.assert_equal("header then five records", (size_t) (40 + 5*rec), r.bytes.size());
        t.assert_equal(std::string("MSSL"), r.str(4));
        t.assert_equal("version",      1u,   r.get<uint32_t>());
        t.assert_equal("header bytes", 40u,  r.get<uint32_t>());
        t.assert_equal("record bytes", rec,  r.get<uint32_t>());
        t.assert_equal("floors",       48u,  r.get<uint32_t>());
        t.assert_equal("slip",         2u,   r.get<uint32_t>());
        t.assert_equal("router input", 4u,   r.get<uint32_t>());
        t.assert_equal("scores",       3u,   r.get<uint32_t>());
        t.assert_equal("shared expert", 4u,  r.get<uint32_t>());
        t.assert_equal("f16",          1u,   r.get<uint32_t>());

        struct want { uint32_t call, ntok; uint16_t row; uint8_t floor; int16_t s0, s1; float sc, cur, sh; float cur_step; };
        const want w[5] = {
            { 0, 2, 0, 0,  5,  7, 2.0f,   1.0f,  -3.0f, 0.5f },
            { 0, 2, 1, 0, 63,  0, 102.0f, 101.0f, 97.0f, 0.5f },
            { 0, 2, 0, 1,  5,  7, 2.5f,  -1.0f,  -4.0f, 0.5f },
            { 0, 2, 1, 1, 63,  0, 102.5f, 99.0f, 96.0f, 0.5f }, // the view's second row is wide's second row
            { 1, 1, 0, 0,  9,  1, 7.0f,   8.0f,   9.0f, 0.5f },
        };
        for (const want & x : w) {
            const size_t start = r.at;
            t.assert_equal("call",     x.call,  r.get<uint32_t>());
            t.assert_equal("n_tokens", x.ntok,  r.get<uint32_t>());
            t.assert_equal("row",      x.row,   r.get<uint16_t>());
            t.assert_equal("floor",    x.floor, r.get<uint8_t>());
            t.assert_equal("zero",     (uint8_t) 0, r.get<uint8_t>());
            t.assert_equal("slip 0",   x.s0,    r.get<int16_t>());
            t.assert_equal("slip 1",   x.s1,    r.get<int16_t>());
            for (int c = 0; c < 4; c++) {
                t.assert_equal("router input", x.cur + 0.5f*c, r.f16());
            }
            for (int c = 0; c < 3; c++) {
                t.assert_equal("score", x.sc + 0.5f*c, r.f16());
            }
            for (int c = 0; c < 4; c++) {
                t.assert_equal("shared expert", x.sh + 0.5f*c, r.f16());
            }
            t.assert_equal("one record's bytes", (size_t) rec, r.at - start);
        }
        t.assert_equal("nothing after the last record", r.bytes.size(), r.at);
        remove(path.c_str());
    });

    t.test("warmup_and_misfits_are_skipped", [&](testing & t) {
        const std::string path = tmp_path("skip");
        remove(path.c_str());
        auto log = std::make_unique<llama_moe_stream_statelog>(path, 4);
        tensors T;

        // the graph's warmup pass routes every word to every book: 4 ids a word where the slip is 2
        log->put(0, 2, T.ids(4, { 0, 1, 2, 3 }), T.f32(4, 1, 0), T.f32(3, 1, 0), T.f32(3, 1, 0));
        t.assert_true("only the warmup so far: no file", !exists(path));

        log->put(0, 2, T.ids(2, { 0, 1 }), T.f32(4, 1, 0), T.f32(3, 1, 0), T.f32(3, 1, 0));
        // after the first record the widths are fixed: a router input of another width is skipped
        log->put(1, 2, T.ids(2, { 0, 1 }), T.f32(4, 1, 0), T.f32(5, 1, 0), T.f32(3, 1, 0));
        // rows that disagree with the slip's word count
        log->put(2, 2, T.ids(2, { 0, 1 }), T.f32(4, 2, 0), T.f32(3, 2, 0), T.f32(3, 2, 0));
        // a missing tensor
        log->put(3, 2, T.ids(2, { 0, 1 }), T.f32(4, 1, 0), T.f32(3, 1, 0), nullptr);
        t.assert_equal("one record", (int64_t) 1, log->n_records);
        t.assert_equal("four skipped", (int64_t) 4, log->n_skipped);
        log.reset();

        reader r;
        t.assert_true(r.load(path));
        t.assert_equal("header and one record", (size_t) (40 + 12 + 2*2 + 2*(3 + 4 + 3)), r.bytes.size());
        remove(path.c_str());
    });

    t.test("unwritable_path_is_off", [&](testing & t) {
        const std::string path = tmp_path("nodir") + "/no/such/dir/log.mssl";
        auto log = std::make_unique<llama_moe_stream_statelog>(path, 4);
        tensors T;
        log->put(0, 2, T.ids(2, { 0, 1 }), T.f32(4, 1, 0), T.f32(3, 1, 0), T.f32(3, 1, 0));
        log->put(1, 2, T.ids(2, { 0, 1 }), T.f32(4, 1, 0), T.f32(3, 1, 0), T.f32(3, 1, 0));
        t.assert_equal("nothing recorded", (int64_t) 0, log->n_records);
        t.assert_true("no file", !exists(path));
    });

    return t.summary();
}

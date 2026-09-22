// The routing trace written behind LLAMA_MOE_STREAM_TRACE: does the file on disk have the layout
// its readers are written against?
//
// The bytes are read back here by hand rather than by anything in llama-moe-stream.cpp, so this
// file is a second, independent statement of the format. If the writer drifts, it drifts away from
// this test as well as from the reader in ~/dev/ai/sim/trace_reader.py.

#include "testing.h"

#include "llama.h"

#include "../src/llama-moe-stream.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define moe_trace_getpid _getpid
#else
#include <unistd.h>
#define moe_trace_getpid getpid
#endif

static std::string tmp_dir() {
    const char * d = std::getenv("TMPDIR");
    return d && *d ? std::string(d) : std::string("/tmp/");
}

// LLAMA_MOE_STREAM_TRACE=<path> is not the path written: the streamed-layer count is spliced in
// before the extension, so a model and its draft head cannot clobber each other.
static std::string suffixed(const std::string & path, int n_streamed) {
    const size_t dot = path.find_last_of('.');
    const std::string suffix = "." + std::to_string(n_streamed) + "L";
    return (dot == std::string::npos || path.find('/', dot) != std::string::npos)
         ? path + suffix
         : path.substr(0, dot) + suffix + path.substr(dot);
}

// a whole file, and a cursor that refuses to read past its end
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

    bool left(size_t n) const { return at + n <= bytes.size(); }

    std::string str(size_t n) {
        std::string s((const char *) bytes.data() + at, n);
        at += n;
        return s;
    }

    uint32_t u32() {
        uint32_t v = 0;
        memcpy(&v, bytes.data() + at, sizeof(v));
        at += sizeof(v);
        return v;
    }

    uint8_t u8() { return bytes[at++]; }

    int16_t i16() {
        int16_t v = 0;
        memcpy(&v, bytes.data() + at, sizeof(v));
        at += sizeof(v);
        return v;
    }
};

// a manager with no tensors and no workers: enough to exercise the trace, nothing else
struct fixture {
    std::string path;
    std::unique_ptr<llama_moe_stream> mgr;

    fixture(const std::string & name, uint32_t n_layer, int n_streamed, uint32_t n_expert_used) {
        path = tmp_dir() + "/moe-trace-" + name + "-" + std::to_string(moe_trace_getpid()) + ".mstr";
        remove(suffixed(path, n_streamed).c_str());

        mgr = std::make_unique<llama_moe_stream>(n_layer, /*n_slots =*/ 8, /*n_io_threads =*/ 1,
                /*direct =*/ false);
        for (int i = 0; i < n_streamed; i++) {
            mgr->layers[i] = std::make_unique<llama_moe_stream_layer>();
            mgr->layers[i]->il = i;
        }
        mgr->n_expert_used = n_expert_used;
        mgr->trace_path    = path;
    }

    // the writer only closes the file at shutdown, so a reader has to wait for that too
    std::string finish(int n_streamed) {
        mgr.reset();
        return suffixed(path, n_streamed);
    }

    ~fixture() { remove(path.c_str()); }
};

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }
    if (!t.verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("header", [&](testing & t) {
        fixture fx("header", /*n_layer =*/ 6, /*n_streamed =*/ 3, /*n_expert_used =*/ 4);
        const std::vector<int32_t> ids = { 0, 1, 2, 3 };
        fx.mgr->trace_record_locked(2, /*n_tokens =*/ 1, /*kind =*/ 0, ids.data(), (int64_t) ids.size());

        const std::string written = fx.finish(3);
        reader r;
        t.assert_true("the streamed-layer count is spliced into the name", r.load(written));
        t.assert_equal("a header is 16 bytes", (size_t) 16, r.bytes.size() - (6 + 4*2));

        t.assert_equal(std::string("MSTR"), r.str(4));
        t.assert_equal("version", 1u, r.u32());
        t.assert_equal("ids per slip", 4u, r.u32());
        t.assert_equal("floors", 6u, r.u32());
        remove(written.c_str());
    });

    t.test("records", [&](testing & t) {
        fixture fx("records", /*n_layer =*/ 48, /*n_streamed =*/ 48, /*n_expert_used =*/ 3);

        // one decode token on floor 0, then a three-token ubatch on floor 47
        const std::vector<int32_t> one  = { 511, 0, 7 };
        const std::vector<int32_t> many = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        fx.mgr->trace_record_locked(0,  1, /*kind =*/ 0, one .data(), (int64_t) one .size());
        fx.mgr->trace_record_locked(47, 3, /*kind =*/ 1, many.data(), (int64_t) many.size());

        const std::string written = fx.finish(48);
        reader r;
        t.assert_true(r.load(written));
        r.at = 16; // past the header, which "header" already checked

        t.assert_equal("floor",  (uint8_t)  0, r.u8());
        t.assert_equal("tokens", (uint32_t) 1, r.u32());
        t.assert_equal("kind",   (uint8_t)  0, r.u8());
        for (size_t i = 0; i < one.size(); i++) {
            t.assert_equal("id", (int16_t) one[i], r.i16());
        }

        t.assert_equal("floor",  (uint8_t)  47, r.u8());
        t.assert_equal("tokens", (uint32_t)  3, r.u32());
        t.assert_equal("kind",   (uint8_t)   1, r.u8());
        for (size_t i = 0; i < many.size(); i++) {
            t.assert_equal("id", (int16_t) many[i], r.i16());
        }

        t.assert_equal("nothing after the last record", r.bytes.size(), r.at);
        remove(written.c_str());
    });

    t.test("warmup_is_skipped", [&](testing & t) {
        // the graph's warmup pass routes to EVERY expert, so its calls carry n_expert ids per
        // token rather than n_expert_used. Recording them would put two strides in one file and
        // count a sweep of the whole collection as routing.
        fixture fx("warmup", /*n_layer =*/ 4, /*n_streamed =*/ 4, /*n_expert_used =*/ 2);
        std::vector<int32_t> wide(16);
        for (size_t i = 0; i < wide.size(); i++) {
            wide[i] = (int32_t) i;
        }
        fx.mgr->trace_record_locked(0, /*n_tokens =*/ 1, /*kind =*/ 1, wide.data(), (int64_t) wide.size());

        const std::vector<int32_t> real = { 3, 4 };
        fx.mgr->trace_record_locked(1, /*n_tokens =*/ 1, /*kind =*/ 0, real.data(), (int64_t) real.size());

        const std::string written = fx.finish(4);
        reader r;
        t.assert_true(r.load(written));
        t.assert_equal("only the real slip is in the file", (size_t) (16 + 6 + 2*2), r.bytes.size());
        r.at = 16;
        t.assert_equal("and it is the one on floor 1", (uint8_t) 1, r.u8());
        remove(written.c_str());
    });

    t.test("off_writes_nothing", [&](testing & t) {
        fixture fx("off", /*n_layer =*/ 4, /*n_streamed =*/ 4, /*n_expert_used =*/ 2);
        fx.mgr->trace_path.clear(); // i.e. LLAMA_MOE_STREAM_TRACE unset

        const std::vector<int32_t> ids = { 1, 2 };
        fx.mgr->trace_record_locked(0, 1, 0, ids.data(), (int64_t) ids.size());

        const std::string written = fx.finish(4);
        reader r;
        t.assert_true("no file at all", !r.load(written));
    });

    return t.summary();
}

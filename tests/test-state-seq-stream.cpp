// llama_state_seq_write_stream / llama_state_seq_read_stream (the bookmark shelf's streaming of a seat's state,
// BOOKMARK-SHELF-PLAN step 2) against llama_state_seq_get_data_ext / set_data_ext, on a generated model:
//
//   same bytes:  the streamed bytes are exactly get_data_ext's, and the returned size is get_size_ext's, however the
//                callback's pieces fall (the recurrent rows of a hybrid model and the KV cells both)
//   round trip:  read back into a second context, the state saved again equals the first
//   failures:    a writer or a reader that fails, or a reader that runs out of bytes, returns 0; so does the device flag
//   bounded:     no piece handed to the callback is over 8 MiB

#include "arg.h"
#include "common.h"
#include "llama-cpp.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

static int n_vocab = 0;
static int n_fail  = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++n_fail; } } while (0)

struct sink {
    std::vector<uint8_t> bytes;
    size_t fail_after = (size_t) -1; // the writer fails once this many bytes are in
    size_t biggest    = 0;
};

static bool put_cb(const void * src, size_t size, void * user_data) {
    auto * s = (sink *) user_data;
    s->biggest = std::max(s->biggest, size);
    if (s->bytes.size() + size > s->fail_after) {
        return false;
    }
    const uint8_t * p = (const uint8_t *) src;
    s->bytes.insert(s->bytes.end(), p, p + size);
    return true;
}

struct source {
    const std::vector<uint8_t> * bytes;
    size_t pos  = 0;
    size_t end  = 0;       // bytes beyond this are "not there"
    bool   fail = false;
};

static bool get_cb(void * dst, size_t size, void * user_data) {
    auto * s = (source *) user_data;
    if (s->fail || s->pos + size > s->end) {
        return false;
    }
    memcpy(dst, s->bytes->data() + s->pos, size);
    s->pos += size;
    return true;
}

static llama_context_ptr make_ctx(llama_model * model) {
    auto cparams = llama_context_default_params();
    cparams.n_ctx     = 256;
    cparams.n_batch   = 64;
    cparams.n_ubatch  = 64;
    cparams.n_seq_max = 2;
    return llama_context_ptr(llama_init_from_model(model, cparams));
}

static bool decode(llama_context * ctx, llama_pos n, llama_seq_id seq) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (llama_pos pos = 0; pos < n; ++pos) {
        common_batch_add(batch, (llama_token) ((7*(uint32_t) pos + 3*(uint32_t) seq + 1) % (uint32_t) n_vocab), pos, { seq }, pos == n - 1);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static std::vector<uint8_t> get_ext(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> buf(llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE));
    buf.resize(llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), seq, LLAMA_STATE_SEQ_FLAGS_NONE));
    return buf;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();

    llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params)));
    if (!model) {
        fprintf(stderr, "failed to load %s\n", params.model.path.c_str());
        return 1;
    }
    n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

    auto ctx = make_ctx(model.get());
    if (!ctx || !decode(ctx.get(), 40, 1)) {
        fprintf(stderr, "cannot decode\n");
        return 1;
    }

    const std::vector<uint8_t> want = get_ext(ctx.get(), 1);
    CHECK(want.size() > 64, "the state is %zu bytes", want.size());

    // same bytes
    sink out;
    const size_t n = llama_state_seq_write_stream(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, put_cb, &out);
    CHECK(n == want.size(), "write_stream returned %zu, get_data_ext %zu", n, want.size());
    CHECK(out.bytes == want, "the streamed bytes differ from get_data_ext's");
    CHECK(out.biggest <= 8u*1024u*1024u, "a piece of %zu bytes", out.biggest);

    // round trip into a second context's seq 1 (the cells carry their seq id), by the format's own piece sizes
    auto ctx2 = make_ctx(model.get());
    source in = { &out.bytes, 0, out.bytes.size(), false };
    const size_t nr = llama_state_seq_read_stream(ctx2.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, get_cb, &in);
    CHECK(nr == want.size(), "read_stream returned %zu of %zu", nr, want.size());
    CHECK(in.pos == in.end, "the reader left %zu bytes", in.end - in.pos);
    CHECK(get_ext(ctx2.get(), 1) == want, "the state read back differs");

    // and set_data_ext reads what the stream wrote
    auto ctx3 = make_ctx(model.get());
    CHECK(llama_state_seq_set_data_ext(ctx3.get(), out.bytes.data(), out.bytes.size(), 1, LLAMA_STATE_SEQ_FLAGS_NONE) == want.size(),
          "set_data_ext does not read the stream's bytes");

    // failures
    for (const size_t at : { (size_t) 0, (size_t) 3, want.size() / 2, want.size() - 1 }) {
        sink bad;
        bad.fail_after = at;
        CHECK(llama_state_seq_write_stream(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, put_cb, &bad) == 0,
              "a writer failing after %zu bytes is not reported", at);
    }
    {
        source dead = { &out.bytes, 0, out.bytes.size(), true };
        CHECK(llama_state_seq_read_stream(ctx2.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, get_cb, &dead) == 0, "a failing reader is not reported");
        source cut = { &out.bytes, 0, out.bytes.size() - 1, false };
        CHECK(llama_state_seq_read_stream(ctx2.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, get_cb, &cut) == 0, "a cut state is not reported");
        std::vector<uint8_t> wrong = out.bytes;
        wrong[0] ^= 0xff;
        source magic = { &wrong, 0, wrong.size(), false };
        CHECK(llama_state_seq_read_stream(ctx2.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, get_cb, &magic) == 0, "a wrong magic is not reported");
    }
    {
        sink dev;
        CHECK(llama_state_seq_write_stream(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE, put_cb, &dev) == 0 && dev.bytes.empty(),
              "the device flag is taken");
    }

    char arch[64] = {};
    llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch));
    fprintf(stderr, "%s: %s, %zu bytes, %d failure(s)\n", arch, n_fail == 0 ? "OK" : "FAILED", want.size(), n_fail);
    return n_fail == 0 ? 0 : 1;
}

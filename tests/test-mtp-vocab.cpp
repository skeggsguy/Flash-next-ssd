// The apprentice's word list (LLAMA_MTP_VOCAB, src/llama-mtp-vocab.h), without a model.
//
//   switch   unset, "" and "0" are today's graph; anything else is a list path; qwen4exp alone takes it
//   parse    a list file's text: ids in any order, blank and '#' lines skipped, repeats dropped; a word that is
//            not an id, an id outside the vocabulary or an empty list is refused, naming the line
//   gather   on every CPU and GPU backend: the copied rows are the head's rows, byte for byte, in the head's
//            type, on the head's device; a head without data, ids out of order or past the head are refused
//   logits   on every backend, for the paperback's head type (Q8_0, at its 2,560 width) and the float types:
//            the listed words' scores are byte-identical to the whole head's (memcmp, not a tolerance) and every
//            other word's is -INF, for 1 output (the guess), 3 and 9 (the small-batch mat-vec) and 33 (the
//            matrix kernel); the top word is the same whenever it is listed; one head's list spans two of the
//            gather's staging chunks. Weights and inputs are O(1), and
//            the scores are checked to differ from each other, so a wrong row cannot compare equal

#include "testing.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "../src/llama-mtp-vocab.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct head_case {
    ggml_type type;
    int64_t   n_embd;
    int64_t   n_vocab;
    std::string name() const {
        return std::string(ggml_type_name(type)) + " " + std::to_string(n_embd) + "x" + std::to_string(n_vocab);
    }
};

// the head's bytes, quantized from O(1) floats
std::vector<uint8_t> make_head(const head_case & hc, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(hc.n_embd*hc.n_vocab);
    std::generate(f.begin(), f.end(), [&] { return u(rng); });
    std::vector<uint8_t> bytes(ggml_row_size(hc.type, hc.n_embd)*hc.n_vocab);
    if (hc.type == GGML_TYPE_F32) {
        memcpy(bytes.data(), f.data(), bytes.size());
    } else if (hc.type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) bytes.data(), (int64_t) f.size());
    } else {
        ggml_quantize_chunk(hc.type, f.data(), bytes.data(), 0, hc.n_vocab, hc.n_embd, nullptr);
    }
    return bytes;
}

// a list with the edges, a long run and ~40% of the rest
std::vector<int64_t> make_ids(int64_t n_vocab, std::mt19937 & rng) {
    std::vector<int64_t> ids = { 0, n_vocab - 1 };
    for (int64_t i = 10; i < 300 && i < n_vocab; i++) {
        ids.push_back(i);
    }
    for (int64_t i = 1; i < n_vocab; i++) {
        if (rng() % 5 < 2) {
            ids.push_back(i);
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

// the head as a model holds it: a weights buffer on the device
struct device_head {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor *           t = nullptr;

    device_head(ggml_backend_dev_t dev, const head_case & hc, const std::vector<uint8_t> & bytes) {
        ggml_init_params p = { ggml_tensor_overhead(), nullptr, true };
        ctx.reset(ggml_init(p));
        t = ggml_new_tensor_2d(ctx.get(), hc.type, hc.n_embd, hc.n_vocab);
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_dev_buffer_type(dev)));
        ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
};

void test_switch(testing & t) {
    t.assert_true("unset is today's graph", !llama_mtp_vocab_wanted(nullptr));
    t.assert_true("\"\" is today's graph", !llama_mtp_vocab_wanted(""));
    t.assert_true("\"0\" is today's graph", !llama_mtp_vocab_wanted("0"));
    t.assert_true("a path is a list", llama_mtp_vocab_wanted("/tmp/list-96k.txt"));
    t.assert_true("qwen4exp takes a list", llama_mtp_vocab_supported(LLM_ARCH_QWEN4EXP));
    t.assert_true("qwen35moe does not", !llama_mtp_vocab_supported(LLM_ARCH_QWEN35MOE));
    t.assert_true("deepseek4 does not", !llama_mtp_vocab_supported(LLM_ARCH_DEEPSEEK4));
}

void test_parse(testing & t) {
    std::vector<int64_t> ids;
    std::string err;
    t.assert_true("ascending ids", llama_mtp_vocab_parse("0\n1\n2\n127\n", 128, ids, err));
    t.assert_true("read as written", ids == std::vector<int64_t>({ 0, 1, 2, 127 }));
    t.assert_true("any order, spaces, CRLF, blank and # lines, repeats",
            llama_mtp_vocab_parse("# a list\n\n  5 \r\n3\n5\n\t9\n", 128, ids, err));
    t.assert_true("sorted, repeats dropped", ids == std::vector<int64_t>({ 3, 5, 9 }));
    t.assert_true("no trailing newline", llama_mtp_vocab_parse("7", 128, ids, err) && ids == std::vector<int64_t>({ 7 }));

    t.assert_true("an id at n_vocab is refused", !llama_mtp_vocab_parse("1\n128\n", 128, ids, err));
    t.assert_true("naming its line", err.find("line 2") != std::string::npos);
    t.assert_true("a negative id is refused", !llama_mtp_vocab_parse("-1\n", 128, ids, err));
    t.assert_true("a word is refused", !llama_mtp_vocab_parse("1\nabc\n", 128, ids, err));
    t.assert_true("naming it", err.find("abc") != std::string::npos && err.find("line 2") != std::string::npos);
    t.assert_true("a trailing word is refused", !llama_mtp_vocab_parse("12x\n", 128, ids, err));
    t.assert_true("two ids on a line are refused", !llama_mtp_vocab_parse("1 2\n", 128, ids, err));
    t.assert_true("a '+' is refused", !llama_mtp_vocab_parse("+3\n", 128, ids, err));
    t.assert_true("an empty list is refused", !llama_mtp_vocab_parse("# nothing\n\n", 128, ids, err));
    t.assert_true("past 64 bits is refused", !llama_mtp_vocab_parse("99999999999999999999\n", 128, ids, err));
}

// argmax over a row, -INF never wins
int64_t top(const float * row, int64_t n) {
    int64_t best = 0;
    for (int64_t v = 1; v < n; v++) {
        best = row[v] > row[best] ? v : best;
    }
    return best;
}

void run_case(testing & t, ggml_backend_dev_t dev, ggml_backend_sched_t sched, const head_case & hc) {
    std::mt19937 rng(1234 + (int) hc.type);
    const std::vector<uint8_t> bytes = make_head(hc, rng);
    const std::vector<int64_t> ids   = make_ids(hc.n_vocab, rng);
    device_head head(dev, hc, bytes);
    const size_t row = ggml_row_size(hc.type, hc.n_embd);

    std::string err;
    auto v = llama_mtp_vocab_gather(head.t, ids, err);
    if (!t.assert_true("gathered: " + err, v != nullptr)) {
        return;
    }
    v->n_vocab = hc.n_vocab;

    // the copy: the head's rows byte for byte, its type, its device, its weights usage, and nothing more
    std::vector<uint8_t> got(ggml_nbytes(v->weight));
    ggml_backend_tensor_get(v->weight, got.data(), 0, got.size());
    bool same_rows = v->weight->type == hc.type && v->weight->ne[0] == hc.n_embd && v->weight->ne[1] == (int64_t) ids.size();
    for (size_t k = 0; same_rows && k < ids.size(); k++) {
        same_rows = memcmp(got.data() + k*row, bytes.data() + ids[k]*row, row) == 0;
    }
    t.assert_true("each copied row is the head's row, byte for byte", same_rows);
    std::vector<int64_t> rows(ids.size());
    ggml_backend_tensor_get(v->rows, rows.data(), 0, rows.size()*sizeof(int64_t));
    t.assert_true("the id map is the list", rows == ids);
    t.assert_true("the copy lives in the head's device's own buffer type",
            ggml_backend_buffer_get_type(v->buf.get()) == ggml_backend_dev_buffer_type(dev));
    t.assert_true("as weights", ggml_backend_buffer_get_usage(v->buf.get()) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    t.assert_true("its size is the rows' bytes (+ the id map, padded)",
            v->nbytes() >= ids.size()*(row + sizeof(int64_t)) && v->nbytes() < ids.size()*(row + sizeof(int64_t)) + 4096);

    for (const int64_t n_out : { 1, 3, 9, 33 }) {
        ggml_init_params p = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context_ptr ctx(ggml_init(p));
        ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hc.n_embd, n_out);
        ggml_set_input(x);
        ggml_tensor * full = ggml_mul_mat(ctx.get(), head.t, x);
        ggml_tensor * trim = llama_mtp_vocab_logits(ctx.get(), *v, x);
        ggml_set_output(full);
        ggml_set_output(trim);
        ggml_cgraph * gf = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(gf, full);
        ggml_build_forward_expand(gf, trim);

        ggml_backend_sched_reset(sched);
        if (!t.assert_true("graph allocated", ggml_backend_sched_alloc_graph(sched, gf))) {
            return;
        }
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> xs(hc.n_embd*n_out);
        std::generate(xs.begin(), xs.end(), [&] { return u(rng); });
        ggml_backend_tensor_set(x, xs.data(), 0, xs.size()*sizeof(float));
        t.assert_true("computed", ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);

        std::vector<float> lf(hc.n_vocab*n_out), lt(hc.n_vocab*n_out);
        ggml_backend_tensor_get(full, lf.data(), 0, lf.size()*sizeof(float));
        ggml_backend_tensor_get(trim, lt.data(), 0, lt.size()*sizeof(float));
        const std::string o = std::to_string(n_out) + " outputs: ";
        t.assert_true(o + "the scores are [n_vocab, n_outputs]", trim->ne[0] == hc.n_vocab && trim->ne[1] == n_out);

        bool kept_same = true, rest_inf = true, top_same = true, signal = true;
        std::vector<bool> listed(hc.n_vocab, false);
        for (const int64_t id : ids) {
            listed[id] = true;
        }
        for (int64_t j = 0; j < n_out; j++) {
            const float * rf = lf.data() + j*hc.n_vocab;
            const float * rt = lt.data() + j*hc.n_vocab;
            for (int64_t w = 0; w < hc.n_vocab; w++) {
                if (listed[w]) {
                    kept_same = kept_same && memcmp(rf + w, rt + w, sizeof(float)) == 0;
                } else {
                    rest_inf = rest_inf && std::isinf(rt[w]) && rt[w] < 0;
                }
            }
            const int64_t tf = top(rf, hc.n_vocab);
            if (listed[tf]) {
                top_same = top_same && top(rt, hc.n_vocab) == tf;
            } else { // the trimmed top is the best listed word
                int64_t best = ids[0];
                for (const int64_t id : ids) {
                    best = rf[id] > rf[best] ? id : best;
                }
                top_same = top_same && top(rt, hc.n_vocab) == best;
            }
            signal = signal && rf[ids[0]] != rf[ids[1]] && std::isfinite(rf[ids[0]]) && rf[tf] > 1.0f;
        }
        t.assert_true(o + "the scores carry signal", signal);
        t.assert_true(o + "every listed word's score byte-identical to the whole head's", kept_same);
        t.assert_true(o + "every other word's score -INF", rest_inf);
        t.assert_true(o + "the top word the whole head's when listed, else the best listed", top_same);
    }
}

void test_refusals(testing & t, ggml_backend_dev_t dev) {
    const head_case hc = { GGML_TYPE_F16, 64, 100 };
    std::mt19937 rng(7);
    device_head head(dev, hc, make_head(hc, rng));
    std::string err;
    t.assert_true("no head", llama_mtp_vocab_gather(nullptr, { 1 }, err) == nullptr);
    ggml_init_params p = { ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(p));
    ggml_tensor * bare = ggml_new_tensor_2d(ctx.get(), hc.type, hc.n_embd, hc.n_vocab);
    t.assert_true("a head without data", llama_mtp_vocab_gather(bare, { 1 }, err) == nullptr);
    t.assert_true("no ids", llama_mtp_vocab_gather(head.t, {}, err) == nullptr);
    t.assert_true("ids out of order", llama_mtp_vocab_gather(head.t, { 5, 3 }, err) == nullptr);
    t.assert_true("an id past the head", llama_mtp_vocab_gather(head.t, { 3, 100 }, err) == nullptr);
    t.assert_true("a negative id", llama_mtp_vocab_gather(head.t, { -1, 3 }, err) == nullptr);
    t.assert_true("the last row is fine", llama_mtp_vocab_gather(head.t, { 99 }, err) != nullptr);
}

} // namespace

int main(int argc, char ** argv) {
    testing t;
    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    t.verbose = verbose && std::string(verbose) == "1";
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    ggml_backend_load_all();

    t.test("switch", test_switch);
    t.test("parse", test_parse);

    // the paperback's head is Q8_0 at 2,560 wide; 4,096+ rows keep Metal's small-batch limit at 10 columns.
    // 16,000 rows list ~6,400, more than the gather's 16 MiB staging holds at once (6,168 rows of 2,720 B), so
    // the copy runs in two chunks and the second lands after the first (cleric: a chunk written at row 0 passed
    // every smaller case)
    const std::vector<head_case> cases = {
        { GGML_TYPE_Q8_0, 2560, 6000 },
        { GGML_TYPE_Q8_0, 2560, 16000 },
        { GGML_TYPE_F16,  256,  1000 },
        { GGML_TYPE_F32,  256,  1000 },
    };

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);
    for (size_t d = 0; d < ggml_backend_dev_count(); d++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_CPU && type != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_t backend = type == GGML_BACKEND_DEVICE_TYPE_CPU ? cpu : ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend != nullptr);
        // the scheduler llama.cpp computes with: the device, then the CPU as its fallback
        std::vector<ggml_backend_t> backends = { backend };
        if (backend != cpu) {
            backends.push_back(cpu);
        }
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(),
                                                            GGML_DEFAULT_GRAPH_SIZE, false, true);
        t.test(ggml_backend_dev_name(dev), [&](testing & t) {
            t.test("refusals", [&](testing & t) { test_refusals(t, dev); });
            for (const head_case & hc : cases) {
                t.test(hc.name(), [&](testing & t) { run_case(t, dev, sched, hc); });
            }
        });
        ggml_backend_sched_free(sched);
        if (backend != cpu) {
            ggml_backend_free(backend);
        }
    }
    ggml_backend_free(cpu);
    return t.summary();
}

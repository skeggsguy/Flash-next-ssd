#include "llama-mtp-vocab.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

// the gather's host staging: at most this many bytes of rows in flight
static constexpr size_t LLAMA_MTP_VOCAB_STAGING = 16u << 20;

size_t llama_mtp_vocab::nbytes() const {
    return buf ? ggml_backend_buffer_get_size(buf.get()) : 0;
}

bool llama_mtp_vocab_wanted(const char * value) {
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool llama_mtp_vocab_supported(llm_arch arch) {
    return arch == LLM_ARCH_QWEN4EXP;
}

bool llama_mtp_vocab_parse(const std::string & text, int64_t n_vocab, std::vector<int64_t> & ids, std::string & err) {
    ids.clear();
    std::istringstream in(text);
    std::string line;
    for (int64_t n_line = 1; std::getline(in, line); n_line++) {
        const size_t b = line.find_first_not_of(" \t\r");
        if (b == std::string::npos || line[b] == '#') {
            continue;
        }
        const size_t e = line.find_last_not_of(" \t\r");
        const std::string word = line.substr(b, e - b + 1);
        char * end = nullptr;
        errno = 0;
        const long long id = std::strtoll(word.c_str(), &end, 10);
        if (errno != 0 || end != word.c_str() + word.size() || word[0] == '+') {
            err = format("line %lld is not a word id: \"%s\"", (long long) n_line, word.c_str());
            return false;
        }
        if (id < 0 || id >= n_vocab) {
            err = format("line %lld: word id %lld is outside the vocabulary (0..%lld)",
                    (long long) n_line, id, (long long) n_vocab - 1);
            return false;
        }
        ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (ids.empty()) {
        err = "the list holds no word ids";
        return false;
    }
    return true;
}

std::unique_ptr<llama_mtp_vocab> llama_mtp_vocab_gather(const ggml_tensor * head, const std::vector<int64_t> & ids,
        std::string & err) {
    if (head == nullptr || head->buffer == nullptr) {
        err = "the head has no data to copy";
        return nullptr;
    }
    const size_t row = ggml_row_size(head->type, head->ne[0]);
    if (ggml_n_dims(head) != 2 || head->nb[1] != row || !ggml_is_contiguous(head)) {
        err = "the head's rows are not one contiguous matrix";
        return nullptr;
    }
    if (ids.empty() || ids.front() < 0 || ids.back() >= head->ne[1] || !std::is_sorted(ids.begin(), ids.end())) {
        err = "the word ids are not ascending ids of the head's rows";
        return nullptr;
    }

    // the copy lives where the head does, in the device's own buffer type, so the same kernel scores it
    ggml_backend_buffer_type_t src_buft = ggml_backend_buffer_get_type(head->buffer);
    ggml_backend_dev_t         dev      = ggml_backend_buft_get_device(src_buft);
    ggml_backend_buffer_type_t buft     = nullptr;
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        if (!ggml_backend_buft_is_host(src_buft)) {
            err = format("the head sits in %s, whose rows are rearranged (repacked) for the CPU",
                    ggml_backend_buft_name(src_buft));
            return nullptr;
        }
        buft = ggml_backend_cpu_buffer_type();
    } else {
        buft = ggml_backend_dev_buffer_type(dev);
    }

    auto v = std::make_unique<llama_mtp_vocab>();
    v->ids = ids;
    const int64_t n = (int64_t) ids.size();

    ggml_init_params params = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    v->ctx.reset(ggml_init(params));
    if (!v->ctx) {
        err = "cannot make a ggml context for the copy";
        return nullptr;
    }
    v->weight = ggml_new_tensor_2d(v->ctx.get(), head->type, head->ne[0], n);
    v->rows   = ggml_new_tensor_1d(v->ctx.get(), GGML_TYPE_I64, n);
    ggml_set_name(v->weight, "mtp_vocab.weight");
    ggml_set_name(v->rows,   "mtp_vocab.rows");

    v->buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(v->ctx.get(), buft));
    if (!v->buf) {
        err = format("cannot allocate %.1f MiB on %s", n*row/1024.0/1024.0, ggml_backend_buft_name(buft));
        return nullptr;
    }
    // weights: the scheduler runs the head's maths where these bytes live, as it does for the head
    ggml_backend_buffer_set_usage(v->buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // runs of consecutive ids are one read each; a chunk of rows is one write
    const int64_t chunk = std::max<int64_t>(1, (int64_t) (LLAMA_MTP_VOCAB_STAGING / row));
    std::vector<uint8_t> staging((size_t) std::min(n, chunk)*row);
    for (int64_t k0 = 0; k0 < n; k0 += chunk) {
        const int64_t k1 = std::min(n, k0 + chunk);
        for (int64_t k = k0; k < k1; ) {
            int64_t e = k + 1;
            while (e < k1 && ids[e] == ids[e - 1] + 1) {
                e++;
            }
            ggml_backend_tensor_get(head, staging.data() + (k - k0)*row, ids[k]*row, (e - k)*row);
            k = e;
        }
        ggml_backend_tensor_set(v->weight, staging.data(), k0*row, (k1 - k0)*row);
    }
    ggml_backend_tensor_set(v->rows, ids.data(), 0, n*sizeof(int64_t));
    return v;
}

std::unique_ptr<llama_mtp_vocab> llama_mtp_vocab_init(const char * value, bool is_mtp_ctx,
        const llama_model & model, const llama_context * ctx_other) {
    if (!is_mtp_ctx) {
        return nullptr;
    }
    const int64_t n_vocab   = model.vocab.n_tokens();
    const bool    supported = llama_mtp_vocab_supported(model.arch);
    if (!llama_mtp_vocab_wanted(value)) {
        if (supported) {
            LLAMA_LOG_WARN("%s: apprentice's word list: off, guesses from all %lld words "
                    "(LLAMA_MTP_VOCAB=<list> guesses from the listed words only)\n", __func__, (long long) n_vocab);
        }
        return nullptr;
    }
    if (!supported) {
        LLAMA_LOG_WARN("%s: apprentice's word list: LLAMA_MTP_VOCAB is set, but %s's apprentice takes no list; "
                "it guesses from all %lld words\n", __func__, llm_arch_name(model.arch), (long long) n_vocab);
        return nullptr;
    }
    if (model.hparams.no_alloc) {
        return nullptr; // sizing only (memory fitting): no bytes to copy, and the whole head is the larger graph
    }
    // graph_mtp's head: the model's own, else the target's (a shared sidecar carries none)
    const ggml_tensor * head = model.output;
    if (head == nullptr && ctx_other != nullptr) {
        head = llama_get_model(ctx_other)->output;
    }
    if (head == nullptr) {
        return nullptr; // graph_mtp names the missing head itself
    }
    if (head->ne[1] != n_vocab || head->ne[0] != (int64_t) model.hparams.n_embd) {
        throw std::runtime_error(format("apprentice's word list: the head is %lld x %lld, the apprentice's words "
                "%u x %lld", (long long) head->ne[0], (long long) head->ne[1], model.hparams.n_embd, (long long) n_vocab));
    }
    if (model.output_s != nullptr) {
        throw std::runtime_error("apprentice's word list: this head has an output scale, which the list does not "
                "carry; unset LLAMA_MTP_VOCAB");
    }

    std::ifstream f(value, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error(format("apprentice's word list: cannot read %s (LLAMA_MTP_VOCAB)", value));
    }
    std::stringstream text;
    text << f.rdbuf();
    std::vector<int64_t> ids;
    std::string err;
    if (!llama_mtp_vocab_parse(text.str(), n_vocab, ids, err)) {
        throw std::runtime_error(format("apprentice's word list %s: %s", value, err.c_str()));
    }
    const int64_t t0 = ggml_time_us();
    auto v = llama_mtp_vocab_gather(head, ids, err);
    if (!v) {
        throw std::runtime_error(format("apprentice's word list %s: %s", value, err.c_str()));
    }
    v->n_vocab = n_vocab;
    v->source  = value;
    LLAMA_LOG_WARN("%s: apprentice's word list: on, guesses from %zu of %lld words (%s), the head's listed rows "
            "copied once: %.1f MiB of %s in %s, %.2f s (LLAMA_MTP_VOCAB=0 guesses from every word)\n", __func__,
            v->ids.size(), (long long) n_vocab, value, v->nbytes()/1024.0/1024.0, ggml_type_name(head->type),
            ggml_backend_buffer_name(v->buf.get()), (ggml_time_us() - t0)/1e6);
    return v;
}

ggml_tensor * llama_mtp_vocab_logits(ggml_context * ctx0, const llama_mtp_vocab & v, ggml_tensor * cur) {
    const int64_t n_kept    = v.weight->ne[1];
    const int64_t n_outputs = cur->ne[1];

    ggml_tensor * kept = ggml_mul_mat(ctx0, v.weight, cur); // [n_kept, n_outputs]
    ggml_set_name(kept, "mtp_vocab_kept");

    // every word -INF, then each listed word's score into its own place (the rows broadcast over outputs)
    ggml_tensor * full = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, v.n_vocab, n_outputs), -INFINITY);
    full = ggml_set_rows(ctx0, full,
            ggml_reshape_3d(ctx0, kept,   1,      n_kept, n_outputs),
            ggml_reshape_3d(ctx0, v.rows, n_kept, 1,      1));
    return ggml_reshape_2d(ctx0, full, v.n_vocab, n_outputs);
}

// The trained two-floor guesser (LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER and _CONF): what it is, and the rules, are in
// llama-moe-stream-guesser.h. This file loads it, gives it to the floors' lookahead, and holds the confidence pick.

#include "llama-moe-stream-guesser.h"

#include "llama-impl.h"
#include "llama-moe-stream.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>

#if defined(__APPLE__)
#include <fcntl.h>
#endif

size_t llama_moe_stream_guesser::nbytes() const {
    size_t n = 0;
    for (const auto & b : bufs) {
        n += ggml_backend_buffer_get_size(b.get());
    }
    return n;
}

const char * llama_moe_stream_guesser_env() {
    const char * s = std::getenv("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER");
    return s != nullptr && *s != '\0' ? s : nullptr;
}

float llama_moe_stream_lookahead_conf_env() {
    const char * s = std::getenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF");
    if (s == nullptr || *s == '\0' || strcmp(s, "0") == 0) {
        return 0.0f;
    }
    char * end = nullptr;
    errno = 0;
    const float cut = std::strtof(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !(cut > 0.0f && cut <= 1.0f)) {
        throw std::runtime_error(format("LLAMA_MOE_STREAM_LOOKAHEAD_CONF=%s is not a confidence cut in (0, 1]", s));
    }
    return cut;
}

namespace {

bool key_u32(const gguf_context * g, const char * key, uint32_t & v, std::string & err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) {
        err = format("no %s (a u32)", key);
        return false;
    }
    v = gguf_get_val_u32(g, id);
    return true;
}

bool key_f32(const gguf_context * g, const char * key, float & v, std::string & err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) {
        err = format("no %s (an f32)", key);
        return false;
    }
    v = gguf_get_val_f32(g, id);
    return true;
}

// the file's lookahead.* keys against the library's widths and floors
bool check_keys(const gguf_context * g, int64_t n_embd, int64_t n_expert, int32_t n_layer,
        llama_moe_stream_guesser & out, std::string & err) {
    uint32_t offset = 0, first = 0, last = 0, f_embd = 0, f_expert = 0;
    if (!key_u32(g, "lookahead.floor_offset", offset, err) || !key_u32(g, "lookahead.first_floor", first, err) ||
            !key_u32(g, "lookahead.last_floor", last, err) || !key_u32(g, "lookahead.n_embd", f_embd, err) ||
            !key_u32(g, "lookahead.n_expert", f_expert, err) ||
            !key_f32(g, "lookahead.confidence.scale", out.conf_scale, err) ||
            !key_f32(g, "lookahead.confidence.max", out.conf_max, err)) {
        return false;
    }
    if (offset != 2) {
        err = format("its guess reads the input %u floors down; the two-floor guess reads 2", offset);
        return false;
    }
    if ((int64_t) f_embd != n_embd || (int64_t) f_expert != n_expert) {
        err = format("it is %u wide with %u books a floor; the library is %lld wide with %lld", f_embd, f_expert,
                (long long) n_embd, (long long) n_expert);
        return false;
    }
    if (first < 2 || first > last || (int64_t) last >= n_layer) {
        err = format("its floors %u..%u do not fit the library's two-floor guesses (floors 2..%d)", first, last,
                n_layer - 1);
        return false;
    }
    if (!(out.conf_scale > 0.0f) || !(out.conf_max > 0.0f)) {
        err = format("its confidence rule min(%g, %g * softmax) is not positive", out.conf_max, out.conf_scale);
        return false;
    }
    out.first = (int32_t) first;
    out.last  = (int32_t) last;
    return true;
}

// tensor `name` of the file's metadata, of a wanted type and shape
const ggml_tensor * find(ggml_context * meta, const std::string & name, bool weight, int64_t n_embd,
        int64_t n_expert, std::string & err) {
    const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
    if (t == nullptr) {
        err = format("no %s", name.c_str());
        return nullptr;
    }
    const bool type_ok  = weight ? (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32) : t->type == GGML_TYPE_F32;
    const bool shape_ok = weight ? (t->ne[0] == n_embd && t->ne[1] == n_expert && t->ne[2] == 1 && t->ne[3] == 1)
                                 : (t->ne[0] == n_expert && t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1);
    if (!type_ok || !shape_ok) {
        err = format("%s is %s [%lld, %lld, %lld, %lld]; wanted %s [%lld%s]", name.c_str(), ggml_type_name(t->type),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                weight ? "F16 or F32" : "F32", (long long) (weight ? n_embd : n_expert),
                weight ? format(", %lld", (long long) n_expert).c_str() : "");
        return nullptr;
    }
    return t;
}

} // namespace

std::unique_ptr<llama_moe_stream_guesser> llama_moe_stream_guesser_load(const char * path, int64_t n_embd,
        int64_t n_expert, int32_t n_layer, const std::function<ggml_backend_buffer_type_t(int32_t)> & buft_of,
        std::string & err) {
    ggml_context * meta_raw = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_raw };
    gguf_context_ptr g(gguf_init_from_file(path, params));
    ggml_context_ptr meta(meta_raw);
    if (!g) {
        err = "cannot read it as a GGUF file";
        return nullptr;
    }
    auto out = std::make_unique<llama_moe_stream_guesser>();
    out->source = path;
    if (!check_keys(g.get(), n_embd, n_expert, n_layer, *out, err)) {
        return nullptr;
    }

    // every floor's two tensors, checked, then grouped by where their routers live
    std::vector<const ggml_tensor *> src_w(n_layer, nullptr), src_b(n_layer, nullptr);
    std::map<ggml_backend_buffer_type_t, std::vector<int32_t>> by_buft;
    for (int32_t n = out->first; n <= out->last; n++) {
        src_w[n] = find(meta.get(), format("blk.%d.la_guess.weight", n), true, n_embd, n_expert, err);
        src_b[n] = src_w[n] ? find(meta.get(), format("blk.%d.la_guess.bias", n), false, n_embd, n_expert, err) : nullptr;
        if (src_b[n] == nullptr) {
            return nullptr;
        }
        by_buft[buft_of(n)].push_back(n);
    }

    out->weight.assign(n_layer, nullptr);
    out->bias.assign(n_layer, nullptr);
    for (const auto & [buft, floors] : by_buft) {
        ggml_init_params ip = { /*.mem_size =*/ 2*floors.size()*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr,
                                /*.no_alloc =*/ true };
        ggml_context_ptr ctx(ggml_init(ip));
        if (!ctx) {
            err = "cannot make a ggml context for it";
            return nullptr;
        }
        for (const int32_t n : floors) {
            out->weight[n] = ggml_new_tensor_2d(ctx.get(), src_w[n]->type, n_embd, n_expert);
            out->bias[n]   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_expert);
            ggml_set_name(out->weight[n], src_w[n]->name);
            ggml_set_name(out->bias[n],   src_b[n]->name);
        }
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!buf) {
            err = format("cannot allocate it in %s", ggml_backend_buft_name(buft));
            return nullptr;
        }
        // weights: the scheduler runs the guess where these bytes live, as it does for the router it replaces
        ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        out->ctxs.push_back(std::move(ctx));
        out->bufs.push_back(std::move(buf));
    }

    // the bytes, one positioned read a tensor, kept out of macOS's file cache (a long read's clean pages are what
    // grows the pile)
    std::unique_ptr<FILE, int (*)(FILE *)> f(std::fopen(path, "rb"), &std::fclose);
    if (!f) {
        err = format("cannot open it: %s", strerror(errno));
        return nullptr;
    }
#if defined(__APPLE__)
    fcntl(fileno(f.get()), F_NOCACHE, 1);
#endif
    const size_t data0 = gguf_get_data_offset(g.get());
    std::vector<uint8_t> staging;
    for (int32_t n = out->first; n <= out->last; n++) {
        for (ggml_tensor * t : { out->weight[n], out->bias[n] }) {
            const int64_t id = gguf_find_tensor(g.get(), t->name);
            const size_t  nb = ggml_nbytes(t);
            staging.resize(nb);
            if (fseeko(f.get(), (off_t) (data0 + gguf_get_tensor_offset(g.get(), id)), SEEK_SET) != 0 ||
                    fread(staging.data(), 1, nb, f.get()) != nb) {
                err = format("cannot read %s's %zu bytes (the file is cut short?)", t->name, nb);
                return nullptr;
            }
            ggml_backend_tensor_set(t, staging.data(), 0, nb);
        }
    }
    return out;
}

void llama_moe_stream_guesser_attach(llama_moe_stream & ms, const std::vector<const ggml_tensor *> & routers,
        int64_t n_embd, int64_t n_expert, bool no_alloc) {
    const char *  path = llama_moe_stream_guesser_env();
    const float   cut  = llama_moe_stream_lookahead_conf_env();
    if (path == nullptr) {
        if (cut > 0.0f) {
            throw std::runtime_error("LLAMA_MOE_STREAM_LOOKAHEAD_CONF picks by the trained guesser's confidence: "
                    "set LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER too, or unset it");
        }
        return; // today's two-floor guess
    }
    if (ms.gpu_slot != 0) {
        throw std::runtime_error("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER: the GPU slot servicer (LLAMA_MOE_STREAM_GPU_SLOT) "
                "writes without the lookahead; unset one of them");
    }
    if (no_alloc) {
        return; // sizing only: no bytes to read
    }
    // floor n's guesser shares its router's device (the mtp word list's rule, llama-mtp-vocab.cpp)
    auto buft_of = [&](int32_t n) -> ggml_backend_buffer_type_t {
        const ggml_tensor * r = (size_t) n < routers.size() ? routers[n] : nullptr;
        ggml_backend_dev_t dev = r && r->buffer ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(r->buffer)) : nullptr;
        if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return ggml_backend_cpu_buffer_type();
        }
        return ggml_backend_dev_buffer_type(dev);
    };
    const int64_t t0 = ggml_time_us();
    std::string err;
    auto g = llama_moe_stream_guesser_load(path, n_embd, n_expert, (int32_t) routers.size(), buft_of, err);
    if (!g) {
        throw std::runtime_error(format("the trained guesser %s (LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER): %s", path, err.c_str()));
    }

    int32_t n_on = 0, n_off = 0;
    for (const auto & sl : ms.layers) {
        if (!sl || !sl->la || !sl->la->sl_next2) {
            continue; // no two-floor guess from this floor
        }
        auto * la = sl->la;
        const int32_t n = la->sl_next2->il;
        if (n >= g->first && n <= g->last && g->weight[n] != nullptr) {
            sl->la_guess   = g->weight[n];
            sl->la_guess_b = g->bias[n];
            la->guessed    = true;
            la->bias_src2  = nullptr; // the guesser's bias is in its scores; floor n's router bias is not
            la->conf_cut   = cut;
            la->conf_scale = g->conf_scale;
            la->conf_max   = g->conf_max;
            n_on++;
        } else {
            sl->la_gate_inp2 = nullptr; // never the untrained guess beside the trained one
            la->sl_next2     = nullptr;
            la->top_k2       = 0;
            n_off++;
        }
    }
    if (n_on == 0) {
        throw std::runtime_error(format("the trained guesser %s serves no floor: it is the two-floor guess, so it needs "
                "LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2 and the lookahead on", path));
    }
    char pick[96];
    if (cut > 0.0f) {
        snprintf(pick, sizeof(pick), "books at confidence >= %.3g of each word's %u best", cut, LLAMA_MOE_STREAM_GUESS_PER_WORD);
    } else {
        snprintf(pick, sizeof(pick), "each word's best books, rank by rank (LLAMA_MOE_STREAM_LOOKAHEAD_CONF unset)");
    }
    LLAMA_LOG_WARN("%s: moe stream: two-floor guess: the trained guesser %s on %d floors (%d with none in it guess "
            "nothing two ahead), %.1f MiB beside the desk in %s, read in %.2f s; fetches %s\n", __func__, path, n_on,
            n_off, g->nbytes()/1048576.0, g->bufs.empty() ? "-" : ggml_backend_buffer_name(g->bufs[0].get()),
            (ggml_time_us() - t0)/1e6, pick);
    ms.guesser = std::move(g);
}

void llama_moe_stream_guess_pick(const float * rows, int64_t stride, int64_t t_first, int64_t n_tok, uint32_t n,
        float cut, float scale, float conf_max, uint32_t per_word, std::vector<int32_t> & books,
        llama_moe_stream_guess_scratch & s, int64_t & n_under) {
    books.clear();
    s.best.assign(n, -1.0f);
    s.under.assign(n, 0);
    s.order.resize(n);
    const uint32_t k = std::min(per_word, n);
    for (int64_t t = n_tok - 1; t >= t_first; t--) { // the last word first
        const float * z = rows + t*stride;
        float zmax = -INFINITY;
        for (uint32_t e = 0; e < n; e++) {
            zmax = std::max(zmax, z[e]);
        }
        double sum = 0.0;
        for (uint32_t e = 0; e < n; e++) {
            sum += std::exp((double) z[e] - zmax);
        }
        // the word's k best by z (equal scores: the lower book first, as the untrained pick's scan has it)
        std::iota(s.order.begin(), s.order.end(), 0);
        std::partial_sort(s.order.begin(), s.order.begin() + k, s.order.end(), [z](int32_t a, int32_t b) {
            return z[a] > z[b] || (z[a] == z[b] && a < b);
        });
        for (uint32_t r = 0; r < k; r++) {
            const int32_t e    = s.order[r];
            const float   conf = (float) std::min<double>(conf_max, scale*std::exp((double) z[e] - zmax)/sum);
            if (conf < cut) {
                s.under[e] = 1;
                continue;
            }
            if (s.best[e] < 0.0f) {
                books.push_back(e); // first seen: the last word's first, then by rank
            }
            s.best[e] = std::max(s.best[e], conf);
        }
    }
    // the most confident first; stable, so equal confidences keep the order they were seen in
    std::stable_sort(books.begin(), books.end(), [&s](int32_t a, int32_t b) { return s.best[a] > s.best[b]; });
    for (uint32_t e = 0; e < n; e++) {
        n_under += s.under[e] && s.best[e] < 0.0f;
    }
}

// The trained two-floor guesser (LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER and _CONF, src/llama-moe-stream-guesser.h),
// without a model:
//
//   pick     the confidence pick against hand-worked scores: a word's confidence is min(max, scale * softmax(z));
//            every word's per_word best by z, the last word first; a book is kept when it reaches the cut in
//            some word, at its highest confidence, most confident first, equal confidences in the order seen
//            (the last word's books first, then by rank); the books left under the cut are counted; only each
//            row's own n cells are read (the rows are a [n1 + n2] concat in the engine)
//   switch   CONF unset, "" and "0" are the fixed pick; a number in (0, 1] is the cut; anything else refuses
//   load     a file written here loads bit for bit into the asked buffer, F16 or F32 weights; a wrong width,
//            book count, floor offset, floor range, tensor shape or type, a missing tensor, a missing or cut
//            short file are refused with a reason
//   attach   on a book manager made by hand: each floor's two-floor guess gets floor L+2's guesser, its raw
//            ranking and the cut; a floor whose floor two up has no guess in the file guesses nothing two
//            ahead; CONF without GUESSER, the GPU slot servicer, and a guesser that serves no floor refuse

#include "testing.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include "../src/llama-moe-stream.h"
#include "../src/llama-moe-stream-guesser.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

// a word's z from its softmax shares (any positive weights; softmax(log w) = w / sum w)
std::vector<float> zs(const std::vector<float> & w) {
    std::vector<float> z;
    for (const float x : w) {
        z.push_back(logf(x));
    }
    return z;
}

struct pick_result {
    std::vector<int32_t> books;
    int64_t under = 0;
};

// words given last-word-last, as a batch holds them; stride = n + pad, each pad cell NAN
pick_result pick(const std::vector<std::vector<float>> & words, int64_t t_first, float cut, float scale,
        uint32_t per_word, uint32_t pad = 0) {
    const uint32_t n = (uint32_t) words[0].size();
    std::vector<float> rows(words.size()*(n + pad), NAN);
    for (size_t t = 0; t < words.size(); t++) {
        std::copy(words[t].begin(), words[t].end(), rows.begin() + t*(n + pad) + pad);
    }
    pick_result r;
    llama_moe_stream_guess_scratch s;
    llama_moe_stream_guess_pick(rows.data() + pad, n + pad, t_first, (int64_t) words.size(), n, cut, scale, 1.0f,
            per_word, r.books, s, r.under);
    return r;
}

struct gfile {
    int64_t   n_embd = 8, n_expert = 4;          // the tensors' shape
    uint32_t  key_embd = 8, key_expert = 4;      // what the keys say
    uint32_t  offset = 2, first = 2, last = 3;
    ggml_type wtype = GGML_TYPE_F16, btype = GGML_TYPE_F32;
    bool      bias_of_last = true;               // false: floor `last` has no bias tensor
    bool      keys = true;                       // false: no lookahead.* keys at all
};

// writes a guesser file; weights[n] holds floor n's weight bytes, biases[n] its bias bytes
std::string write_file(const std::string & name, const gfile & c, std::vector<std::vector<uint8_t>> & weights,
        std::vector<std::vector<uint8_t>> & biases) {
    const std::string path = (std::filesystem::temp_directory_path() /
            ("test-moe-stream-guesser-" + std::to_string(getpid()) + "-" + name + ".gguf")).string();
    gguf_context * g = gguf_init_empty();
    if (c.keys) {
        gguf_set_val_u32(g, "lookahead.floor_offset", c.offset);
        gguf_set_val_u32(g, "lookahead.first_floor", c.first);
        gguf_set_val_u32(g, "lookahead.last_floor", c.last);
        gguf_set_val_u32(g, "lookahead.n_embd", c.key_embd);
        gguf_set_val_u32(g, "lookahead.n_expert", c.key_expert);
        gguf_set_val_f32(g, "lookahead.confidence.scale", 10.0f);
        gguf_set_val_f32(g, "lookahead.confidence.max", 1.0f);
    }
    ggml_init_params ip = { 64*ggml_tensor_overhead() + (size_t) (c.last + 1)*(c.n_embd*c.n_expert*4 + c.n_expert*4 + 256),
                            nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(c.first*31 + c.last);
    std::uniform_int_distribution<int> byte(0, 255);
    weights.assign(c.last + 1, {});
    biases.assign(c.last + 1, {});
    for (uint32_t n = c.first; n <= c.last; n++) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, c.wtype, c.n_embd, c.n_expert);
        ggml_set_name(w, ("blk." + std::to_string(n) + ".la_guess.weight").c_str());
        auto * wd = (uint8_t *) w->data;
        for (size_t i = 0; i < ggml_nbytes(w); i++) {
            wd[i] = (uint8_t) byte(rng); // any bit pattern must survive, NaNs included
        }
        weights[n].assign(wd, wd + ggml_nbytes(w));
        gguf_add_tensor(g, w);
        if (n == c.last && !c.bias_of_last) {
            continue;
        }
        ggml_tensor * b = ggml_new_tensor_1d(ctx, c.btype, c.n_expert);
        ggml_set_name(b, ("blk." + std::to_string(n) + ".la_guess.bias").c_str());
        auto * bd = (uint8_t *) b->data;
        for (size_t i = 0; i < ggml_nbytes(b); i++) {
            bd[i] = (uint8_t) byte(rng);
        }
        biases[n].assign(bd, bd + ggml_nbytes(b));
        gguf_add_tensor(g, b);
    }
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

std::unique_ptr<llama_moe_stream_guesser> load(const std::string & path, std::string & err, int32_t n_layer = 4) {
    err.clear();
    return llama_moe_stream_guesser_load(path.c_str(), 8, 4, n_layer,
            [](int32_t) { return ggml_backend_cpu_buffer_type(); }, err);
}

std::vector<uint8_t> bytes_of(const ggml_tensor * t) {
    std::vector<uint8_t> v(ggml_nbytes(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size());
    return v;
}

// a book manager of n_layer floors with the lookahead set up as llama-model.cpp sets it, two floors ahead from
// every floor that has one
struct floors {
    std::unique_ptr<llama_moe_stream> mgr;
    floors(int32_t n_layer, int gpu_slot = 0) {
        mgr = std::make_unique<llama_moe_stream>(n_layer, 8, 1, false);
        mgr->gpu_slot = gpu_slot;
        for (int32_t il = 0; il < n_layer; il++) {
            mgr->layers[il] = std::make_unique<llama_moe_stream_layer>();
            mgr->layers[il]->mgr = mgr.get();
            mgr->layers[il]->il = il;
            mgr->layers[il]->n_expert = 4;
        }
        for (int32_t il = 0; il + 1 < n_layer; il++) {
            auto * sl = mgr->layers[il].get();
            sl->la = new llama_moe_stream_lookahead();
            sl->la->sl = sl;
            sl->la->sl_next = mgr->layers[il + 1].get();
            sl->la->top_k = 8;
            if (il + 2 < n_layer) {
                sl->la->sl_next2 = mgr->layers[il + 2].get();
                sl->la->top_k2 = 4;
                sl->la_gate_inp2 = (ggml_tensor *) 0x1; // stands for floor L+2's router; never read here
            }
        }
    }
    ~floors() {
        for (auto & sl : mgr->layers) {
            delete sl->la;
        }
    }
    llama_moe_stream_layer * sl(int il) { return mgr->layers[il].get(); }
};

bool throws(const std::function<void()> & f, const char * want) {
    try {
        f();
    } catch (const std::exception & e) {
        return strstr(e.what(), want) != nullptr;
    }
    return false;
}

} // namespace

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
    ggml_backend_load_all();
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER");
    unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF");

    t.test("pick_one_word", [&](testing & t) {
        // shares 0.5 0.25 0.1 0.1 0.05, scale 2: confidences 1, 0.5, 0.2, 0.2, 0.1
        const auto z = zs({ 10, 5, 2, 2, 1 });
        auto r = pick({ z }, 0, 0.15f, 2.0f, 24);
        t.assert_true("books over the cut, most confident first; the equal two by book", r.books == std::vector<int32_t>({ 0, 1, 2, 3 }));
        t.assert_equal("the one book under the cut is counted", (int64_t) 1, r.under);
        r = pick({ z }, 0, 0.15f, 2.0f, 2);
        t.assert_true("only the word's per_word best are offered", r.books == std::vector<int32_t>({ 0, 1 }));
        t.assert_equal("books outside the best are not under the cut, just not offered", (int64_t) 0, r.under);
        r = pick({ z }, 0, 0.6f, 2.0f, 24);
        t.assert_true("cut 0.6: the first book alone", r.books == std::vector<int32_t>({ 0 }));
        t.assert_equal("and four under it", (int64_t) 4, r.under);
        r = pick({ z }, 0, 0.15f, 2.0f, 24, /*pad =*/ 3);
        t.assert_true("rows in a wider concat read only their own cells", r.books == std::vector<int32_t>({ 0, 1, 2, 3 }) && r.under == 1);
    });

    t.test("pick_clamps_at_max_and_keeps_a_cut_it_equals", [&](testing & t) {
        // shares 0.4 0.3 0.2 0.1, scale 10: 4, 3, 2, 1 -> every confidence is the max, 1, exactly
        const auto r = pick({ zs({ 4, 3, 2, 1 }) }, 0, 1.0f, 10.0f, 24);
        t.assert_true("a cut of 1 keeps every book at the max, in rank order", r.books == std::vector<int32_t>({ 0, 1, 2, 3 }));
        t.assert_equal("none under", (int64_t) 0, r.under);
    });

    t.test("pick_every_word_last_first", [&](testing & t) {
        // scale 1, so a confidence is the share itself
        const auto early = zs({ 1, 1, 7, 1 });  // 0.1 0.1 0.7 0.1
        const auto last  = zs({ 6, 2, 1, 1 });  // 0.6 0.2 0.1 0.1
        auto r = pick({ early, last }, 0, 0.15f, 1.0f, 24);
        t.assert_true("each book at its highest confidence over the words: 2 (0.7), 0 (0.6), 1 (0.2)",
                r.books == std::vector<int32_t>({ 2, 0, 1 }));
        t.assert_equal("a book under the cut in one word but over it in another is not counted under", (int64_t) 1, r.under);
        r = pick({ early, last }, 1, 0.15f, 1.0f, 24);
        t.assert_true("t_first = the last word: its books alone", r.books == std::vector<int32_t>({ 0, 1 }));
        t.assert_equal("and the earlier word's book is not counted", (int64_t) 2, r.under);

        // equal confidences: the last word's books first, then by rank
        const auto a = zs({ 1, 1, 4, 4 });      // books 2, 3 at 0.4
        const auto b = zs({ 4, 4, 1, 1 });      // books 0, 1 at 0.4, the same bits
        r = pick({ a, b }, 0, 0.3f, 1.0f, 24);
        t.assert_true("ties keep the order seen: the last word's 0, 1, then the earlier word's 2, 3",
                r.books == std::vector<int32_t>({ 0, 1, 2, 3 }));
        r = pick({ b, a }, 0, 0.3f, 1.0f, 24);
        t.assert_true("and turned around: 2, 3, then 0, 1", r.books == std::vector<int32_t>({ 2, 3, 0, 1 }));

        // a book's place comes from its best word, not the first word that offered it
        const auto lo = zs({ 3, 3, 3, 1 });     // 0.3 0.3 0.3 0.1
        const auto hi = zs({ 1, 1, 7, 1 });     // book 2 at 0.7
        r = pick({ hi, lo }, 0, 0.25f, 1.0f, 24);
        t.assert_true("book 2, first seen at 0.3 in the last word, ranks by its 0.7", r.books == std::vector<int32_t>({ 2, 0, 1 }));
    });

    t.test("conf_switch", [&](testing & t) {
        auto conf = [](const char * v) {
            if (v) {
                setenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF", v, 1);
            } else {
                unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF");
            }
            return llama_moe_stream_lookahead_conf_env();
        };
        t.assert_true("unset, \"\" and \"0\" are the fixed pick", conf(nullptr) == 0.0f && conf("") == 0.0f && conf("0") == 0.0f);
        t.assert_true("0.2 and 1 are cuts", conf("0.2") == 0.2f && conf("1") == 1.0f);
        for (const char * bad : { "abc", "0.2x", "-0.1", "1.5", "nan" }) {
            t.assert_true(std::string("\"") + bad + "\" refuses", throws([&] { conf(bad); }, "LLAMA_MOE_STREAM_LOOKAHEAD_CONF"));
        }
        unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF");
    });

    t.test("load", [&](testing & t) {
        std::vector<std::vector<uint8_t>> w, b;
        std::string err;
        for (const ggml_type wt : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            gfile c;
            c.wtype = wt;
            const std::string path = write_file(std::string("good-") + ggml_type_name(wt), c, w, b);
            auto g = load(path, err);
            t.assert_true(std::string("a good ") + ggml_type_name(wt) + " file loads (" + err + ")", g != nullptr);
            if (g) {
                t.assert_true("floors 2..3, nothing below", g->first == 2 && g->last == 3 && !g->weight[0] && !g->weight[1]);
                t.assert_true("every tensor bit for bit", bytes_of(g->weight[2]) == w[2] && bytes_of(g->weight[3]) == w[3] &&
                        bytes_of(g->bias[2]) == b[2] && bytes_of(g->bias[3]) == b[3]);
                t.assert_true("the weight keeps its type", g->weight[2]->type == wt);
                t.assert_true("the confidence rule from the keys", g->conf_scale == 10.0f && g->conf_max == 1.0f);
                t.assert_true("its bytes are counted", g->nbytes() >= 2*(w[2].size() + b[2].size()));
            }
            std::filesystem::remove(path);
        }

        struct refusal { const char * name; gfile c; const char * want; int32_t n_layer; };
        std::vector<refusal> cases;
        auto add = [&](const char * name, const char * want, const std::function<void(gfile &)> & f, int32_t n_layer = 4) {
            gfile c;
            f(c);
            cases.push_back({ name, c, want, n_layer });
        };
        add("width", "wide", [](gfile & c) { c.key_embd = 16; });
        add("books", "books a floor", [](gfile & c) { c.key_expert = 8; });
        add("offset", "floors down", [](gfile & c) { c.offset = 1; });
        add("floors past the library", "do not fit", [](gfile & c) { c.last = 4; }, 4);
        add("floors below two", "do not fit", [](gfile & c) { c.first = 1; });
        add("tensor shape", "wanted", [](gfile & c) { c.n_embd = 16; });
        add("weight type", "wanted", [](gfile & c) { c.wtype = GGML_TYPE_Q8_0; c.n_embd = 32; c.key_embd = 8; });
        add("bias type", "wanted", [](gfile & c) { c.btype = GGML_TYPE_F16; });
        add("missing bias", "no blk.3.la_guess.bias", [](gfile & c) { c.bias_of_last = false; });
        add("no keys", "no lookahead.floor_offset", [](gfile & c) { c.keys = false; });
        for (const auto & rc : cases) {
            const std::string path = write_file(rc.name, rc.c, w, b);
            auto g = load(path, err, rc.n_layer);
            t.assert_true(std::string(rc.name) + ": refused, saying why (" + err + ")",
                    g == nullptr && err.find(rc.want) != std::string::npos);
            std::filesystem::remove(path);
        }

        auto g = load("/nonexistent/guesser.gguf", err);
        t.assert_true("a missing file is refused (" + err + ")", g == nullptr && !err.empty());

        const std::string path = write_file("short", gfile(), w, b);
        // past the 32-byte padding after the last tensor (a padded file still holds every byte), into its data
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 40);
        g = load(path, err);
        t.assert_true("a file cut short is refused (" + err + ")", g == nullptr && !err.empty());
        std::filesystem::remove(path);
    });

    t.test("attach", [&](testing & t) {
        std::vector<std::vector<uint8_t>> w, b;
        const std::vector<const ggml_tensor *> routers(4, nullptr); // no device: the CPU buffer
        gfile c;
        c.last = 2; // floor 2 alone: floor 0 guesses with it, floor 1 (two up: 3) has none
        const std::string path = write_file("attach", c, w, b);

        setenv("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER", path.c_str(), 1);
        setenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF", "0.2", 1);
        {
            floors f(4);
            ggml_tensor router_bias {}; // floor 2's router bias, as a library with one would set it
            f.sl(0)->la->bias_src2 = &router_bias;
            llama_moe_stream_guesser_attach(*f.mgr, routers, 8, 4, false);
            t.assert_true("the manager holds the guesser", f.mgr->guesser != nullptr);
            auto * s0 = f.sl(0);
            t.assert_true("floor 0 guesses floor 2 with floor 2's guesser", s0->la_guess == f.mgr->guesser->weight[2] &&
                    s0->la_guess_b == f.mgr->guesser->bias[2] && s0->la->guessed && s0->la->sl_next2 == f.sl(2));
            t.assert_true("with the cut and the file's rule", s0->la->conf_cut == 0.2f && s0->la->conf_scale == 10.0f &&
                    s0->la->conf_max == 1.0f && s0->la->bias_src2 == nullptr);
            auto * s1 = f.sl(1);
            t.assert_true("floor 1 (floor 3 has no guess in the file) guesses nothing two ahead",
                    s1->la->sl_next2 == nullptr && s1->la->top_k2 == 0 && s1->la_gate_inp2 == nullptr && s1->la_guess == nullptr);
            t.assert_true("its one-floor guess is untouched", s1->la->sl_next == f.sl(2) && !s1->la->guessed);
        }
        unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF");
        {
            floors f(4);
            llama_moe_stream_guesser_attach(*f.mgr, routers, 8, 4, false);
            t.assert_true("CONF unset beside the guesser: the everyday cut, 0.2 (since 2026-10-15)",
                    f.sl(0)->la->guessed && f.sl(0)->la->conf_cut == 0.2f);
        }
        setenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF", "0", 1);
        {
            floors f(4);
            llama_moe_stream_guesser_attach(*f.mgr, routers, 8, 4, false);
            t.assert_true("CONF \"0\": the guesser, ranked by z, top_k2 books", f.sl(0)->la->guessed &&
                    f.sl(0)->la->conf_cut == 0.0f && f.sl(0)->la->top_k2 == 4);
        }
        unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF");
        {
            floors f(2); // the lookahead is on, one floor ahead only (DEPTH2 unset): no floor has a two-floor guess
            t.assert_true("a guesser serving no floor refuses",
                    throws([&] { llama_moe_stream_guesser_attach(*f.mgr, routers, 8, 4, false); }, "serves no floor"));
        }
        {
            floors f(1); // the apprentice's sidecar: one streamed floor, so no lookahead at all, opened with the
                         // library's switches (common_init_from_params passes them on)
            llama_moe_stream_guesser_attach(*f.mgr, routers, 8, 4, false);
            t.assert_true("a model with no lookahead loads untouched, with nothing attached (the apprentice beside the "
                    "trained guess)", f.mgr->guesser == nullptr && f.sl(0)->la == nullptr && f.sl(0)->la_guess == nullptr);
        }
        {
            floors f(4, /*gpu_slot =*/ 3);
            t.assert_true("the GPU slot servicer refuses",
                    throws([&] { llama_moe_stream_guesser_attach(*f.mgr, routers, 8, 4, false); }, "GPU_SLOT"));
        }
        {
            floors f(4);
            llama_moe_stream_guesser_attach(*f.mgr, routers, 8, 4, /*no_alloc =*/ true);
            t.assert_true("sizing only: nothing loaded", f.mgr->guesser == nullptr && !f.sl(0)->la->guessed);
        }
        {
            floors f(4);
            t.assert_true("a file that does not fit refuses, naming it",
                    throws([&] { llama_moe_stream_guesser_attach(*f.mgr, routers, 16, 4, false); }, path.c_str()));
        }
        unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER");
        setenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF", "0.2", 1);
        {
            floors f(4);
            t.assert_true("CONF without GUESSER refuses",
                    throws([&] { llama_moe_stream_guesser_attach(*f.mgr, routers, 8, 4, false); }, "set LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER"));
        }
        unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF");
        {
            floors f(4);
            llama_moe_stream_guesser_attach(*f.mgr, routers, 8, 4, false);
            t.assert_true("both unset: today's two-floor guess, untouched", f.mgr->guesser == nullptr &&
                    !f.sl(0)->la->guessed && f.sl(0)->la_guess == nullptr && f.sl(1)->la->sl_next2 == f.sl(3));
        }
        std::filesystem::remove(path);
    });

    return t.summary();
}

// The trained one-floor guess (LLAMA_MOE_STREAM_GUESSER1, #43, src/llama-moe-stream-guesser.h), without a model:
//
//   switch   GUESSER1 unset, "" and "0" are today's borrowed guess; anything else is the file's path
//   load     a file at floor offset 1 (floors 1..) loads bit for bit; a two-floor file (offset 2) is refused as a
//            one-floor guesser and a one-floor file as a two-floor one; a floor 0 in it is refused
//   attach   on a book manager made by hand: each floor's one-floor guess gets floor L+1's guesser and its raw
//            ranking, and drops the router bias; a floor whose next floor has no guess in the file keeps the
//            borrowed guess; the two-floor guesser loads beside it, each on its own fields; the apprentice's
//            sidecar shape (one streamed floor, no lookahead) loads untouched with both switches set; the GPU slot
//            servicer, a guesser serving no floor and a file that does not fit refuse; the startup line names the
//            one-floor guess in use, trained or borrowed, and its words after "moe stream:" never start with
//            "room" or "drives" (runner/roomstats.py keys on those)
//   pick     the trained guess's top_k books by z as they are; the borrowed guess's by sqrt(softplus) plus the router
//            bias; with LLAMA_MOE_STREAM_LOOKAHEAD_ALL a check's every word, rank by rank, the last word first;
//            only the trained pick counts a stop

#include "testing.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include "../src/llama-moe-stream.h"
#include "../src/llama-moe-stream-guesser.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

std::string g_log; // the engine's log lines

// a guesser file, 8 wide with 4 books a floor, floors first..last at `offset`; random bytes, kept for the bit check
std::string write_file(const std::string & name, uint32_t offset, uint32_t first, uint32_t last,
        std::vector<std::vector<uint8_t>> & weights, int64_t key_embd = 8) {
    const std::string path = (std::filesystem::temp_directory_path() /
            ("test-moe-stream-guesser1-" + std::to_string(getpid()) + "-" + name + ".gguf")).string();
    gguf_context * g = gguf_init_empty();
    gguf_set_val_u32(g, "lookahead.floor_offset", offset);
    gguf_set_val_u32(g, "lookahead.first_floor", first);
    gguf_set_val_u32(g, "lookahead.last_floor", last);
    gguf_set_val_u32(g, "lookahead.n_embd", (uint32_t) key_embd);
    gguf_set_val_u32(g, "lookahead.n_expert", 4);
    gguf_set_val_f32(g, "lookahead.confidence.scale", 10.0f);
    gguf_set_val_f32(g, "lookahead.confidence.max", 1.0f);
    ggml_init_params ip = { 64*ggml_tensor_overhead() + (size_t) (last + 1)*(8*4*2 + 4*4 + 256), nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(first*7 + last*3 + offset);
    std::uniform_int_distribution<int> byte(0, 255);
    weights.assign(last + 1, {});
    for (uint32_t n = first; n <= last; n++) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 4);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_name(w, ("blk." + std::to_string(n) + ".la_guess.weight").c_str());
        ggml_set_name(b, ("blk." + std::to_string(n) + ".la_guess.bias").c_str());
        for (ggml_tensor * t : { w, b }) {
            for (size_t i = 0; i < ggml_nbytes(t); i++) {
                ((uint8_t *) t->data)[i] = (uint8_t) byte(rng);
            }
        }
        weights[n].assign((uint8_t *) w->data, (uint8_t *) w->data + ggml_nbytes(w));
        gguf_add_tensor(g, w);
        gguf_add_tensor(g, b);
    }
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

std::unique_ptr<llama_moe_stream_guesser> load(const std::string & path, uint32_t gap, std::string & err) {
    err.clear();
    return llama_moe_stream_guesser_load(path.c_str(), 8, 4, 4, [](int32_t) { return ggml_backend_cpu_buffer_type(); },
            err, gap);
}

std::vector<uint8_t> bytes_of(const ggml_tensor * t) {
    std::vector<uint8_t> v(ggml_nbytes(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size());
    return v;
}

// a book manager of n_layer floors, 4 books each, with the lookahead set up as llama-model.cpp sets it: one floor
// ahead from every floor that has a next one (the borrowed router stands in as a non-null pointer, never read
// here), two floors ahead where there is a floor two up
ggml_tensor g_router_bias; // floor L+1's router bias, as a library with one sets it

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
            sl->la_gate_inp = (ggml_tensor *) 0x1;
            sl->la = new llama_moe_stream_lookahead();
            sl->la->sl = sl;
            sl->la->sl_next = mgr->layers[il + 1].get();
            sl->la->top_k = 2;
            sl->la->bias_src = &g_router_bias;
            if (il + 2 < n_layer) {
                sl->la->sl_next2 = mgr->layers[il + 2].get();
                sl->la->top_k2 = 4;
                sl->la_gate_inp2 = (ggml_tensor *) 0x1;
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

void set(const char * name, const char * v) {
    v ? setenv(name, v, 1) : unsetenv(name);
}

const std::vector<const ggml_tensor *> g_routers(4, nullptr); // no device: the CPU buffer

void attach_both(llama_moe_stream & ms, bool no_alloc = false) {
    llama_moe_stream_guesser_attach(ms, g_routers, 8, 4, no_alloc);
    llama_moe_stream_guesser1_attach(ms, g_routers, 8, 4, no_alloc);
}

} // namespace

int main(int argc, char ** argv) {
    testing t;
    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    t.verbose = verbose && std::string(verbose) == "1";
    llama_log_set([](ggml_log_level, const char * text, void *) { g_log += text; }, nullptr);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    ggml_backend_load_all();
    for (const char * e : { "LLAMA_MOE_STREAM_GUESSER1", "LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER", "LLAMA_MOE_STREAM_LOOKAHEAD_CONF" }) {
        unsetenv(e);
    }
    std::vector<std::vector<uint8_t>> w1, w2;
    const std::string one = write_file("one", 1, 1, 3, w1);      // floors 1..3: every floor's one-floor guess
    const std::string two = write_file("two", 2, 2, 3, w2);      // the two-floor guesser's file

    t.test("switch", [&](testing & t) {
        for (const char * off : { (const char *) nullptr, "", "0" }) {
            set("LLAMA_MOE_STREAM_GUESSER1", off);
            t.assert_true(std::string("\"") + (off ? off : "unset") + "\" is the borrowed guess", llama_moe_stream_guesser1_env() == nullptr);
        }
        set("LLAMA_MOE_STREAM_GUESSER1", one.c_str());
        t.assert_true("a path is the file", llama_moe_stream_guesser1_env() != nullptr && one == llama_moe_stream_guesser1_env());
        unsetenv("LLAMA_MOE_STREAM_GUESSER1");
    });

    t.test("load", [&](testing & t) {
        std::string err;
        auto g = load(one, 1, err);
        t.assert_true("a one-floor file loads at gap 1 (" + err + ")", g != nullptr);
        if (g) {
            t.assert_true("floors 1..3, nothing at floor 0", g->first == 1 && g->last == 3 && g->weight[0] == nullptr);
            t.assert_true("every weight bit for bit", bytes_of(g->weight[1]) == w1[1] && bytes_of(g->weight[3]) == w1[3]);
        }
        g = load(two, 1, err);
        t.assert_true("a two-floor file is refused as the one-floor guesser (" + err + ")",
                g == nullptr && err.find("2 floors down; the one-floor guess reads 1") != std::string::npos);
        g = load(one, 2, err);
        t.assert_true("a one-floor file is refused as the two-floor guesser (" + err + ")",
                g == nullptr && err.find("1 floors down; the two-floor guess reads 2") != std::string::npos);
        std::vector<std::vector<uint8_t>> w0;
        const std::string zero = write_file("zero", 1, 0, 3, w0);
        g = load(zero, 1, err);
        t.assert_true("a floor 0 in a one-floor file is refused (" + err + ")",
                g == nullptr && err.find("do not fit the library's one-floor guesses (floors 1..3)") != std::string::npos);
        std::filesystem::remove(zero);
    });

    t.test("attach", [&](testing & t) {
        set("LLAMA_MOE_STREAM_GUESSER1", one.c_str());
        {
            floors f(4);
            g_log.clear();
            attach_both(*f.mgr);
            t.assert_true("the manager holds the one-floor guesser, not the two-floor one",
                    f.mgr->guesser1 != nullptr && f.mgr->guesser == nullptr);
            bool all = true;
            for (int il = 0; il < 3; il++) {
                auto * s = f.sl(il);
                all = all && s->la_guess1 == f.mgr->guesser1->weight[il + 1] && s->la_guess1_b == f.mgr->guesser1->bias[il + 1] &&
                        s->la->guessed1 && s->la->bias_src == nullptr && s->la->sl_next == f.sl(il + 1) && s->la->top_k == 2;
            }
            t.assert_true("floors 0..2 guess floor L+1 with its guesser, raw, no router bias, the same width", all);
            t.assert_true("the two-floor guess is untouched", !f.sl(0)->la->guessed && f.sl(0)->la_guess == nullptr &&
                    f.sl(0)->la->sl_next2 == f.sl(2) && f.sl(0)->la_gate_inp2 != nullptr);
            t.assert_true("the startup line names the trained guess", g_log.find("moe stream: one-floor guess: trained, the guesser " + one +
                    " on 3 floors (0 with none in it keep the borrowed guess)") != std::string::npos);
        }
        std::vector<std::vector<uint8_t>> w;
        const std::string part = write_file("part", 1, 1, 2, w); // floors 1..2: floor 2 (guessing 3) has none
        set("LLAMA_MOE_STREAM_GUESSER1", part.c_str());
        {
            floors f(4);
            attach_both(*f.mgr);
            auto * s2 = f.sl(2);
            t.assert_true("floor 2, whose next floor has no guess in the file, keeps the borrowed guess",
                    s2->la_guess1 == nullptr && !s2->la->guessed1 && s2->la->bias_src == &g_router_bias &&
                    s2->la->sl_next == f.sl(3) && s2->la->top_k == 2 && s2->la_gate_inp != nullptr);
            t.assert_true("floors 0 and 1 are trained", f.sl(0)->la->guessed1 && f.sl(1)->la->guessed1);
        }
        std::filesystem::remove(part);

        set("LLAMA_MOE_STREAM_GUESSER1", one.c_str());
        set("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER", two.c_str());
        set("LLAMA_MOE_STREAM_LOOKAHEAD_CONF", "0.2");
        {
            floors f(4);
            attach_both(*f.mgr);
            auto * s0 = f.sl(0);
            t.assert_true("both guessers load, each its own file", f.mgr->guesser && f.mgr->guesser1 && f.mgr->guesser->first == 2 &&
                    f.mgr->guesser1->first == 1);
            t.assert_true("floor 0: floor 1's one-floor guesser and floor 2's two-floor guesser, side by side",
                    s0->la_guess1 == f.mgr->guesser1->weight[1] && s0->la_guess == f.mgr->guesser->weight[2] &&
                    s0->la->guessed1 && s0->la->guessed && s0->la->conf_cut == 0.2f && s0->la->bias_src == nullptr);
        }
        {
            floors f(1); // the apprentice's sidecar: one streamed floor, no lookahead, opened with the library's switches
            g_log.clear();
            attach_both(*f.mgr);
            t.assert_true("the sidecar shape loads untouched with both guessers set", f.mgr->guesser == nullptr &&
                    f.mgr->guesser1 == nullptr && f.sl(0)->la == nullptr && f.sl(0)->la_guess1 == nullptr);
            t.assert_true("and says the one-floor guesser is not loaded there",
                    g_log.find("one-floor guess: the trained guesser " + one + " is not loaded here") != std::string::npos);
        }
        unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER");
        unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD_CONF");
        {
            floors f(4, /*gpu_slot =*/ 3);
            t.assert_true("the GPU slot servicer refuses", throws([&] { attach_both(*f.mgr); }, "GPU_SLOT"));
        }
        {
            floors f(4);
            attach_both(*f.mgr, /*no_alloc =*/ true);
            t.assert_true("sizing only: nothing loaded", f.mgr->guesser1 == nullptr && !f.sl(0)->la->guessed1);
        }
        {
            floors f(4);
            t.assert_true("a file that does not fit refuses, naming it and the switch",
                    throws([&] { llama_moe_stream_guesser1_attach(*f.mgr, g_routers, 16, 4, false); },
                            ("one-floor guesser " + one + " (LLAMA_MOE_STREAM_GUESSER1)").c_str()));
        }
        set("LLAMA_MOE_STREAM_GUESSER1", two.c_str());
        {
            floors f(4);
            t.assert_true("the two-floor file in GUESSER1 refuses", throws([&] { attach_both(*f.mgr); }, "the one-floor guess reads 1"));
        }
        std::vector<std::vector<uint8_t>> w3;
        const std::string top = write_file("top", 1, 3, 3, w3); // floor 3 alone: floors(3)'s guesses go to 1 and 2
        set("LLAMA_MOE_STREAM_GUESSER1", top.c_str());
        {
            floors f(4);
            attach_both(*f.mgr);
            t.assert_true("floor 3 alone: floor 2 trained, floors 0 and 1 borrowed", f.sl(2)->la->guessed1 &&
                    !f.sl(0)->la->guessed1 && !f.sl(1)->la->guessed1);
        }
        {
            floors f(3);
            t.assert_true("a guesser serving no floor refuses", throws([&] { attach_both(*f.mgr); }, "serves no floor"));
        }
        std::filesystem::remove(top);
        for (const char * off : { (const char *) nullptr, "0" }) {
            set("LLAMA_MOE_STREAM_GUESSER1", off);
            floors f(4);
            g_log.clear();
            attach_both(*f.mgr);
            t.assert_true(std::string(off ? "\"0\"" : "unset") + ": today's borrowed guess, untouched", f.mgr->guesser1 == nullptr &&
                    !f.sl(0)->la->guessed1 && f.sl(0)->la_guess1 == nullptr && f.sl(0)->la->bias_src == &g_router_bias);
            t.assert_true("and the startup line says so",
                    g_log.find("moe stream: one-floor guess: borrowed, each floor's next floor's own router") != std::string::npos);
        }
        unsetenv("LLAMA_MOE_STREAM_GUESSER1");
        {
            floors f(1);
            g_log.clear();
            attach_both(*f.mgr);
            t.assert_true("unset on the sidecar shape: no line (nothing guesses ahead there)", g_log.find("one-floor guess") == std::string::npos);
        }
        // every line it wrote: after "moe stream: ", never "room" or "drives"
        bool clean = true;
        for (size_t at = g_log.find("moe stream: "); at != std::string::npos; at = g_log.find("moe stream: ", at + 1)) {
            clean = clean && g_log.compare(at + 12, 4, "room") != 0 && g_log.compare(at + 12, 6, "drives") != 0;
        }
        t.assert_true("no line it writes starts with \"room\" or \"drives\" after \"moe stream:\"", clean);
    });

    t.test("pick", [&](testing & t) {
        floors f(3);
        auto * la = f.sl(0)->la;
        la->bias.assign({ 0.0f, 0.0f, 0.0f, 5.0f }); // the router's bias, as read at the first stop
        la->bias_read = true;
        la->all = true;
        // one word: z ranks 0, 1, 2, 3; the router's bias lifts book 3 to the top
        const std::vector<float> word = { 3.0f, 2.0f, 1.0f, -10.0f };
        std::vector<int32_t> books;
        auto & st = f.mgr->stats;
        llama_moe_stream_lookahead_pick1(la, word.data(), 1, 4, books);
        t.assert_true("borrowed: sqrt(softplus) plus the router bias: books 3, 0", books == std::vector<int32_t>({ 3, 0 }));
        t.assert_equal("the borrowed pick counts no trained stop", (int64_t) 0, st.n_guess1_stops);
        la->guessed1 = true;
        llama_moe_stream_lookahead_pick1(la, word.data(), 1, 4, books);
        t.assert_true("trained: z as it is, no bias: books 0, 1", books == std::vector<int32_t>({ 0, 1 }));
        t.assert_equal("and counts one stop", (int64_t) 1, st.n_guess1_stops);
        // two words (a check): every word's books rank by rank, the last word first, each book once
        const std::vector<float> rows = { -1.0f, 4.0f, 3.0f, -2.0f,     // the earlier word: books 1, 2
                                          2.0f, -3.0f, -1.0f, 1.0f };   // the last word: books 0, 3
        llama_moe_stream_lookahead_pick1(la, rows.data(), 2, 4, books);
        t.assert_true("ALL: rank 1 of the last word, then of the earlier, then rank 2: 0, 1, 3, 2",
                books == std::vector<int32_t>({ 0, 1, 3, 2 }));
        la->all = false;
        llama_moe_stream_lookahead_pick1(la, rows.data(), 2, 4, books);
        t.assert_true("without ALL: the last word's alone: 0, 3", books == std::vector<int32_t>({ 0, 3 }));
        t.assert_equal("three trained stops", (int64_t) 3, st.n_guess1_stops);
        // a wider row (the depth-2 concat, [n1 + n2] a word): only its own n1 cells are read
        la->all = true;
        const std::vector<float> wide = { -1.0f, 4.0f, 3.0f, -2.0f, 99.0f, 99.0f, 99.0f, 99.0f,
                                          2.0f, -3.0f, -1.0f, 1.0f, 99.0f, 99.0f, 99.0f, 99.0f };
        llama_moe_stream_lookahead_pick1(la, wide.data(), 2, 8, books);
        t.assert_true("rows of the concat read their first n1 cells: 0, 1, 3, 2", books == std::vector<int32_t>({ 0, 1, 3, 2 }));
    });

    std::filesystem::remove(one);
    std::filesystem::remove(two);
    return t.summary();
}

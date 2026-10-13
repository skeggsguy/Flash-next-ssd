#pragma once

// The trained two-floor guesser (#30's C1, the study's KEEP-C1-PLAN.md step 5).
//
// LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER=<gguf>: floor L's two-floor guess (LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2) stops
// borrowing floor L+2's own slip-writer (its router, ffn_gate_inp) and uses a straight line trained for the job:
// for each floor n it scores n's books from floor n-2's router input, z = W[n] . cur + b[n]. The file (the
// study's lookahead/guesser_gguf.py) holds, for its floors n,
//     blk.{n}.la_guess.weight   [n_embd, n_expert], F16 (F32 is taken too): the layout of ffn_gate_inp
//     blk.{n}.la_guess.bias     [n_expert], F32
// and lookahead.* keys: the floor offset (2), the first and last floor, n_embd, n_expert and the confidence rule
// (confidence.scale, confidence.max). Every key and shape is checked against the library at load, and a file that
// does not fit refuses to start. The weights sit beside the library, outside the desk (the library's 46 floors:
// ~121 MB), each floor's in the buffer of the router it stands in for. A floor whose floor two up has no guess in
// the file guesses nothing two floors ahead.
//
// LLAMA_MOE_STREAM_LOOKAHEAD_CONF=<cut>: which books the guess fetches, by confidence. A word's confidence in a
// book is min(max, scale * softmax(z)) over all its floor's books (1 and 10: each of a word's ten slips is a
// tenth of what the guesser was trained to put there). Every word of the check (with
// LLAMA_MOE_STREAM_LOOKAHEAD_ALL; else the batch's last word), the last word first, offers its 24 best books; a
// book is fetched when its confidence reaches the cut in at least one word, ranked by its highest confidence,
// most confident first (ties: the last word's first, then by rank), and the issue loop caps the list as it
// always has (the backlog, the slots a guess may take). Unset, "" or "0": each word's top_k2 books by z, rank by
// rank, as the untrained guess picks them.
//
// Unset GUESSER: today's graph and pick exactly (CONF alone refuses to start: it reads the guesser's scores). A model
// with no lookahead at all (the apprentice's sidecar: one streamed floor, opened with the library's switches) loads
// with GUESSER set and nothing attached; a library whose lookahead guesses one floor only (DEPTH2 unset) refuses.
// Words never change either way: a guess only says which books are fetched, and when.
//
// The trained one-floor guess (#43, the study's TRIPS-AND-LOOKUP-PLAN.md step 3; llama-moe-stream-guesser1.cpp):
// LLAMA_MOE_STREAM_GUESSER1=<gguf> is the same kind of file at floor offset 1 (floors 1..), and floor L's one-floor
// guess (the lookahead, on in every rung) scores floor L+1's books with it, z = W[L+1] . cur + b[L+1], in place of
// borrowing L+1's own router; its top_k books by z, ranked as they are (no router transform, no router bias), the
// same width as the borrowed guess and no confidence cut. Unset, "" or "0": today's borrowed guess exactly. A floor
// whose next floor has no guess in the file keeps the borrowed guess. It loads beside the two-floor guesser (both
// may be on), and like it leaves a model with no lookahead (the apprentice's sidecar) untouched. One startup line
// names the one-floor guess in use, trained or borrowed.

#include "ggml-cpp.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct llama_moe_stream;
struct llama_moe_stream_lookahead;

// books each word offers the confidence pick (the study's training saved and the sim priced each word's 24 best)
static const uint32_t LLAMA_MOE_STREAM_GUESS_PER_WORD = 24;

struct llama_moe_stream_guesser {
    std::string source;           // the file, for the startup line
    int32_t     first = -1;       // the floors with a guess, first..last
    int32_t     last  = -1;
    float       conf_scale = 10.0f;
    float       conf_max   = 1.0f;

    std::vector<ggml_tensor *> weight; // [n_layer], null outside first..last
    std::vector<ggml_tensor *> bias;   // [n_layer]

    std::vector<ggml_context_ptr>        ctxs; // one per buffer type the floors' routers live in
    std::vector<ggml_backend_buffer_ptr> bufs;

    size_t nbytes() const;
};

// LLAMA_MOE_STREAM_LOOKAHEAD_GUESSER: the file's path, or null (unset or "")
const char * llama_moe_stream_guesser_env();

// LLAMA_MOE_STREAM_GUESSER1: the one-floor file's path, or null (unset, "" or "0")
const char * llama_moe_stream_guesser1_env();

// LLAMA_MOE_STREAM_LOOKAHEAD_CONF: the cut, or 0 for the fixed pick (unset, "" or "0"). Throws on anything that is
// not a number in (0, 1], so a mistyped arm never runs on the fixed pick.
float llama_moe_stream_lookahead_conf_env();

// reads a guesser file for a library of n_layer floors, n_embd wide, n_expert books a floor, whose guess reads the
// input `gap` floors down (2: the two-floor guesser, 1: the one-floor); floor n's tensors go in buft_of(n). Null
// with err set when the file cannot be read or does not fit.
std::unique_ptr<llama_moe_stream_guesser> llama_moe_stream_guesser_load(const char * path, int64_t n_embd,
        int64_t n_expert, int32_t n_layer, const std::function<ggml_backend_buffer_type_t(int32_t)> & buft_of,
        std::string & err, uint32_t gap = 2);

// where floor n's guesser lives: its router's device (routers[n] is floor n's ffn_gate_inp; the mtp word list's
// rule, llama-mtp-vocab.cpp), the CPU when the router has none
std::function<ggml_backend_buffer_type_t(int32_t)> llama_moe_stream_guesser_buft_of(
        const std::vector<const ggml_tensor *> & routers);

// At model load, after the lookahead is set up: with GUESSER set, loads the file and gives each floor's
// two-floor guess its floor's guesser and the CONF cut (one startup line); routers[n] is floor n's ffn_gate_inp,
// whose device floor n's guesser shares. Unset, nothing. Throws when the switches or the file cannot be used.
void llama_moe_stream_guesser_attach(llama_moe_stream & ms, const std::vector<const ggml_tensor *> & routers,
        int64_t n_embd, int64_t n_expert, bool no_alloc);

// The same for the one-floor guess (GUESSER1): each floor's lookahead gets its next floor's guesser, ranked raw,
// and the startup line names the one-floor guess in use. Throws when the switch or the file cannot be used.
void llama_moe_stream_guesser1_attach(llama_moe_stream & ms, const std::vector<const ggml_tensor *> & routers,
        int64_t n_embd, int64_t n_expert, bool no_alloc);

// The one-floor pick (llama-moe-stream-lookahead.cpp): rows[t*stride + e] is word t's next-floor score for book e;
// with LLAMA_MOE_STREAM_LOOKAHEAD_ALL a small batch (2..16 words) offers every word's books rank by rank, the last
// word first, else the last word's alone; top_k books, the trained guess's z ranked raw (la->guessed1), the
// borrowed router's by sqrt(softplus) plus its bias.
void llama_moe_stream_lookahead_pick1(struct llama_moe_stream_lookahead * la, const float * rows, int64_t n_tok,
        int64_t stride, std::vector<int32_t> & books);

// the confidence pick's scratch, one per floor's lookahead (graph thread only)
struct llama_moe_stream_guess_scratch {
    std::vector<float>   best;  // [n] a book's highest confidence over the words so far, -1 = not over the cut
    std::vector<uint8_t> under; // [n] in some word's best, under the cut
    std::vector<int32_t> order; // [n] book ids, sorted by z
};

// The pick by confidence (above): rows[t*stride + e] is word t's z for book e (n books), words t_first..n_tok-1.
// Fills books, most confident first; adds to n_under the books in some word's per_word best that never reached
// the cut.
void llama_moe_stream_guess_pick(const float * rows, int64_t stride, int64_t t_first, int64_t n_tok, uint32_t n,
        float cut, float scale, float conf_max, uint32_t per_word, std::vector<int32_t> & books,
        llama_moe_stream_guess_scratch & s, int64_t & n_under);

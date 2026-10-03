#pragma once

// The apprentice's word list (WRITING-PLAN.md step 4b): its guesses come from a trimmed list of words.
//
// The apprentice (the MTP draft head) ends every guess with the library's own head: a matrix with one row
// per word of the vocabulary (248,320 at Q8_0, ~0.63 GiB), multiplied by the guess's hidden state to score
// every word. Most of those words never come up in a guess worth making. With a list (step 4a's
// frequency list, ~/models/flashnext/mtp-vocab/list-96k.txt), the listed rows of the head are copied once,
// when the apprentice's context is made, into a buffer of its own beside the head (on the head's device,
// in the head's type), and each guess multiplies only those rows. Their scores are scattered back into a
// full row of the vocabulary, every unlisted word at -INF (the d2t precedent: src/models/eagle3.cpp,
// dflash.cpp), so every sampler and every caller still sees n_vocab scores.
//
// Exact where it matters: a listed word's score is the same dot product of the same bytes the whole head
// computes (a matrix's rows are independent), so it is byte-identical to the whole head's; whenever the
// whole head's top word is listed, the guess is the same word. The library checks every guess, so the
// words written never change; a guess whose top word is not listed is a different (usually wrong) guess,
// and the confidence (p-min) is now over the listed words only, so guess counts can move (the depth-change
// class). The copy costs its rows' bytes: 98,304 rows of 2,560 at Q8_0 are 255 MiB.
//
// LLAMA_MTP_VOCAB (read when the context is made, so per context; only an MTP context of an arch whose
// MTP graph takes the list, qwen4exp's graph_mtp): unset, empty or "0" is today's graph exactly (the
// whole head), anything else is the path of a list file: one word id per line (blank lines and lines
// starting with '#' are skipped), any order, each id below n_vocab. A file that cannot be read or holds a
// bad id fails the context, so a mistyped path never runs silently on the whole head.

#include "llama-arch.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct llama_context;
struct llama_model;

struct llama_mtp_vocab {
    int64_t              n_vocab = 0; // the full row the scores are scattered into
    std::vector<int64_t> ids;         // the listed word ids, ascending, no repeats
    std::string          source;      // where the list came from (the file's name), for the startup line

    ggml_tensor * weight = nullptr;   // [n_embd, ids.size()] the listed rows of the head, in its type
    ggml_tensor * rows   = nullptr;   // I64 [ids.size()] each row's word id (= ids)

    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;

    size_t nbytes() const;
};

// LLAMA_MTP_VOCAB's value (nullptr: unset) as wanted or not: a list path, or nothing (unset, "", "0")
bool llama_mtp_vocab_wanted(const char * value);

// the archs whose MTP graph takes the list (qwen4exp's graph_mtp)
bool llama_mtp_vocab_supported(llm_arch arch);

// parses a list (the file's text): fills ids (ascending, repeats dropped) or returns false with err set
bool llama_mtp_vocab_parse(const std::string & text, int64_t n_vocab, std::vector<int64_t> & ids, std::string & err);

// copies head's listed rows into a new buffer on head's device, in head's type. Returns null with err set
// when head cannot be gathered (no data, rows not contiguous, or rows the CPU has repacked)
std::unique_ptr<llama_mtp_vocab> llama_mtp_vocab_gather(const ggml_tensor * head, const std::vector<int64_t> & ids,
        std::string & err);

// the context's list: null (today's graph) unless an MTP context of a supported arch asks for one. Finds the
// head as graph_mtp does (the model's own output, else the target's through ctx_other), gathers, and says
// which it runs in one startup line; other contexts say nothing. Throws when the list cannot be used
std::unique_ptr<llama_mtp_vocab> llama_mtp_vocab_init(const char * value, bool is_mtp_ctx,
        const llama_model & model, const llama_context * ctx_other);

// the graph: scores [n_vocab, n_outputs] from the listed rows only, every other word -INF. cur is the
// head's input [n_embd, n_outputs]
ggml_tensor * llama_mtp_vocab_logits(ggml_context * ctx0, const llama_mtp_vocab & v, ggml_tensor * cur);

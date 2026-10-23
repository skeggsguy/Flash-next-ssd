#pragma once

// The p/q record's test run (test-speculative-pq-record-mtp.cpp): one generation on the MTP fixture, driven as the
// server drives speculation, with the record on or off, and the record's lines read back.

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "speculative-dflash-walk.h"
#include "speculative-pq-record.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace pq_test {

inline std::string g_path;
inline int  g_ngl      = 99;
inline int  g_failures = 0;
inline float g_p_min   = 0.0f; // the MTP draft's p-min (0: it drafts to its cap)

constexpr int N_MAX     = 6;
constexpr int N_PREDICT = 160;
constexpr int K         = 4; // the made-up lattice's candidates
constexpr int N_POS     = 3; // and drafted positions

inline void check(bool ok, const std::string & what) {
    printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

inline std::string tmp_path(const std::string & tag) {
    const char * dir = std::getenv("TMPDIR");
    return std::string(dir && *dir ? dir : "/tmp") + "/test-spec-pq-" + tag + "-" + std::to_string((long) getpid());
}

// the balanced value after "key": (an array or object), or the scalar up to , or }
inline std::string field(const std::string & s, const std::string & key) {
    const size_t at = s.find("\"" + key + "\":");
    if (at == std::string::npos) {
        return "";
    }
    const size_t beg = at + key.size() + 3;
    if (s[beg] != '[' && s[beg] != '{') {
        return s.substr(beg, s.find_first_of(",}", beg) - beg);
    }
    int depth = 0;
    for (size_t i = beg; i < s.size(); i++) {
        depth += (s[i] == '[' || s[i] == '{') - (s[i] == ']' || s[i] == '}');
        if (depth == 0) {
            return s.substr(beg, i - beg + 1);
        }
    }
    return "";
}

inline std::vector<double> nums(const std::string & flat) {
    std::vector<double> out;
    for (size_t i = 0; i < flat.size();) {
        if (std::strchr("-0123456789", flat[i])) {
            char * end = nullptr;
            out.push_back(std::strtod(flat.c_str() + i, &end));
            i = end - flat.c_str();
        } else {
            i++;
        }
    }
    return out;
}

// [[..],[..]] -> each inner array's numbers
inline std::vector<std::vector<double>> rows(const std::string & nested) {
    std::vector<std::vector<double>> out;
    for (size_t i = 1; i + 1 < nested.size(); i++) {
        if (nested[i] == '[') {
            const size_t end = nested.find(']', i);
            out.push_back(nums(nested.substr(i, end - i + 1)));
            i = end;
        }
    }
    return out;
}

struct round_out {
    llama_tokens draft;
    llama_tokens ids;
};

struct run_out {
    bool                     ok = false;
    llama_tokens             text;
    std::vector<round_out>   rounds;
    std::vector<std::string> lines;      // the record's round lines
    std::string              end;        // its closing line
    std::vector<int>         trace_kept; // the rate trace's kept, a cycle each
    std::vector<std::vector<float>> lat; // the made-up lattice handed over before each check
    std::vector<llama_tokens> tail;      // the replay's and the restoring check's words, after the rounds
    int                      replaced = 0; // MTP drafts the made-up lattice took the place of (each an orphan)
};

inline std::vector<float> fake_lattice(int round, size_t stride) {
    std::vector<float> rows_(N_POS * stride, 0.0f);
    for (int i = 0; i < N_POS; i++) {
        for (int c = 0; c < K; c++) {
            rows_[i * stride + c] = (float) (1 + (round * 7 + i * 3 + c) % 100); // an id
        }
        for (int c = 0; c < K * K; c++) {
            rows_[i * stride + K + c] = 0.125f * (float) ((round + i * 5 + c * 11) % 37) - 2.0f; // a score
        }
        if (round == 0 && i == 0) {
            rows_[i * stride + K + K * K - 1] = -INFINITY; // a masked score: written as JSON can read it
        }
    }
    return rows_;
}

inline void set_env(const char * name, const char * value) {
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
}

// one generation; `tag` names the record's file ("" off); `lattice` hands a made-up lattice to its draft side (round
// 3's from the wrong position) and, after the rounds, checks the last position twice: a check that restores, then
// its replay. `oracle` (a text the same seed wrote) replaces the fixture's drafts, which the library almost never
// keeps, with that text's next words: all of them every other round, one of them wrong (the third, or the last)
// the rounds between, so checks keep words, miss after keeping some, and draw the library's own word after a full draft
inline run_out run(float temp, uint32_t seed, const llama_tokens & prompt, const std::string & tag, bool lattice,
        const llama_tokens * oracle = nullptr) {
    run_out out;
    const std::string path = tag.empty() ? "" : tmp_path(tag) + ".pqr";
    const std::string tpath = tag.empty() ? "" : tmp_path(tag) + ".srtr";
    std::remove(path.c_str());
    std::remove(tpath.c_str());
    set_env("LLAMA_SPEC_PQ_RECORD", tag.empty() ? nullptr : path.c_str());
    set_env("LLAMA_RS_SKIP_SLOTS", "0"); // every undo copy, so the tail's restore can roll a whole check back
    set_env("LLAMA_SPEC_RATE_TRACE", tag.empty() ? nullptr : tpath.c_str());

    common_params params;
    params.model.path   = g_path;
    params.n_gpu_layers = g_ngl;
    params.fit_params   = false;
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    params.warmup          = false;
    params.n_ctx           = 512;
    params.n_batch         = 512;
    params.n_ubatch        = 64;
    params.sampling.temp   = temp;
    params.sampling.top_k  = 20;
    params.sampling.top_p  = 0.95f;
    params.sampling.min_p  = 0.05f;
    params.sampling.seed   = seed;
    params.speculative.types       = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = N_MAX;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = g_p_min;
    const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel,
            common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (model == nullptr || ctx == nullptr) {
        return out;
    }
    common_params params_dft = common_base_params_to_speculative(params);
    auto spec_init = common_speculative_init_from_params(params_dft, model, ctx);
    params.speculative.draft.ctx_tgt = ctx;
    params.speculative.draft.ctx_dft = spec_init->context();
    llama_context * ctx_dft = params.speculative.draft.ctx_dft;
    if (ctx_dft == nullptr || common_context_can_seq_rm(ctx) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        fprintf(stderr, "no MTP context, or the library needs checkpoints (this test does not drive them)\n");
        return out;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    common_sampler_ptr smpl(common_sampler_init(model, params.sampling));
    common_speculative * spec = common_speculative_init(params.speculative, 1);
    common_spec_pq_record * pq = common_speculative_pq_record(spec);
    if (spec == nullptr || (pq != nullptr) == tag.empty()) {
        fprintf(stderr, "the record is %s with LLAMA_SPEC_PQ_RECORD %s\n", pq ? "on" : "off", tag.empty() ? "unset" : "set");
        return out;
    }
    if (pq) {
        pq->sampling(0, params.sampling);
    }

    const llama_seq_id seq = 0;
    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);
    for (size_t i = 0; i + 1 < prompt.size(); i++) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { seq }, false);
    }
    bool ok = llama_decode(ctx, batch) == 0 && common_speculative_process(spec, batch);
    llama_token  id_last = prompt.back();
    llama_tokens prompt_tgt(prompt.begin(), prompt.end() - 1);
    int          n_past  = (int) prompt.size() - 1;
    common_speculative_begin(spec, seq, prompt_tgt);

    const size_t stride = K + K * K + 3; // a row's floats, padded like the real lattice
    llama_tokens draft;
    while (ok && (int) out.text.size() < N_PREDICT) {
        const int  r       = (int) out.rounds.size();
        const bool no_spec = r % 5 == 4; // every fifth word is written with no draft, as the server's plain path
        const int  n_draft_max = no_spec ? 0 : std::max(0, std::min((int) llama_n_ctx(ctx) - n_past - 2,
                N_PREDICT - (int) out.text.size() - 1));
        if (n_draft_max > 0) {
            common_speculative_get_draft_params(spec, seq) = { true, n_draft_max, n_past, id_last, &prompt_tgt, &draft };
            common_speculative_draft(spec);
        }
        if (oracle != nullptr && n_draft_max > 0) {
            const size_t at = out.text.size();
            const size_t n  = std::min<size_t>(std::min(N_MAX, n_draft_max), oracle->size() - std::min(at, oracle->size()));
            draft.assign(oracle->begin() + at, oracle->begin() + at + n);
            if (r % 2 == 1 && !draft.empty()) {
                llama_token & w = draft[std::min<size_t>(2, draft.size() - 1)];
                w = 1 + w % 100; // another of the fixture's words
            }
        }
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, n_past, -1);
        if (pq && lattice) { // the draft side: from this round's anchor, but round 3's from the wrong position
            out.replaced += n_draft_max > 0; // the MTP head's own draft from this anchor, never checked
            out.lat.push_back(fake_lattice(r, stride));
            const common_dflash2_lattice lat = { out.lat.back().data(), stride, N_POS, K };
            pq->draft(seq, r == 3 ? n_past + 1 : n_past, lat, draft);
        }
        const int anchor = n_past;
        common_batch_clear(batch);
        common_batch_add(batch, id_last, n_past++, { seq }, true);
        for (size_t i = 0; i < draft.size(); i++) {
            common_batch_add(batch, draft[i], n_past + (llama_pos) i, { seq }, true);
        }
        ok = llama_decode(ctx, batch) == 0 && common_speculative_process(spec, batch);
        if (!ok) {
            break;
        }
        std::vector<int32_t> idxs(draft.size() + 1);
        for (size_t i = 0; i < idxs.size(); i++) {
            idxs[i] = (int32_t) i;
        }
        llama_tokens ids;
        if (draft.empty()) { // the server's plain path: one draw, then the record
            ids = { common_sampler_sample(smpl.get(), ctx, 0) };
            if (pq) {
                pq->plain(seq, anchor, smpl.get(), ids[0]);
            }
            common_sampler_accept(smpl.get(), ids[0], true);
        } else {
            ids = pq ? pq->accept_n(smpl.get(), ctx, idxs, draft, seq, anchor, false)
                     : common_sampler_sample_and_accept_n(smpl.get(), ctx, idxs, draft);
            if (pq) {
                pq->check_end(seq, false);
            }
            common_speculative_accept(spec, seq, (uint16_t) (ids.size() - 1));
        }
        out.rounds.push_back({ draft, ids });
        n_past += (int) ids.size() - 1;
        bool eog = false;
        for (llama_token id : ids) {
            prompt_tgt.push_back(id_last);
            id_last = id;
            out.text.push_back(id);
            if ((eog = llama_vocab_is_eog(vocab, id))) {
                break;
            }
        }
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(ctx), seq, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, n_past, -1);
        if (eog) {
            break;
        }
    }
    if (ok && pq && lattice && out.text.size() >= 2) {
        // the server's restore and replay at the last position: a check that restores (the sampler and the memory
        // put back), then the replay of the words it returned, from the same anchor
        common_sampler_ptr save(common_sampler_clone(smpl.get()));
        out.lat.push_back(fake_lattice(98, stride));
        pq->draft(seq, n_past, { out.lat.back().data(), stride, N_POS, K }, {});
        llama_tokens draft_t = { out.text[0], out.text[1] };
        for (const bool replay : { false, true }) {
            if (replay) { // a draft from the replay's own position, which the replay must leave
                out.lat.push_back(fake_lattice(99, stride));
                pq->draft(seq, n_past, { out.lat.back().data(), stride, N_POS, K }, {});
            }
            common_batch_clear(batch);
            common_batch_add(batch, id_last, n_past, { seq }, true);
            for (size_t i = 0; i < draft_t.size(); i++) {
                common_batch_add(batch, draft_t[i], n_past + 1 + (llama_pos) i, { seq }, true);
            }
            if (!(ok = llama_decode(ctx, batch) == 0)) {
                break;
            }
            std::vector<int32_t> idxs(draft_t.size() + 1);
            for (size_t i = 0; i < idxs.size(); i++) {
                idxs[i] = (int32_t) i;
            }
            out.tail.push_back(pq->accept_n(smpl.get(), ctx, idxs, draft_t, seq, n_past, replay));
            pq->check_end(seq, /* restore */ !replay);
            llama_memory_seq_rm(llama_get_memory(ctx), seq, n_past, -1);
            common_sampler_copy(save.get(), smpl.get());
            draft_t = out.tail.back(); // the replay re-checks the words the check returned
        }
    }
    out.ok = ok;
    llama_batch_free(batch);
    common_speculative_free(spec); // closes the record and the trace
    set_env("LLAMA_SPEC_PQ_RECORD", nullptr);
    set_env("LLAMA_SPEC_RATE_TRACE", nullptr);
    set_env("LLAMA_RS_SKIP_SLOTS", nullptr);
    if (!tag.empty()) {
        std::ifstream in(path);
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("{\"line\":", 0) == 0) {
                out.lines.push_back(line);
            } else if (line.rfind("{\"end\":", 0) == 0) {
                out.end = line;
            }
        }
        std::ifstream tin(tpath);
        for (std::string line; std::getline(tin, line);) {
            if (line.rfind("{\"cycle\":", 0) == 0 && std::atoi(field(line, "checked").c_str()) > 0) {
                out.trace_kept.push_back(std::atoi(field(line, "kept").c_str()));
            }
        }
        std::remove(path.c_str());
        std::remove(tpath.c_str());
    }
    return out;
}

} // namespace pq_test

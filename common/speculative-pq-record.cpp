#include "speculative-pq-record.h"
#include "speculative-pq-record-util.h"

#include "common.h"
#include "sampling.h"
#include "speculative-dflash-walk.h"
#include "speculative-lenient.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <exception>

using pq_util::add;
using pq_util::add_int;
using pq_util::add_ids;
using pq_util::steady_us;

namespace {

int64_t wall_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

void add_positions(std::string & out, const std::vector<common_spec_pq_record::position> & ps) {
    out += '[';
    for (size_t i = 0; i < ps.size(); i++) {
        const auto & q = ps[i];
        out += i > 0 ? ",[" : "[";
        add_int(out, q.drawn);   out += ',';
        add_int(out, q.g);       out += ',';
        add_int(out, q.b);       out += ',';
        add_int(out, q.refused); out += ',';
        add_int(out, q.size);
        for (const auto & c : q.cand) {
            out += ',';
            add_int(out, c.first);
            out += ',';
            add(out, "%.6g", (double) c.second);
        }
        out += ']';
    }
    out += ']';
}

} // namespace

std::unique_ptr<common_spec_pq_record> common_spec_pq_record::open_from_env(const char * env_name, uint32_t n_seq) {
    const char * path = std::getenv(env_name);
    if (path == nullptr || *path == '\0' || std::strcmp(path, "0") == 0) {
        return nullptr;
    }
    FILE * f = std::fopen(path, "a");
    if (f == nullptr) {
        fprintf(stderr, "%s: cannot open '%s' (%s); no p/q record\n", env_name, path, std::strerror(errno));
        return nullptr;
    }
    fprintf(stderr, "p/q record: the library's and the draft's chances, every round, to %s (%s)\n", path, env_name);
    return std::make_unique<common_spec_pq_record>(f, n_seq, wall_us());
}

common_spec_pq_record::common_spec_pq_record(FILE * f, uint32_t n_seq, int64_t wall)
    : f(f), drafts(n_seq), checks(n_seq), answer_of(n_seq, -1), sampling_of(n_seq), resample_seen(n_seq, 0) {
    char buf[160];
    snprintf(buf, sizeof(buf), "{\"record\":\"spec-pq\",\"version\":1,\"n_seq\":%u,\"max_cand\":%d,\"wall_us\":%" PRId64
            "}\n", n_seq, MAX_CAND, wall);
    put(buf);
}

common_spec_pq_record::~common_spec_pq_record() {
    for (const auto & d : drafts) {
        orphans += d.active;
    }
    char buf[160];
    snprintf(buf, sizeof(buf), "{\"end\":\"spec-pq\",\"lines\":%" PRId64 ",\"orphans\":%" PRId64
            ",\"record_us\":%" PRId64 "}\n", n_lines, orphans, t_self_us);
    put(buf);
    if (f != nullptr) {
        std::fclose(f);
    }
}

void common_spec_pq_record::put(const std::string & line) {
    if (f == nullptr) {
        return;
    }
    // flushed per line: a server stopped by a signal loses at most the line in flight
    std::fwrite(line.data(), 1, line.size(), f);
    std::fflush(f);
}

void common_spec_pq_record::sampling(int seq, const common_params_sampling & sp) {
    std::lock_guard<std::mutex> lk(mtx);
    if (!valid(seq)) {
        return;
    }
    std::string s = ",\"temp\":";
    add(s, "%.6g", sp.temp);
    s += ",\"top_k\":";           add_int(s, sp.top_k);
    s += ",\"top_p\":";           add(s, "%.6g", sp.top_p);
    s += ",\"min_p\":";           add(s, "%.6g", sp.min_p);
    s += ",\"penalty_present\":"; add(s, "%.6g", sp.penalty_present);
    s += ",\"penalty_repeat\":";  add(s, "%.6g", sp.penalty_repeat);
    s += ",\"seed\":";            add_int(s, sp.seed);
    s += ",\"grammar\":";         add_int(s, sp.grammar.empty() ? 0 : 1);
    s += ",\"grammar_lazy\":";    add_int(s, sp.grammar_lazy ? 1 : 0);
    s += ",\"budget\":";          add_int(s, sp.reasoning_budget_tokens);
    s += ",\"backend\":";         add_int(s, sp.backend_sampling ? 1 : 0);
    sampling_of[seq] = s;
}

void common_spec_pq_record::answer_begin(int seq, int n_prompt) {
    const int64_t t0 = steady_us();
    std::lock_guard<std::mutex> lk(mtx);
    if (!valid(seq)) {
        return;
    }
    answer_of[seq] = n_answers++;
    std::string s = "{\"answer\":";
    add_int(s, answer_of[seq]);
    s += ",\"seq\":";      add_int(s, seq);
    s += ",\"n_prompt\":"; add_int(s, n_prompt);
    s += ",\"wall_us\":";  add_int(s, wall_us());
    s += sampling_of[seq];
    s += "}\n";
    put(s);
    t_self_us += steady_us() - t0;
}

common_spec_pq_record::position common_spec_pq_record::read_position(common_sampler * smpl, llama_token drawn,
        uint64_t r0) {
    position q;
    q.drawn = drawn;
    const common_sampler_draw_state st = common_sampler_get_draw_state(smpl);
    q.g = (st.grammar ? 1 : 0) | (st.lazy ? 2 : 0) | (st.n_resample != r0 ? 4 : 0);
    q.b = st.budget;
    const llama_token_data_array * raw = common_sampler_get_candidates(smpl, false);
    if (raw->data == nullptr || raw->selected < 0 || (size_t) raw->selected >= raw->size) {
        return q; // a backend-sampled word leaves no chain to read (and sorting needs the selected word)
    }
    const llama_token_data_array * cur_p = common_sampler_get_candidates(smpl, true); // sorts; no draw
    q.size = (int) cur_p->size;
    const size_t n = std::min<size_t>(cur_p->size, MAX_CAND);
    q.cand.reserve(n);
    std::vector<llama_token> ids(n);
    for (size_t i = 0; i < n; i++) {
        q.cand.emplace_back(cur_p->data[i].id, cur_p->data[i].p);
        ids[i] = cur_p->data[i].id;
    }
    if (st.grammar) {
        q.refused = (int) common_sampler_grammar_refuses(smpl, ids.data(), ids.size());
    }
    return q;
}

std::vector<llama_token> common_spec_pq_record::accept_n(common_sampler * smpl, llama_context * ctx,
        const std::vector<int32_t> & idxs, const std::vector<llama_token> & draft, int seq, int pos, bool replay,
        const common_spec_lenient_draw * len) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    pending_check scratch;
    pending_check & pc = valid(seq) ? checks[seq] : scratch;
    pc = {};
    pc.active = true;
    pc.pos    = pos;
    pc.replay = replay;
    pc.draft  = draft;

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // common_sampler_sample_and_accept_n, call for call (grammar_first false); the record only reads in between
    int64_t t_rec = 0;
    size_t i = 0;
    for (; i < draft.size(); i++) {
        const uint64_t r0 = common_sampler_get_draw_state(smpl).n_resample;
        const llama_token drawn = common_sampler_sample(smpl, ctx, idxs[i]);

        int64_t t0 = steady_us();
        pc.p.push_back(read_position(smpl, drawn, r0));
        t_rec += steady_us() - t0;

        // the lenient check (speculative-lenient.h): the dial may write the drafted word over the draw
        const bool dial = len != nullptr && common_spec_lenient_keep(*len->lp, smpl, draft[i], drawn, r0, *len->rng);
        const llama_token id = dial ? draft[i] : drawn;
        if (dial) {
            pc.len.push_back((int) i);
        }

        common_sampler_ptr past;
        if (id != draft[i] && !replay && !(pc.p.back().g & 1)) {
            t0 = steady_us();
            past.reset(common_sampler_clone(smpl)); // before the library's accept: the state that drew the miss
            t_rec += steady_us() - t0;
        }

        common_sampler_accept(smpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            if (past) {
                t0 = steady_us();
                read_past_miss(past.get(), ctx, idxs, draft, i, pc);
                t_rec += steady_us() - t0;
            }
            break;
        }
    }

    if (i == draft.size()) {
        const uint64_t r0 = common_sampler_get_draw_state(smpl).n_resample;
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);

        const int64_t t0 = steady_us();
        pc.p.push_back(read_position(smpl, id, r0));
        t_rec += steady_us() - t0;

        common_sampler_accept(smpl, id, true);

        result.push_back(id);
    }

    pc.words = result;
    std::lock_guard<std::mutex> lk(mtx);
    if (valid(seq)) {
        resample_seen[seq] = common_sampler_get_draw_state(smpl).n_resample;
    }
    t_self_us += t_rec;
    return result;
}

// the positions after a miss at `miss`: a clone takes the drafted word and reads on (its draws are dropped), so
// each p follows the draft, as the check decoded it. A clone's grammar could refuse a drafted word, so none
// is made while a grammar applies (the caller's test); any other failure drops what was read.
void common_spec_pq_record::read_past_miss(common_sampler * clone, llama_context * ctx,
        const std::vector<int32_t> & idxs, const std::vector<llama_token> & draft, size_t miss,
        pending_check & pc) const {
    std::vector<position> x;
    try {
        common_sampler_accept(clone, draft[miss], true);
        for (size_t j = miss + 1; j < idxs.size(); j++) {
            const uint64_t r0 = common_sampler_get_draw_state(clone).n_resample;
            common_sampler_sample(clone, ctx, idxs[j]);
            x.push_back(read_position(clone, -1, r0));
            if (j < draft.size()) {
                common_sampler_accept(clone, draft[j], true);
            }
        }
    } catch (const std::exception &) {
        x.clear();
    }
    pc.x = std::move(x);
}

void common_spec_pq_record::check_end(int seq, bool restore) {
    const int64_t t0 = steady_us();
    std::lock_guard<std::mutex> lk(mtx);
    if (!valid(seq) || !checks[seq].active) {
        return;
    }
    pending_check & pc = checks[seq];
    // the rate trace's count (common_speculative_rate_trace::accepted): a replay re-checks the kept words plus
    // the library's own, so it counts one less, as the server's n_accepted-- does
    const int n_accepted = (int) pc.words.size() - 1;
    const int kept = !restore && pc.replay && n_accepted > 0 ? n_accepted - 1 : n_accepted;
    write_line("check", seq, pc, restore, kept);
    pc = {};
    t_self_us += steady_us() - t0;
}

void common_spec_pq_record::plain(int seq, int pos, common_sampler * smpl, llama_token id) {
    const int64_t t0 = steady_us();
    std::lock_guard<std::mutex> lk(mtx);
    if (!valid(seq)) {
        return;
    }
    // the draw is over, so its count before is the one at this sequence's last recorded draw (every draw of a
    // sequence is recorded); a lower count now is a new request's sampler, which starts at 0
    const uint64_t now = common_sampler_get_draw_state(smpl).n_resample;
    const uint64_t r0  = resample_seen[seq] <= now ? resample_seen[seq] : 0;
    resample_seen[seq] = now;
    pending_check pc;
    pc.pos   = pos;
    pc.words = { id };
    pc.p.push_back(read_position(smpl, id, r0));
    write_line("plain", seq, pc, false, 0);
    t_self_us += steady_us() - t0;
}

void common_spec_pq_record::write_line(const char * kind, int seq, const pending_check & pc, bool restore, int kept) {
    std::string s;
    s.reserve(4096);
    s += "{\"line\":\"";   s += kind;
    s += "\",\"i\":";      add_int(s, n_lines);
    s += ",\"answer\":";   add_int(s, answer_of[seq]);
    s += ",\"seq\":";      add_int(s, seq);
    s += ",\"pos\":";      add_int(s, pc.pos);
    s += ",\"replay\":";   add_int(s, pc.replay ? 1 : 0);
    s += ",\"restore\":";  add_int(s, restore ? 1 : 0);
    s += ",\"wall_us\":";  add_int(s, wall_us());
    s += ",\"n\":";        add_int(s, (int64_t) pc.draft.size());
    s += ",\"acc\":";      add_int(s, (int64_t) pc.words.size());
    s += ",\"kept\":";     add_int(s, kept);
    s += ",\"draft\":";    add_ids(s, pc.draft);
    s += ",\"words\":";    add_ids(s, pc.words);
    s += ",\"p\":";        add_positions(s, pc.p);
    s += ",\"x\":";        add_positions(s, pc.x);
    if (!pc.len.empty()) {
        s += ",\"len\":";
        add_ids(s, std::vector<llama_token>(pc.len.begin(), pc.len.end()));
    }

    // the draft from this line's anchor rides on its first line; a replay drafts nothing
    pending_draft & d = drafts[seq];
    if (d.active && !pc.replay) {
        if (d.pos0 == pc.pos) {
            add_draft(s, d);
        } else {
            orphans++;
        }
        d.active = false;
    }
    s += "}\n";
    put(s);
    n_lines++;
}

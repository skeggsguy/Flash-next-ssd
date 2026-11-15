// The p/q record's draft side (common/speculative-pq-record.h): DFlash 2's lattice, the MTP head's own chances, and
// how each rides on its round's line.

#include "speculative-pq-record.h"
#include "speculative-pq-record-util.h"

#include "speculative-dflash-walk.h"

#include "../src/llama-ext.h" // llama_get_raw_logits_ith

#include <algorithm>
#include <cmath>
#include <cstring>

using pq_util::add;
using pq_util::add_int;
using pq_util::add_ids;
using pq_util::steady_us;

void common_spec_pq_record::draft(int seq, int pos0, const common_dflash2_lattice & lat,
        const std::vector<llama_token> & words, llama_context * ctx_dft, int idx0) {
    const int64_t t0 = steady_us();
    std::vector<std::vector<std::pair<llama_token, float>>> top;
    std::vector<double> lse;
    if (ctx_dft != nullptr) { // the logits the hook asked the draft batch for, read before the lock
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_dft)));
        top.resize(std::max(lat.n_pos, 0));
        lse.assign(top.size(), NAN);
        for (int32_t i = 0; i < lat.n_pos; i++) {
            top[i] = logits_top(llama_get_raw_logits_ith(ctx_dft, idx0 + i), n_vocab, TOP_L, lse[i]);
        }
    }
    std::lock_guard<std::mutex> lk(mtx);
    if (!valid(seq)) {
        return;
    }
    pending_draft & d = drafts[seq];
    orphans += d.active; // a draft no line took
    d        = {};
    d.active = true;
    d.pos0   = pos0;
    d.k      = lat.top_k;
    d.n_pos  = lat.n_pos;
    d.walk   = words;
    const size_t row = (size_t) lat.top_k + (size_t) lat.top_k * lat.top_k;
    d.rows.resize((size_t) lat.n_pos * row);
    for (int32_t i = 0; i < lat.n_pos; i++) { // the buffer is the draft's output, reused by its next decode
        std::memcpy(d.rows.data() + (size_t) i * row, lat.rows + (size_t) i * lat.stride, row * sizeof(float));
    }
    d.wide = ctx_dft != nullptr;
    d.top  = std::move(top);
    d.lse  = std::move(lse);
    t_self_us += steady_us() - t0;
}

namespace {

// the top k (id, logit) of n logits, highest first (ties: the lower id first), with the best logit and the sum of
// exp(logit - best) over every logit no more than `skip` below it; `all`: the top k from every finite logit, else
// only from those the sum takes; empty with no row, or none finite
std::vector<std::pair<llama_token, float>> scan_top(const float * logits, int n, int k, float skip, bool all,
        float & best, double & total) {
    std::vector<std::pair<llama_token, float>> top;
    best  = -INFINITY;
    total = 0.0;
    if (logits == nullptr || n <= 0 || k <= 0) {
        return top;
    }
    for (int i = 0; i < n; i++) {
        best = std::max(best, logits[i]);
    }
    if (!std::isfinite(best)) {
        return top;
    }
    const float lo = best - skip;
    top.reserve((size_t) k + 1);
    float floor_ = -INFINITY; // the k-th best logit so far, once there are k
    for (int i = 0; i < n; i++) {
        const float l = logits[i];
        if (l >= lo) {
            total += std::exp((double) (l - best));
        } else if (!all || !(l > -INFINITY)) {
            continue;
        }
        if ((int) top.size() < k || l > floor_) {
            // after any equal logit: a lower id stays first
            const auto at = std::upper_bound(top.begin(), top.end(), l,
                    [](float v, const std::pair<llama_token, float> & e) { return v > e.second; });
            top.insert(at, { (llama_token) i, l });
            if ((int) top.size() > k) {
                top.pop_back();
            }
            floor_ = (int) top.size() < k ? -INFINITY : top.back().second;
        }
    }
    return top;
}

} // namespace

std::vector<std::pair<llama_token, float>> common_spec_pq_record::logits_top(const float * logits, int n, int k,
        double & lse) {
    float  best  = -INFINITY;
    double total = 0.0;
    auto   top   = scan_top(logits, n, k, SKIP, true, best, total);
    lse = top.empty() ? NAN : (double) best + std::log(total);
    return top;
}

std::vector<std::pair<llama_token, float>> common_spec_pq_record::softmax_top(const float * logits, int n, int k) {
    float  best  = -INFINITY;
    double total = 0.0;
    auto   top   = scan_top(logits, n, k, SKIP, false, best, total);
    for (auto & e : top) {
        e.second = (float) (std::exp((double) (e.second - best)) / total);
    }
    return top;
}

void common_spec_pq_record::mtp_begin(int seq, int pos0) {
    const int64_t t0 = steady_us();
    std::lock_guard<std::mutex> lk(mtx);
    if (!valid(seq)) {
        return;
    }
    pending_draft & d = drafts[seq];
    orphans += d.active; // a draft no line took
    d      = {};
    d.mtp  = true;       // active once it ends
    d.pos0 = pos0;
    t_self_us += steady_us() - t0;
}

void common_spec_pq_record::mtp_step(int seq, llama_context * ctx_dft, int idx, llama_token pick, float conf) {
    const int64_t t0 = steady_us();
    std::vector<std::pair<llama_token, float>> q;
    // the draft's backend sampler keeps only its top 10 on the host; the hook asked for the full logits too
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_dft)));
    q = softmax_top(llama_get_raw_logits_ith(ctx_dft, idx), n_vocab, TOP_Q);
    std::lock_guard<std::mutex> lk(mtx);
    if (!valid(seq) || !drafts[seq].mtp) {
        return;
    }
    pending_draft & d = drafts[seq];
    d.pick.push_back(pick);
    d.conf.push_back(conf);
    d.q.push_back(std::move(q));
    t_self_us += steady_us() - t0;
}

void common_spec_pq_record::mtp_end(int seq, int cap, const std::vector<llama_token> & words) {
    const int64_t t0 = steady_us();
    std::lock_guard<std::mutex> lk(mtx);
    if (!valid(seq) || !drafts[seq].mtp) {
        return;
    }
    pending_draft & d = drafts[seq];
    d.cap    = cap;
    d.walk   = words;
    d.active = true;
    t_self_us += steady_us() - t0;
}

void common_spec_pq_record::add_draft(std::string & s, const pending_draft & d) {
    if (d.mtp) {
        s += ",\"mtp\":{\"pos0\":"; add_int(s, d.pos0);
        s += ",\"cap\":";           add_int(s, d.cap);
        s += ",\"walk\":";          add_ids(s, d.walk);
        s += ",\"q\":[";
        for (size_t i = 0; i < d.q.size(); i++) {
            s += i > 0 ? ",[" : "[";
            add_int(s, d.pick[i]);
            s += ',';
            add(s, "%.6g", (double) d.conf[i]);
            for (const auto & c : d.q[i]) {
                s += ',';
                add_int(s, c.first);
                s += ',';
                add(s, "%.6g", (double) c.second);
            }
            s += ']';
        }
        s += "]}";
        return;
    }
    const size_t row = (size_t) d.k + (size_t) d.k * d.k;
    s += ",\"lat\":{\"pos0\":"; add_int(s, d.pos0);
    s += ",\"k\":";             add_int(s, d.k);
    s += ",\"walk\":";          add_ids(s, d.walk);
    s += ",\"ids\":[";
    for (int i = 0; i < d.n_pos; i++) {
        s += i > 0 ? ",[" : "[";
        for (int c = 0; c < d.k; c++) {
            if (c > 0) {
                s += ',';
            }
            add_int(s, (int64_t) d.rows[(size_t) i * row + c]); // ids are floats in the lattice
        }
        s += ']';
    }
    s += "],\"s\":[";
    for (int i = 0; i < d.n_pos; i++) {
        s += i > 0 ? ",[" : "[";
        for (int c = 0; c < d.k * d.k; c++) {
            if (c > 0) {
                s += ',';
            }
            add(s, "%.6g", (double) d.rows[(size_t) i * row + d.k + c]);
        }
        s += ']';
    }
    s += ']';
    if (d.wide) { // the whole-vocabulary logits, a row a lattice position
        s += ",\"lse\":[";
        for (size_t i = 0; i < d.lse.size(); i++) {
            if (i > 0) {
                s += ',';
            }
            add(s, "%.9g", d.lse[i]);
        }
        s += "],\"top\":[";
        for (size_t i = 0; i < d.top.size(); i++) {
            s += i > 0 ? ",[" : "[";
            for (size_t c = 0; c < d.top[i].size(); c++) {
                if (c > 0) {
                    s += ',';
                }
                add_int(s, d.top[i][c].first);
                s += ',';
                add(s, "%.9g", (double) d.top[i][c].second);
            }
            s += ']';
        }
        s += ']';
    }
    s += '}';
}

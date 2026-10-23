// The p/q record (LLAMA_SPEC_PQ_RECORD, common/speculative-pq-record.h, FAIR-CHECK-PLAN step 1) on the MTP fixture
// (qwen4exp-moe-e64mtp.gguf), driven as the server drives speculation: a check through the record's accept_n and
// check_end, every fifth round a word with no draft through plain, and a made-up lattice handed to the record's draft
// side before each check (no test model has DFlash 2's selector):
//   words      the record changes no words: at temperature 0, and at 1.0 with top-k 20 / top-p 0.95 / min-p 0.05 and a
//              seed, the same tokens and the same rounds with the record on as off (and another seed writes other
//              tokens, so the draws are real)
//   lines      one line a round; the lines' words are the text; each drawn word is among its candidates, sorted, their
//              chances summing to 1; at 0 the drawn word has chance 1; a missed check reads on past its miss (`x`,
//              one position a word the check decoded and never drew)
//   kept       each line's kept is the rate trace's kept for its cycle (the apprentice's own trace, on in the same run)
//   lattice    a draft from the round's own position rides on its line with its ids and scores exactly; one from
//              another position is not taken and counts as an orphan, as does one a replay from its position leaves
//              untouched; a replay's kept is one less than its count, a restoring check's is what it kept
// Mutation check (FAIR-CHECK-PLAN): reading past a miss on the library's own sampler instead of a clone turns
// "words" red.

#include "speculative-pq-record-run.h"

using namespace pq_test;

namespace {

bool same(const run_out & a, const run_out & b) {
    if (a.text != b.text || a.rounds.size() != b.rounds.size()) {
        return false;
    }
    for (size_t i = 0; i < a.rounds.size(); i++) {
        if (a.rounds[i].draft != b.rounds[i].draft || a.rounds[i].ids != b.rounds[i].ids) {
            return false;
        }
    }
    return true;
}

llama_tokens ids_of(const std::string & line, const std::string & key) {
    llama_tokens out;
    for (double v : nums(field(line, key))) {
        out.push_back((llama_token) v);
    }
    return out;
}

// one position [drawn, g, b, refused, size, id, p, ...]: its candidates sorted, holding the drawn word, chances summing
// to 1 (the chain's renormalised p, all of them written); at temperature 0 the drawn word is the first, at chance 1
bool position_ok(const std::vector<double> & q, bool greedy) {
    if (q.size() < 7 || (q.size() - 5) % 2 != 0 || q[4] != (double) (q.size() - 5) / 2) {
        return false;
    }
    double sum = 0.0;
    bool sorted = true, holds = q[0] < 0;
    for (size_t c = 5; c < q.size(); c += 2) {
        sum += q[c + 1];
        sorted = sorted && (c == 5 || q[c + 1] <= q[c - 1]);
        holds = holds || q[c] == q[0];
    }
    const bool top = !greedy || q[0] < 0 || (q[5] == q[0] && std::fabs(q[6] - 1.0) < 1e-6);
    return sorted && holds && top && std::fabs(sum - 1.0) < 1e-4;
}

void test_lines(const run_out & on, bool greedy) {
    const size_t n = on.rounds.size();
    check(on.lines.size() == n + 2, "one line a round, plus the replay's and the restoring check's");
    if (on.lines.size() != n + 2) {
        return;
    }
    llama_tokens words;
    std::vector<int> kept;
    bool shape = true, pos_ok = true, past = true, lat_ok = true;
    int n_past_miss = 0, n_plain = 0, n_sampled = 0;
    for (size_t r = 0; r < n; r++) {
        const std::string & l = on.lines[r];
        const auto & rd = on.rounds[r];
        const auto w = ids_of(l, "words");
        words.insert(words.end(), w.begin(), w.end());
        const bool plain = rd.draft.empty();
        n_plain += plain;
        shape = shape && field(l, "line") == (plain ? "\"plain\"" : "\"check\"") && ids_of(l, "draft") == rd.draft &&
                w == rd.ids && field(l, "replay") == "0" && field(l, "restore") == "0";
        const auto p = rows(field(l, "p"));
        const auto x = rows(field(l, "x"));
        shape = shape && p.size() == w.size();
        for (size_t i = 0; i < p.size(); i++) {
            pos_ok = pos_ok && position_ok(p[i], greedy) && p[i][0] == (double) w[i];
            n_sampled += p[i][0] != p[i][5];
        }
        const size_t miss = rd.draft.size() + 1 - w.size(); // the positions decoded and never drawn
        past = past && x.size() == miss;
        for (const auto & q : x) {
            past = past && q[0] == -1 && position_ok(q, greedy);
        }
        n_past_miss += !x.empty();
        if (!plain) {
            kept.push_back(std::atoi(field(l, "kept").c_str()));
        }
        // the made-up lattice: on its own round's line, exactly; round 3's came from the wrong position
        const std::string lat = field(l, "lat");
        if (r == 3) {
            lat_ok = lat_ok && lat.empty();
            continue;
        }
        const auto lids = rows(field(lat, "ids"));
        const auto ls   = rows(field(lat, "s"));
        bool exact = lids.size() == N_POS && ls.size() == N_POS && ids_of(lat, "walk") == rd.draft;
        const size_t stride = on.lat[r].size() / N_POS;
        for (int i = 0; exact && i < N_POS; i++) {
            for (int c = 0; c < K; c++) {
                exact = exact && lids[i].size() == K && lids[i][c] == on.lat[r][i * stride + c];
            }
            for (int c = 0; c < K * K; c++) {
                exact = exact && ls[i].size() == K * K && ls[i][c] == on.lat[r][i * stride + K + c];
            }
        }
        lat_ok = lat_ok && exact;
    }
    check(shape && words == on.text, "each line is its round: check or plain, its draft, its words (the text)");
    check(pos_ok, "each drawn word is among its candidates, sorted, chances summing to 1" +
          std::string(greedy ? ", and at 0 it is the first at chance 1" : ""));
    if (!greedy) {
        check(n_sampled > 0, std::to_string(n_sampled) + " words drawn below the top candidate (the draws are real)");
    }
    check(past && n_past_miss > 0 && n_plain > 0, "a missed check reads on past its miss, one position a decoded word (" +
          std::to_string(n_past_miss) + " rounds), and " + std::to_string(n_plain) + " plain lines");
    check(kept == on.trace_kept, "each check's kept is the rate trace's (" + std::to_string(kept.size()) + " cycles)");
    check(lat_ok, "the draft's lattice rides on its own round's line exactly; round 3's (another position) on none");
    bool json_nums = on.lines[0].find("-Infinity") != std::string::npos;
    for (const auto & l : on.lines) {
        json_nums = json_nums && l.find("inf") == std::string::npos && l.find("nan") == std::string::npos;
    }
    check(json_nums, "a masked score is written -Infinity, never C's inf (JSON can read it)");

    // the tail: the check that restores (kept as checked, the draft from its position), then its replay (one less
    // than its count, as the server counts it; it takes no draft)
    const std::string & rs = on.lines[n], & rp = on.lines[n + 1];
    const int acc_s = (int) on.tail[0].size(), acc_p = (int) on.tail[1].size();
    const auto lat_s = rows(field(field(rs, "lat"), "s"));
    check(field(rs, "replay") == "0" && field(rs, "restore") == "1" && lat_s.size() == N_POS &&
          lat_s[0][0] == on.lat[on.lat.size() - 2][K] && std::atoi(field(rs, "kept").c_str()) == acc_s - 1,
          "a restoring check keeps what it checked, and takes its draft (kept " + field(rs, "kept") + " of " +
          std::to_string(acc_s) + " words)");
    check(acc_p >= 2 && field(rp, "replay") == "1" && field(rp, "restore") == "0" && field(rp, "lat").empty() &&
          std::atoi(field(rp, "kept").c_str()) == acc_p - 2, "its replay keeps one less than its count and takes no "
          "draft (kept " + field(rp, "kept") + " of " + std::to_string(acc_p) + " words)");
    // round 3's draft, and the one the replay left
    check(field(on.end, "lines") == std::to_string(n + 2) && field(on.end, "orphans") == "2",
          "the closing line counts the lines and the two orphans (" + on.end + ")");
}

void test_prompt(const llama_tokens & prompt) {
    printf("prompt of %zu tokens, ngl %d\n", prompt.size(), g_ngl);
    for (float temp : { 0.0f, 1.0f }) {
        const std::string t = "temperature " + std::to_string(temp).substr(0, 3);
        const run_out mtp_off = run(temp, 42, prompt, "", false);
        const run_out mtp_on  = run(temp, 42, prompt, "mtp", false);
        check(mtp_off.ok && mtp_on.ok && mtp_off.text.size() > 50 && same(mtp_on, mtp_off),
              t + ", the fixture's drafts: the same words and rounds with the record on (" +
              std::to_string(mtp_on.rounds.size()) + " rounds)");
        const run_out off = run(temp, 42, prompt, "", false, &mtp_off.text);
        const run_out on  = run(temp, 42, prompt, "on", true, &mtp_off.text);
        int kept = 0, full = 0;
        for (const auto & rd : off.rounds) {
            kept += (int) rd.ids.size() - 1;
            full += !rd.draft.empty() && rd.ids.size() == rd.draft.size() + 1;
        }
        check(off.ok && on.ok && same(on, off), t + ", drafts from the text: the same words and rounds with the record on");
        // the text's own words are drawn again, until a draw near a tie moves (the check's batch is not the one that
        // wrote the text): checks keep words, and keep whole drafts
        check(kept >= 50 && full >= 5, "  " + std::to_string(off.rounds.size()) + " rounds kept " + std::to_string(kept) +
              " words, " + std::to_string(full) + " drafts whole (not vacuous)");
        if (temp > 0) {
            const run_out other = run(temp, 7, prompt, "", false);
            check(other.ok && other.text != mtp_off.text, "another seed writes other words (the draws are seeded)");
        }
        if (on.ok) {
            test_lines(on, temp == 0.0f);
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) {
            g_path = argv[++i];
        } else if (a == "-ngl" && i + 1 < argc) {
            g_ngl = std::atoi(argv[++i]);
        }
    }
    if (g_path.empty()) {
        fprintf(stderr, "usage: %s -m qwen4exp-moe-e64mtp.gguf [-ngl N]\n", argv[0]);
        return 1;
    }
    common_init();
    llama_backend_init();

    std::mt19937 rng(42);
    for (int n : { 37, 120 }) {
        llama_tokens prompt;
        for (int i = 0; i < n; i++) {
            prompt.push_back((llama_token) (1 + rng() % 100)); // the fixture has 128 words
        }
        test_prompt(prompt);
    }

    llama_backend_free();
    printf("%s\n", g_failures ? "test-speculative-pq-record-mtp: FAILED" : "test-speculative-pq-record-mtp: OK");
    return g_failures ? 1 : 0;
}

// The chain stop and the cut every draft takes (common/speculative-chain.h, CHAIN-PLAN step 2), no model:
//   off       a threshold of 0 keeps every word and needs no confidence; on, the cut needs one
//   edge      a product exactly at the threshold is kept (the draft goes on while product >= threshold), and the
//             next word under it ends the draft; a word exactly at p-min is kept
//   double    the product runs in double: a threshold equal to the double product of float confidences keeps them
//             all, where a float product would have rounded under it
//   order     p-min is checked first, so a word under both is named p-min; `start` opens a fresh product
// With --script it reads lines of "p_min chain_p conf..." from stdin and prints "<words drafted> <stop>" a line:
// the study's sim/test_chain_engine.py feeds it generated drafts and checks sim/chain_replay.py cuts the same.

#include "speculative-chain.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_failures++; } } while (0)

using trace_t = common_speculative_rate_trace;

// how many words of `conf` the cut keeps, and why it stopped ("cap" when it kept them all)
static size_t run_cut(common_speculative_cut cut, const std::vector<float> & conf, trace_t::stop_reason & why) {
    why = trace_t::STOP_CAP;
    cut.start();
    size_t n = 0;
    for (float p : conf) {
        if (!cut.keep(p, why)) {
            return n;
        }
        n++;
    }
    return n;
}

static void test_off() {
    common_speculative_cut cut(0.0f, 0.0);
    CHECK(!cut.needs_conf() && !cut.chain.on(), "p-min 0 and no chain: no confidence needed");
    CHECK(common_speculative_cut(0.0f, 0.25).needs_conf(), "the chain on needs the confidence");
    CHECK(common_speculative_cut(0.3f, 0.0).needs_conf(), "p-min on needs the confidence");
    trace_t::stop_reason why;
    CHECK(run_cut(cut, {1e-6f, 1e-6f, 1e-6f, 1e-6f}, why) == 4 && why == trace_t::STOP_CAP, "off keeps every word");
}

static void test_edge() {
    trace_t::stop_reason why;
    // 0.5 x 0.5 = 0.25 exactly: kept at a threshold of 0.25, and the third 0.5 takes it under
    CHECK(run_cut(common_speculative_cut(0.0f, 0.25), {0.5f, 0.5f, 0.5f}, why) == 2 && why == trace_t::STOP_CHAIN,
          "a product exactly at the threshold is kept; the next word under it ends the draft");
    CHECK(run_cut(common_speculative_cut(0.0f, 0.5), {0.5f, 1.0f, 1.0f}, why) == 3 && why == trace_t::STOP_CAP,
          "a first word exactly at the threshold, then words of 1: all kept");
    CHECK(run_cut(common_speculative_cut(0.0f, 0.5), {0.25f}, why) == 0 && why == trace_t::STOP_CHAIN,
          "a first word under the threshold: nothing drafted");
    CHECK(run_cut(common_speculative_cut(0.5f, 0.0), {0.5f, 0.5f, 0.4999999f}, why) == 2 && why == trace_t::STOP_PMIN,
          "a word exactly at p-min is kept (p < p_min cuts)");
}

static void test_double_product() {
    // a draft whose float product rounds under its double product: the threshold at the double product keeps it
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(0.3f, 1.0f);
    bool found = false;
    for (int t = 0; t < 10000 && !found; t++) {
        std::vector<float> conf(6);
        double dp = 1.0;
        float  fp = 1.0f;
        for (auto & c : conf) {
            c = u(rng);
            dp *= (double) c;
            fp *= c;
        }
        if ((double) fp >= dp) {
            continue;
        }
        found = true;
        trace_t::stop_reason why;
        CHECK(run_cut(common_speculative_cut(0.0f, dp), conf, why) == conf.size() && why == trace_t::STOP_CAP,
              "a threshold at the double product keeps all %zu words (a float product %.17g < %.17g would cut)",
              conf.size(), (double) fp, dp);
    }
    CHECK(found, "found a draft whose float product rounds under its double product");
}

static void test_order_and_start() {
    trace_t::stop_reason why;
    // 0.4 is under p-min 0.5 and takes the product (0.9 x 0.4 = 0.36) under 0.5 too: named p-min
    CHECK(run_cut(common_speculative_cut(0.5f, 0.5), {0.9f, 0.4f}, why) == 1 && why == trace_t::STOP_PMIN,
          "a word under both cuts is named p-min");
    CHECK(run_cut(common_speculative_cut(0.1f, 0.5), {0.9f, 0.5f, 0.05f}, why) == 1 && why == trace_t::STOP_CHAIN,
          "the chain (0.45 at the second word) before p-min (0.05 at the third): chain");
    common_speculative_cut cut(0.0f, 0.5);
    cut.start();
    CHECK(cut.keep(0.6f, why) && !cut.keep(0.6f, why), "0.6 then 0.36: the second ends it");
    cut.start();
    CHECK(cut.keep(0.6f, why) && cut.chain.product == (double) 0.6f, "start opens a fresh product");
}

// "p_min chain_p conf..." a line -> "<words drafted> <stop>"; p-min and the confidences parse as floats, the threshold
// as a double, as the engine reads each
static int script() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string tok;
        std::vector<std::string> toks;
        while (in >> tok) {
            toks.push_back(tok);
        }
        if (toks.size() < 2) {
            fprintf(stderr, "a script line is \"p_min chain_p conf...\": %s\n", line.c_str());
            return 1;
        }
        const float  p_min = std::strtof(toks[0].c_str(), nullptr);
        const double chain = std::strtod(toks[1].c_str(), nullptr);
        std::vector<float> conf;
        for (size_t i = 2; i < toks.size(); i++) {
            conf.push_back(std::strtof(toks[i].c_str(), nullptr));
        }
        trace_t::stop_reason why;
        const size_t n = run_cut(common_speculative_cut(p_min, chain), conf, why);
        printf("%zu %s\n", n, trace_t::stop_name(why));
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--script") == 0) {
        return script();
    }
    test_off();
    test_edge();
    test_double_product();
    test_order_and_start();
    if (g_failures) {
        fprintf(stderr, "test-speculative-chain: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-speculative-chain: OK\n");
    return 0;
}

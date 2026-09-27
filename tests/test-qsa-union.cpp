// LLAMA_QSA_UNION (src/models/qwen4exp-qsa-union.h): the switch's parse and startup line, and the route a
// layer takes, worked through every condition. The bounds the GPU bias is built from are brute-forced in
// tests/test-qsa-picks.cpp; the graph (the same bytes as the host bias) in test-qsa-keep's union runs.

#include "../src/models/qwen4exp-qsa-union.h"
#include "../src/llama-qsa-picks.h"

#include <cstdio>
#include <cstring>
#include <string>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static void test_parse() {
    bool bad = true;
    CHECK(llama_qsa_union_parse(nullptr, &bad) == LLAMA_QSA_UNION_DEFAULT && !bad);
    CHECK(LLAMA_QSA_UNION_DEFAULT == LLAMA_QSA_UNION_BIAS); // "bias" since the U-union rung; an off arm writes "0"
    CHECK(llama_qsa_union_parse("0", &bad) == LLAMA_QSA_UNION_OFF && !bad);
    CHECK(llama_qsa_union_parse("bias", &bad) == LLAMA_QSA_UNION_BIAS && !bad);
    CHECK(llama_qsa_union_parse("1", &bad) == LLAMA_QSA_UNION_ATTN && !bad);
    for (const char * junk : { "", "2", "on", "BIAS", "yes", "01", " 1" }) {
        bad = false;
        CHECK(llama_qsa_union_parse(junk, &bad) == LLAMA_QSA_UNION_OFF && bad);
    }
    CHECK(llama_qsa_union_parse("junk") == LLAMA_QSA_UNION_OFF); // no flag asked for

    // each startup line says what runs and how to get the others
    const std::string off  = llama_qsa_union_describe(LLAMA_QSA_UNION_OFF);
    const std::string bias = llama_qsa_union_describe(LLAMA_QSA_UNION_BIAS);
    const std::string attn = llama_qsa_union_describe(LLAMA_QSA_UNION_ATTN);
    CHECK(off.rfind("off,", 0) == 0 && off.find("LLAMA_QSA_UNION=bias") != std::string::npos);
    CHECK(bias.rfind("bias,", 0) == 0 && bias.find("exact") != std::string::npos && bias.find("LLAMA_QSA_UNION=0") != std::string::npos);
    CHECK(attn.rfind("on,", 0) == 0 && attn.find("no mask") != std::string::npos && attn.find("LLAMA_QSA_UNION=0") != std::string::npos);

    // "1" in a context that cannot take union attention: the line says bias, and why
    const std::string fell = llama_qsa_union_describe(LLAMA_QSA_UNION_ATTN, "this context has no flash attention");
    CHECK(fell.rfind("bias (LLAMA_QSA_UNION=1", 0) == 0 && fell.find("no flash attention") != std::string::npos &&
          fell.find("exact") != std::string::npos && fell.find("keeps the mask") != std::string::npos);
    // ... and only "1" falls back: the other lines never change
    CHECK(llama_qsa_union_describe(LLAMA_QSA_UNION_OFF,  "x") == off);
    CHECK(llama_qsa_union_describe(LLAMA_QSA_UNION_BIAS, "x") == bias);
    CHECK(llama_qsa_union_describe(LLAMA_QSA_UNION_ATTN, nullptr) == attn);
}

// a reading-in batch on the library's shape: the GPU bias route with every condition met
static llama_qsa_union_gate ok_gate(llama_qsa_union_mode mode) {
    llama_qsa_union_gate g;
    g.mode       = mode;
    g.block_topk = true;
    g.causal     = true;
    g.one_seq    = true;
    g.gather     = false;
    g.n_stream   = 1;
    g.n_tokens   = 4096;
    g.n_kv       = 163840;
    g.flash_attn = true;
    g.kv_f16     = true;
    g.same_head  = true;
    return g;
}

static void test_route() {
    CHECK(llama_qsa_union_route(ok_gate(LLAMA_QSA_UNION_OFF))  == LLAMA_QSA_UNION_OFF);
    CHECK(llama_qsa_union_route(ok_gate(LLAMA_QSA_UNION_BIAS)) == LLAMA_QSA_UNION_BIAS);
    CHECK(llama_qsa_union_route(ok_gate(LLAMA_QSA_UNION_ATTN)) == LLAMA_QSA_UNION_ATTN);

    // any bias condition missing: today's host inputs, whatever was asked
    for (llama_qsa_union_mode mode : { LLAMA_QSA_UNION_BIAS, LLAMA_QSA_UNION_ATTN }) {
        llama_qsa_union_gate g;
        g = ok_gate(mode); g.block_topk = false;     CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_OFF);
        g = ok_gate(mode); g.causal = false;         CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_OFF);
        g = ok_gate(mode); g.one_seq = false;        CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_OFF);
        g = ok_gate(mode); g.gather = true;          CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_OFF);
        g = ok_gate(mode); g.n_stream = 2;           CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_OFF);
        g = ok_gate(mode); g.n_tokens = LLAMA_QSA_UNION_MIN_ROWS - 1; CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_OFF);
        g = ok_gate(mode); g.n_tokens = 1;           CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_OFF);
        g = ok_gate(mode); g.n_tokens = LLAMA_QSA_UNION_MIN_ROWS; CHECK(llama_qsa_union_route(g) == mode);
        g = ok_gate(mode); g.n_kv = LLAMA_QSA_UNION_MAX_KV + 1;   CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_OFF);
        g = ok_gate(mode); g.n_kv = LLAMA_QSA_UNION_MAX_KV;       CHECK(llama_qsa_union_route(g) == mode);
    }

    // union attention's own conditions missing: the exact GPU bias under the masked path
    llama_qsa_union_gate g;
    g = ok_gate(LLAMA_QSA_UNION_ATTN); g.flash_attn = false; CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_BIAS);
    g = ok_gate(LLAMA_QSA_UNION_ATTN); g.kv_f16 = false;     CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_BIAS);
    g = ok_gate(LLAMA_QSA_UNION_ATTN); g.same_head = false;  CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_BIAS);
    // ... which the bias route never needs
    g = ok_gate(LLAMA_QSA_UNION_BIAS); g.flash_attn = false; g.kv_f16 = false; CHECK(llama_qsa_union_route(g) == LLAMA_QSA_UNION_BIAS);
}

// the reason the startup line gives when "1" builds "bias" in a whole context, and none otherwise
static void test_fallback() {
    CHECK(llama_qsa_union_fallback(ok_gate(LLAMA_QSA_UNION_ATTN)) == nullptr);
    CHECK(llama_qsa_union_fallback(ok_gate(LLAMA_QSA_UNION_BIAS)) == nullptr);
    CHECK(llama_qsa_union_fallback(ok_gate(LLAMA_QSA_UNION_OFF))  == nullptr);

    llama_qsa_union_gate g;
    g = ok_gate(LLAMA_QSA_UNION_ATTN); g.flash_attn = false;
    CHECK(llama_qsa_union_fallback(g) != nullptr && strstr(llama_qsa_union_fallback(g), "flash attention") != nullptr);
    g = ok_gate(LLAMA_QSA_UNION_ATTN); g.kv_f16 = false;
    CHECK(llama_qsa_union_fallback(g) != nullptr && strstr(llama_qsa_union_fallback(g), "F16") != nullptr);
    g = ok_gate(LLAMA_QSA_UNION_ATTN); g.same_head = false;
    CHECK(llama_qsa_union_fallback(g) != nullptr && strstr(llama_qsa_union_fallback(g), "heads") != nullptr);

    // a batch's own conditions (too few rows, several streams, the gather) are no fallback: the line describes the context
    g = ok_gate(LLAMA_QSA_UNION_ATTN); g.n_tokens = 1; g.n_stream = 2; g.gather = true;
    CHECK(llama_qsa_union_fallback(g) == nullptr);
    // the reason and the route agree
    for (int k = 0; k < 3; ++k) {
        g = ok_gate(LLAMA_QSA_UNION_ATTN);
        (k == 0 ? g.flash_attn : k == 1 ? g.kv_f16 : g.same_head) = false;
        CHECK((llama_qsa_union_fallback(g) != nullptr) == (llama_qsa_union_route(g) == LLAMA_QSA_UNION_BIAS));
    }
    // "bias" asked for in the same context: no reason, the plain bias line
    g = ok_gate(LLAMA_QSA_UNION_BIAS); g.flash_attn = false;
    CHECK(llama_qsa_union_fallback(g) == nullptr && llama_qsa_union_describe(g.mode, llama_qsa_union_fallback(g)).rfind("bias,", 0) == 0);
}

int main() {
    test_parse();
    test_route();
    test_fallback();

    fprintf(stderr, "test-qsa-union: %s\n", n_fail == 0 ? "all tests OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}

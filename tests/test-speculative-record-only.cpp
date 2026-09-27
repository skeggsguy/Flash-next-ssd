// LLAMA_MTP_RECORD_ONLY's parse and the decisions around it (src/llama-mtp-record.h), worked by hand:
// which contexts take the switch, which batches record only, what the big reserve asks for, and the
// startup line an apprentice's context writes. The graph side (the records byte-identical to the whole
// floor's, the working space smaller) is test-qwen4exp-mtp-record on the MTP fixture.

#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-mtp-record.h"

#include <cstdio>
#include <string>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static std::string g_log;

static void test_parse() {
    CHECK(llama_mtp_record_only_parse(nullptr) == LLAMA_MTP_RECORD_ONLY_DEFAULT);
    CHECK(llama_mtp_record_only_parse(nullptr) == true); // on since the A-record rung; "0" is the whole floor
    CHECK(llama_mtp_record_only_parse("1")   == true);
    CHECK(llama_mtp_record_only_parse("2")   == true);
    CHECK(llama_mtp_record_only_parse("0")   == false); // today's graph
    CHECK(llama_mtp_record_only_parse("")    == false); // not a positive number: off, like the QSA switches
    CHECK(llama_mtp_record_only_parse("off") == false);
    CHECK(llama_mtp_record_only_parse("on")  == false);
    CHECK(llama_mtp_record_only_parse("-1")  == false);
}

static void test_supported() {
    CHECK(llama_mtp_record_only_supported(LLM_ARCH_QWEN4EXP));
    // MTP heads of other archs build their whole floor for every batch: the switch must not reach
    // their reserve, which would then size a graph with no outputs that their builder never makes
    CHECK(!llama_mtp_record_only_supported(LLM_ARCH_QWEN35MOE));
    CHECK(!llama_mtp_record_only_supported(LLM_ARCH_DEEPSEEK4));
    CHECK(!llama_mtp_record_only_supported(LLM_ARCH_GLM_DSA));
}

static void test_init() {
    // the library's own context never takes it, and says nothing
    g_log.clear();
    CHECK(!llama_mtp_record_only_init("1", false, LLM_ARCH_QWEN4EXP));
    CHECK(!llama_mtp_record_only_init(nullptr, false, LLM_ARCH_QWEN4EXP));
    CHECK(g_log.empty());

    // an apprentice's context: on, and the line says so and how to turn it off
    g_log.clear();
    CHECK(llama_mtp_record_only_init("1", true, LLM_ARCH_QWEN4EXP));
    CHECK(g_log.find("apprentice: reading in records K/V only") != std::string::npos);
    CHECK(g_log.find("LLAMA_MTP_RECORD_ONLY=0") != std::string::npos);

    // off: the whole floor, and how to turn it on
    g_log.clear();
    CHECK(!llama_mtp_record_only_init("0", true, LLM_ARCH_QWEN4EXP));
    CHECK(g_log.find("apprentice: reading in runs the whole floor") != std::string::npos);
    CHECK(g_log.find("LLAMA_MTP_RECORD_ONLY=1") != std::string::npos);

    // unset: the default, on since the A-record rung, and the line says how to turn it off
    g_log.clear();
    CHECK(llama_mtp_record_only_init(nullptr, true, LLM_ARCH_QWEN4EXP) == LLAMA_MTP_RECORD_ONLY_DEFAULT);
    CHECK(llama_mtp_record_only_init(nullptr, true, LLM_ARCH_QWEN4EXP));
    CHECK(g_log.find("apprentice: reading in records K/V only") != std::string::npos);
    CHECK(g_log.find("LLAMA_MTP_RECORD_ONLY=0") != std::string::npos);

    // asked for on an arch without the graph: off, and the line says why
    g_log.clear();
    CHECK(!llama_mtp_record_only_init("1", true, LLM_ARCH_QWEN35MOE));
    CHECK(g_log.find("has no record-only graph") != std::string::npos);
}

static void test_decisions() {
    // the graph: only a batch without outputs, and only when on
    CHECK( llama_mtp_records_only(true,  0));
    CHECK(!llama_mtp_records_only(true,  1));   // a guess step
    CHECK(!llama_mtp_records_only(true,  6));   // chained heads' guess batch (other archs), never recorded only
    CHECK(!llama_mtp_records_only(false, 0));   // off: today's graph for every batch

    // the big reserve: none when on (it sizes the record-only graph), the caller's count when off
    CHECK(llama_mtp_reserve_outputs(true,  1) == 0);
    CHECK(llama_mtp_reserve_outputs(true,  4) == 0);
    CHECK(llama_mtp_reserve_outputs(false, 1) == 1);
    CHECK(llama_mtp_reserve_outputs(false, 4) == 4);
}

int main() {
    llama_log_set([](ggml_log_level, const char * text, void *) { g_log += text; }, nullptr);

    test_parse();
    test_supported();
    test_init();
    test_decisions();

    if (n_fail == 0) {
        printf("test-speculative-record-only: OK\n");
        return 0;
    }
    fprintf(stderr, "test-speculative-record-only: %d failed\n", n_fail);
    return 1;
}

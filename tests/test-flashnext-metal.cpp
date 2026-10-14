// The fork's proven defaults, ggml-metal's two (the Flash-Next study's standard roll-out, 2026-10-15): what
// ggml-metal's own parsers read unset, "0" and the rest, and that common's start-up line names the same defaults
// (common/flashnext.h). No device needed: the parsers are plain functions.

#include "testing.h"

#include "flashnext.h"

#include <cstdint>
#include <iostream>

extern "C" int     ggml_metal_staff_mv_parse(const char * s, bool * bad);
extern "C" int     ggml_metal_keep_alive_parse(const char * s);

namespace {

void test_staff(testing & t) {
    bool bad = true;
    t.assert_equal("unset: today's route", 0, ggml_metal_staff_mv_parse(nullptr, &bad));
    t.assert_true("unset is not bad", !bad);
    t.assert_equal("\"\": today's route", 0, ggml_metal_staff_mv_parse("", &bad));
    t.assert_true("\"\" is not bad", !bad);
    t.assert_equal("\"0\": today's route", 0, ggml_metal_staff_mv_parse("0", &bad));
    t.assert_equal("\"1\": the matrix units", 1, ggml_metal_staff_mv_parse("1", &bad));
    t.assert_true("\"1\" is not bad", !bad);
    t.assert_equal("a word: today's route", 0, ggml_metal_staff_mv_parse("on", &bad));
    t.assert_true("a word is bad", bad);
    t.assert_equal("the line's default is the device's", COMMON_FLASHNEXT_STAFF_MV,
                   ggml_metal_staff_mv_parse(nullptr, nullptr));
}

void test_keep_alive(testing & t) {
    t.assert_equal("unset: ~115 days", 10000000, ggml_metal_keep_alive_parse(nullptr));
    t.assert_equal("\"0\": upstream's 3 minutes", 180, ggml_metal_keep_alive_parse("0"));
    t.assert_equal("\"-5\": upstream's 3 minutes", 180, ggml_metal_keep_alive_parse("-5"));
    t.assert_equal("\"\": upstream's 3 minutes", 180, ggml_metal_keep_alive_parse(""));
    t.assert_equal("\"600\"", 600, ggml_metal_keep_alive_parse("600"));
    t.assert_equal("the line's default is the device's", COMMON_FLASHNEXT_KEEP_ALIVE_S,
                   (int64_t) ggml_metal_keep_alive_parse(nullptr));
}

} // namespace

int main(int argc, char ** argv) {
    testing t(std::cout);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    t.test("GGML_METAL_STAFF_MV", test_staff);
    t.test("GGML_METAL_RESIDENCY_KEEP_ALIVE_S", test_keep_alive);
    return t.summary();
}

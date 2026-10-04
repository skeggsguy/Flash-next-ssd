// The Anthropic endpoint's request conversion (server_chat_convert_anthropic_to_oai), without a model.
//
// LANES-PLAN step 1: a /v1/messages request names its seat with "id_slot" (the front door sets it), so the
// conversion to a chat completions body must carry it through, as it does temperature and stream; without it the
// server picks a seat on its own and chat and coding wipe each other's cache.
// - id_slot passes through unchanged (seat 0, seat 1, and -1 = any seat)
// - no id_slot asked for, none added (the server's default, -1, then applies)
// - the params already passed through still are

#include "server-chat.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int n_failed = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            n_failed++;                                                          \
        }                                                                        \
    } while (0)

static json messages_request() {
    return json{
        {"model", "test"},
        {"max_tokens", 16},
        {"messages", json::array({ json{{"role", "user"}, {"content", "hi"}} })},
    };
}

static void test_id_slot_passes_through() {
    for (int id : {0, 1, -1}) {
        json body = messages_request();
        body["id_slot"] = id;
        const json oai = server_chat_convert_anthropic_to_oai(body);
        CHECK(oai.contains("id_slot"));
        CHECK(oai.contains("id_slot") && oai.at("id_slot").is_number_integer());
        CHECK(json_value(oai, "id_slot", -2) == id);
    }
}

static void test_no_id_slot_none_added() {
    const json oai = server_chat_convert_anthropic_to_oai(messages_request());
    CHECK(!oai.contains("id_slot"));
    CHECK(json_value(oai, "id_slot", -1) == -1); // what the completion handler reads: any seat
}

static void test_other_params_still_pass() {
    json body = messages_request();
    body["temperature"] = 0.5;
    body["top_p"]       = 0.9;
    body["top_k"]       = 20;
    body["stream"]      = true;
    body["id_slot"]     = 1;
    const json oai = server_chat_convert_anthropic_to_oai(body);
    CHECK(json_value(oai, "temperature", 0.0) == 0.5);
    CHECK(json_value(oai, "top_p", 0.0) == 0.9);
    CHECK(json_value(oai, "top_k", 0) == 20);
    CHECK(json_value(oai, "stream", false) == true);
    CHECK(json_value(oai, "max_tokens", 0) == 16);
    CHECK(json_value(oai, "id_slot", -2) == 1);
}

int main() {
    test_id_slot_passes_through();
    test_no_id_slot_none_added();
    test_other_params_still_pass();
    if (n_failed) {
        fprintf(stderr, "test-server-anthropic: %d check(s) failed\n", n_failed);
        return 1;
    }
    printf("test-server-anthropic: OK\n");
    return 0;
}

// The chat window's rules (tools/server/server-lane.h, study CHAT-WINDOW-PLAN.md step 1), without a model.
//
// - the lane lookup: the header's name and the background value match ignoring case; a missing header, another
//   value or an empty one is chat; a renamed header and value work the same way
// - should_cut: only a chat arriving while code runs cuts
// - the deferred pick: chat first, oldest first; code only once it may start; examples and generated queues
// - the window on an injected clock: opens when a chat ends, is extended by each later chat, holds while a chat runs,
//   lapses hold_ms after the last chat, and take_lapse() reports each lapse exactly once (examples and generated
//   event sequences against a plain reference)
// - the real server_queue: pop_deferred_task() follows the registered pick, the timed wake ticks while idle and can
//   start a held task with no seat released, and the server does not sleep on idle while tasks are held
// - in test-server-lane-config.cpp: the three options, off by default, from the command line, environment and INI

#include "server-lane.h"
#include "server-queue.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

int n_fail = 0;

void test_config(); // test-server-lane-config.cpp

#define CHECK(cond) do { if (!(cond)) { ++n_fail; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static void test_lane_of() {
    const std::string h = "X-Lane", bg = "code";
    CHECK(server_lane_of({{"X-Lane", "code"}}, h, bg) == SERVER_LANE_CODE);
    // HTTP header names are case-insensitive, and the server's header map keeps the client's spelling
    CHECK(server_lane_of({{"x-lane", "code"}}, h, bg) == SERVER_LANE_CODE);
    CHECK(server_lane_of({{"X-LANE", "code"}}, h, bg) == SERVER_LANE_CODE);
    CHECK(server_lane_of({{"X-Lane", "code"}}, "x-lane", bg) == SERVER_LANE_CODE);
    // the value ignores case and surrounding spaces
    CHECK(server_lane_of({{"X-Lane", "Code"}}, h, bg) == SERVER_LANE_CODE);
    CHECK(server_lane_of({{"X-Lane", " code "}}, h, bg) == SERVER_LANE_CODE);
    CHECK(server_lane_of({{"X-Lane", "code"}}, h, "CODE") == SERVER_LANE_CODE);
    // missing, another value, empty, or a value that only contains the background word: chat
    CHECK(server_lane_of({}, h, bg) == SERVER_LANE_CHAT);
    CHECK(server_lane_of({{"Content-Type", "application/json"}, {"User-Agent", "curl/8"}}, h, bg) == SERVER_LANE_CHAT);
    CHECK(server_lane_of({{"X-Lane", "chat"}}, h, bg) == SERVER_LANE_CHAT);
    CHECK(server_lane_of({{"X-Lane", ""}}, h, bg) == SERVER_LANE_CHAT);
    CHECK(server_lane_of({{"X-Lane", "codex"}}, h, bg) == SERVER_LANE_CHAT);
    CHECK(server_lane_of({{"X-Lanes", "code"}}, h, bg) == SERVER_LANE_CHAT);
    // the background value under another header does not count
    CHECK(server_lane_of({{"X-Role", "code"}}, h, bg) == SERVER_LANE_CHAT);
    // configured differently: --chat-window-header X-Client-Role --chat-window-background agent
    CHECK(server_lane_of({{"x-client-role", "agent"}}, "X-Client-Role", "agent") == SERVER_LANE_CODE);
    CHECK(server_lane_of({{"X-Lane", "code"}}, "X-Client-Role", "agent") == SERVER_LANE_CHAT);

    CHECK(std::string(server_lane_name(SERVER_LANE_CHAT)) == "chat");
    CHECK(std::string(server_lane_name(SERVER_LANE_CODE)) == "code");
    CHECK(std::string(server_lane_name(SERVER_LANE_NONE)) == "none");
}

static void test_should_cut() {
    const server_lane lanes[] = {SERVER_LANE_NONE, SERVER_LANE_CHAT, SERVER_LANE_CODE};
    for (server_lane running : lanes) {
        for (server_lane arriving : lanes) {
            const bool want = running == SERVER_LANE_CODE && arriving == SERVER_LANE_CHAT;
            CHECK(server_lane_should_cut(running, arriving) == want);
        }
    }
}

static void test_pick() {
    const auto N = SERVER_LANE_NONE, C = SERVER_LANE_CHAT, K = SERVER_LANE_CODE;
    CHECK(server_lane_pick({}, true) == -1);
    CHECK(server_lane_pick({}, false) == -1);
    CHECK(server_lane_pick({K, C, K}, false) == 1); // chat jumps the queue
    CHECK(server_lane_pick({K, C, K}, true) == 1);  // also outside the window
    CHECK(server_lane_pick({K, K}, false) == -1);   // code waits while the window is open
    CHECK(server_lane_pick({K, K}, true) == 0);     // and goes in order once it lapses
    CHECK(server_lane_pick({K, N, K}, false) == 1); // a request without a lane is not held
    CHECK(server_lane_pick({K, N, C}, true) == 2);
    CHECK(server_lane_pick({N, K}, true) == 0);

    // generated queues: the pick is chat whenever any chat waits, never code while code may not start, and the
    // oldest of the lanes allowed to go
    std::mt19937 rng(42);
    for (int it = 0; it < 2000; it++) {
        std::vector<server_lane> q(rng() % 7);
        for (auto & l : q) {
            l = (server_lane) (rng() % 3);
        }
        const bool code_ok = rng() % 2;
        const int  i = server_lane_pick(q, code_ok);
        const bool any_chat = std::find(q.begin(), q.end(), C) != q.end();
        const bool any_ok = std::any_of(q.begin(), q.end(), [&](server_lane l) { return code_ok || l != K; });
        CHECK((i >= 0) == any_ok);
        if (i < 0) {
            continue;
        }
        CHECK(i < (int) q.size());
        CHECK(!any_chat || q[i] == C);
        CHECK(code_ok || q[i] != K);
        for (int j = 0; j < i; j++) {
            // nothing older was allowed to go
            CHECK(!(any_chat ? q[j] == C : (code_ok || q[j] != K)));
        }
    }
}

static void test_window() {
    server_lane_window off;
    CHECK(!off.on());
    off.on_chat_done(1000);
    CHECK(off.code_may_start(1000, false));
    CHECK(off.code_may_start(1000, true)); // off: upstream, nothing waits for chat
    CHECK(!off.take_lapse(5000, false));

    server_lane_window w(60000);
    CHECK(w.on());
    CHECK(w.code_may_start(0, false));     // no chat yet
    CHECK(!w.code_may_start(0, true));     // a chat holds the seat
    CHECK(w.seconds_left(0, true) == -1);
    CHECK(!w.take_lapse(0, false));        // no window has opened

    w.on_chat_done(10000);                 // a chat ends at 10 s: the window runs to 70 s
    CHECK(!w.code_may_start(10000, false));
    CHECK(!w.code_may_start(69999, false));
    CHECK(w.seconds_left(10000, false) == 60);
    CHECK(w.seconds_left(69001, false) == 1); // rounded up, never 0 while code still waits
    CHECK(!w.take_lapse(69999, false));
    CHECK(w.code_may_start(70000, false));
    CHECK(w.seconds_left(70000, false) == 0);

    w.on_chat_done(40000);                 // another chat ended at 40 s: extended to 100 s
    CHECK(!w.code_may_start(70000, false));
    CHECK(!w.take_lapse(99999, false));
    CHECK(!w.take_lapse(100000, true));    // lapsed by the clock, but a new chat holds the seat
    CHECK(w.take_lapse(100000, false));
    CHECK(!w.take_lapse(100001, false));   // once per window
    CHECK(!w.take_lapse(500000, false));

    // a chat that ends earlier than the window already open does not shorten it
    w.on_chat_done(200000);
    w.on_chat_done(150000);
    CHECK(!w.code_may_start(259999, false));
    CHECK(w.take_lapse(260000, false));
}

// generated sequences of chat starts, chat ends and clock steps against the plain rule: code may start when no chat
// runs and hold has passed since the last chat ended; a lapse is reported once, at the first look after that
static void test_window_generated() {
    std::mt19937 rng(7);
    for (int run = 0; run < 300; run++) {
        const int64_t hold = 1 + rng() % 5000;
        server_lane_window w(hold);
        int64_t now = 0, last_end = -1;
        int     running = 0;
        bool    reported = true; // nothing to report before the first chat ends
        for (int step = 0; step < 200; step++) {
            switch (rng() % 4) {
                case 0: running++; break;
                case 1: if (running > 0) { running--; last_end = now; w.on_chat_done(now); reported = false; } break;
                default: now += rng() % 3000; break;
            }
            const bool want = running == 0 && (last_end < 0 || now >= last_end + hold);
            CHECK(w.code_may_start(now, running > 0) == want);
            if (!want) {
                CHECK(w.seconds_left(now, running > 0) == (running > 0 ? -1 : (last_end + hold - now + 999) / 1000));
            }
            const bool lapse = w.take_lapse(now, running > 0);
            CHECK(lapse == (want && !reported));
            if (lapse) {
                reported = true;
            }
        }
    }
}

// the real queue with the chat window's pick: which deferred tasks come out, and in what order
static void test_queue_pick() {
    server_queue q;
    std::vector<int> ran;
    bool code_ok = false;
    q.on_new_task([&](server_task && task, bool) { ran.push_back(task.id); return true; });
    q.on_update_slots([&]() { q.terminate(); });
    q.on_pick_deferred([&](const std::deque<server_task> & deferred) {
        std::vector<server_lane> lanes;
        for (const auto & t : deferred) {
            lanes.push_back(t.lane);
        }
        return server_lane_pick(lanes, code_ok);
    });

    auto task_of = [&](server_lane lane) {
        server_task t(SERVER_TASK_TYPE_COMPLETION);
        t.id   = q.get_new_id();
        t.lane = lane;
        return t;
    };
    server_task code0 = task_of(SERVER_LANE_CODE), chat1 = task_of(SERVER_LANE_CHAT), none2 = task_of(SERVER_LANE_NONE);
    const int id_code = code0.id, id_chat = chat1.id, id_none = none2.id;
    q.defer(std::move(code0));
    q.defer(std::move(chat1));
    q.defer(std::move(none2));

    q.pop_deferred_task(-1); // the chat
    q.pop_deferred_task(-1); // the lane-less task, not held
    q.pop_deferred_task(-1); // nothing: code is held
    CHECK(q.queue_tasks_deferred_size() == 1);
    q.start_loop(-1);
    // popped tasks go to the front, so the later pop runs first, as upstream's pop does
    CHECK(ran == std::vector<int>({id_none, id_chat}));

    code_ok = true;
    q.pop_deferred_task(-1);
    CHECK(q.queue_tasks_deferred_size() == 0);
    ran.clear();
    q.start_loop(-1);
    CHECK(ran == std::vector<int>({id_code}));
}

// the timed wake: ticks while the loop is idle, and a held task starts from it with no seat released
static void test_queue_tick() {
    server_queue q;
    std::atomic<int> n_tick{0};
    std::vector<int> ran;
    bool code_ok = false;
    q.on_new_task([&](server_task && task, bool) { ran.push_back(task.id); q.terminate(); return true; });
    q.on_update_slots([]() {});
    q.on_pick_deferred([&](const std::deque<server_task> & deferred) {
        std::vector<server_lane> lanes;
        for (const auto & t : deferred) {
            lanes.push_back(t.lane);
        }
        return server_lane_pick(lanes, code_ok);
    });
    q.on_tick([&]() {
        // the window lapses on the third tick: the tick moves the held task, as the server's tick does
        if (++n_tick == 3) {
            code_ok = true;
            q.pop_deferred_task(-1);
        }
    });
    server_task t(SERVER_TASK_TYPE_COMPLETION);
    t.id   = q.get_new_id();
    t.lane = SERVER_LANE_CODE;
    const int id = t.id;
    q.defer(std::move(t));

    // a watchdog, so a broken tick fails the test instead of hanging it
    std::atomic<bool> done{false};
    std::thread dog([&]() {
        for (int i = 0; i < 100 && !done; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        q.terminate();
    });
    const auto t0 = std::chrono::steady_clock::now();
    q.start_loop(-1);
    done = true;
    dog.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(n_tick >= 3);
    CHECK(ran == std::vector<int>({id}));
    CHECK(s < 5.0); // about two idle waits of 1 s each
}

// no sleep on idle while the chat window holds tasks (the held request would wait on a sleeping server)
static void test_queue_no_sleep_while_held() {
    server_queue q;
    std::atomic<bool> slept{false};
    std::atomic<int>  n_tick{0};
    q.on_new_task([](server_task &&, bool) { return true; });
    q.on_update_slots([]() {});
    q.on_sleeping_state([&](bool sleeping) { if (sleeping) { slept = true; } });
    q.on_pick_deferred([](const std::deque<server_task> &) { return -1; });
    q.on_tick([&]() {
        if (++n_tick == 3) {
            q.terminate();
        }
    });
    server_task t(SERVER_TASK_TYPE_COMPLETION);
    t.id   = q.get_new_id();
    t.lane = SERVER_LANE_CODE;
    q.defer(std::move(t));

    std::atomic<bool> done{false};
    std::thread dog([&]() {
        for (int i = 0; i < 50 && !done; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        q.terminate();
    });
    q.start_loop(0); // sleep at once when idle, unless tasks are held
    done = true;
    dog.join();
    CHECK(!slept);
    CHECK(n_tick >= 3);
}

// a child task (n_cmpl > 1) is in its parent's lane: it must not run lane-less beside a held or cut parent
static void test_child_lane() {
    server_task parent(SERVER_TASK_TYPE_COMPLETION);
    parent.id   = 7;
    parent.lane = SERVER_LANE_CODE;
    parent.add_child(parent.id, 8);
    CHECK(parent.child_tasks.size() == 1);
    CHECK(parent.child_tasks[0].is_child());
    CHECK(parent.child_tasks[0].lane == SERVER_LANE_CODE);
}

int main() {
    test_lane_of();
    test_should_cut();
    test_pick();
    test_window();
    test_window_generated();
    test_queue_pick();
    test_queue_tick();
    test_queue_no_sleep_while_held();
    test_child_lane();
    test_config();
    if (n_fail > 0) {
        std::fprintf(stderr, "test-server-lane: %d check(s) failed\n", n_fail);
        return 1;
    }
    std::printf("test-server-lane: all checks passed\n");
    return 0;
}

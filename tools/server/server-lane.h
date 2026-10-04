#pragma once

// The chat window (study CHAT-WINDOW-PLAN.md step 1, Tom 2026-10-04): one seat (slot) shared by a person chatting
// and a background job (a coding agent). A chat request takes the seat at once: a background request running on it
// is cut off with a "busy, try again" error (503) at the next batch boundary, and background requests wait until
// --chat-window seconds after the last chat reply ends. Each side keeps its bookmark (prompt cache) while it holds
// the seat; there is no saving or restoring of bookmarks.
//
// Which lane a request is in comes from one header: requests whose --chat-window-header (default X-Lane) carries
// --chat-window-background (default code) are the background lane; every other request, header or not, is chat.
// So only the background client needs configuring (OpenCode: provider options.headers).
//
// Off by default (--chat-window 0): no header is read, no task gets a lane and the server behaves as upstream.
// The rules live here, free of the server, so test-server-lane covers them without a model; server-context.cpp
// applies them, server-queue.cpp gives it the deferred pick and a timed wake.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

enum server_lane : uint8_t {
    SERVER_LANE_NONE = 0, // not looked up: the chat window is off, or not a completion request
    SERVER_LANE_CHAT,     // a person waiting for a reply
    SERVER_LANE_CODE,     // the background lane, marked by the header
};

const char * server_lane_name(server_lane lane);

// The lane of a request from its headers: the header named header_name (ignoring case) whose value is
// background_value (ignoring case and surrounding spaces) gives CODE; any other value, or no such header, gives CHAT.
server_lane server_lane_of(const std::map<std::string, std::string> & headers,
        const std::string & header_name, const std::string & background_value);

// A request arriving in one lane while the seat runs another: cut the running one off? Only chat over code.
bool server_lane_should_cut(server_lane running, server_lane arriving);

// Which deferred task (lanes in queue order, oldest first) moves next: the oldest chat task; else, when code may
// start, the oldest task; else the oldest task that is not code. -1 leaves them all waiting.
int server_lane_pick(const std::vector<server_lane> & deferred, bool code_may_start);

// The window, on an injected clock (milliseconds). Chat holds the seat while a chat task runs and for hold_ms after
// the last one ends; background tasks may start only outside that. hold_ms <= 0 is off: code may always start.
struct server_lane_window {
    explicit server_lane_window(int64_t hold_ms = 0) : hold_ms(hold_ms) {}

    bool on() const { return hold_ms > 0; }

    // a chat task left its seat (finished, failed or cancelled): the window runs hold_ms from now
    void on_chat_done(int64_t now_ms);

    // may a background task start now? chat_running: a chat task holds a seat at this moment
    bool code_may_start(int64_t now_ms, bool chat_running) const;

    // true exactly once per window, on the first call after it has lapsed: the server then starts held tasks,
    // since nothing else would wake them (no seat is released when a window runs out)
    bool take_lapse(int64_t now_ms, bool chat_running);

    // seconds left until code may start, for the log: -1 while a chat task holds a seat
    int64_t seconds_left(int64_t now_ms, bool chat_running) const;

    int64_t hold_ms;
    int64_t open_until_ms = 0; // code may start from here (once no chat task runs)
    bool    open          = false; // a chat has ended and take_lapse() has not yet reported the window's end
};

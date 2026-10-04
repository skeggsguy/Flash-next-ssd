#include "server-lane.h"

#include <algorithm>
#include <cctype>

const char * server_lane_name(server_lane lane) {
    switch (lane) {
        case SERVER_LANE_CHAT: return "chat";
        case SERVER_LANE_CODE: return "code";
        default:               return "none";
    }
}

static std::string lower_trimmed(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) {
        b++;
    }
    while (e > b && std::isspace((unsigned char) s[e - 1])) {
        e--;
    }
    std::string out = s.substr(b, e - b);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return out;
}

server_lane server_lane_of(const std::map<std::string, std::string> & headers,
        const std::string & header_name, const std::string & background_value) {
    // the map keeps the client's spelling of each name, so look at every entry
    const std::string name = lower_trimmed(header_name);
    const std::string bg   = lower_trimmed(background_value);
    for (const auto & [key, value] : headers) {
        if (lower_trimmed(key) == name) {
            return !bg.empty() && lower_trimmed(value) == bg ? SERVER_LANE_CODE : SERVER_LANE_CHAT;
        }
    }
    return SERVER_LANE_CHAT;
}

bool server_lane_should_cut(server_lane running, server_lane arriving) {
    return running == SERVER_LANE_CODE && arriving == SERVER_LANE_CHAT;
}

int server_lane_pick(const std::vector<server_lane> & deferred, bool code_may_start) {
    for (size_t i = 0; i < deferred.size(); i++) {
        if (deferred[i] == SERVER_LANE_CHAT) {
            return (int) i;
        }
    }
    for (size_t i = 0; i < deferred.size(); i++) {
        if (code_may_start || deferred[i] != SERVER_LANE_CODE) {
            return (int) i;
        }
    }
    return -1;
}

void server_lane_window::on_chat_done(int64_t now_ms) {
    if (!on()) {
        return;
    }
    open_until_ms = std::max(open_until_ms, now_ms + hold_ms);
    open          = true;
}

bool server_lane_window::code_may_start(int64_t now_ms, bool chat_running) const {
    return !on() || (!chat_running && now_ms >= open_until_ms);
}

bool server_lane_window::take_lapse(int64_t now_ms, bool chat_running) {
    if (!open || !code_may_start(now_ms, chat_running)) {
        return false;
    }
    open = false;
    return true;
}

int64_t server_lane_window::seconds_left(int64_t now_ms, bool chat_running) const {
    if (code_may_start(now_ms, chat_running)) {
        return 0;
    }
    if (chat_running) {
        return -1;
    }
    return (open_until_ms - now_ms + 999) / 1000;
}

#pragma once

// The p/q record's helpers (common/speculative-pq-record.h), shared by its two sources: the clock, and JSON numbers
// and id lists.

#include "llama.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace pq_util {

inline int64_t steady_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}


inline void add(std::string & out, const char * fmt, double v) {
    if (!std::isfinite(v)) { // JSON has no inf or nan; these are the words Python's json reads (sim/pq_record.py)
        out += std::isnan(v) ? "NaN" : v > 0 ? "Infinity" : "-Infinity";
        return;
    }
    char buf[40];
    snprintf(buf, sizeof(buf), fmt, v);
    out += buf;
}

inline void add_int(std::string & out, int64_t v) {
    char buf[24];
    snprintf(buf, sizeof(buf), "%" PRId64, v);
    out += buf;
}

inline void add_ids(std::string & out, const std::vector<llama_token> & ids) {
    out += '[';
    for (size_t i = 0; i < ids.size(); i++) {
        if (i > 0) {
            out += ',';
        }
        add_int(out, ids[i]);
    }
    out += ']';
}

} // namespace pq_util

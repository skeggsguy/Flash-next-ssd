#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Adaptive draft depth controller for MTP speculative decoding (draft-mtp-adaptive).
//
// Hysteresis state machine with a climb counter and a weighted drop-pressure
// accumulator. The depth N climbs one step after N_CLIMB(N) consecutive verifies
// that accepted every drafted token. The climb cost is low at the floor and at
// depth, high in the middle: 2 at depth 1, 4 at depth 2, 10 at depth 3, then
// 6/3/2/2 from depth 4 upward. Getting from the floor to depth 3 needs only 6
// full accepts, but pushing past 3 (where prose acceptance collapses) costs 10
// full accepts of 3-token drafts, which predictable content clears quickly and
// marginal content never does. Any miss adds (n_draft - n_accepted) to a
// drop-pressure accumulator; when it reaches depth * 5 the depth drops one step
// and the pressure resets. A near miss (n_draft-1) adds 1, a total miss adds
// n_draft, so high depths fall quickly while low depths hold. The drop budget
// scales with depth but never drops below 20, so shallow depths shed bad content
// quickly without collapsing to the floor on a few bad rounds; deep depths hold
// a little longer. At the floor no pressure accumulates at all. The depth starts
// at the floor max(1, --spec-draft-n-min-adaptive) and stays in
// [floor, n_max]; --spec-draft-n-max bounds the upper end of the adaptive
// range.
//
// LLAMA_SPEC_ADAPTIVE_CLIMB replaces the climb table: comma-separated thresholds
// for depth 1, 2, 3, ..., the last one covering every deeper depth (e.g.
// "2,4,3,3,2" lowers the 3->4 barrier from 10 to 3 for content that is
// predictable throughout, such as tool calls). Unset keeps the table below; a
// value that does not parse as positive integers aborts rather than silently
// running the default. The drop rule is unchanged.
struct common_speculative_adaptive {
    int n_cur   = 0; // current adaptive draft depth N
    int n_climb = 0; // consecutive verifies that accepted every drafted token
    int n_drop  = 0; // accumulated drop pressure: sum of (n_draft - n_accepted)

    // consecutive full accepts needed to climb one step from depth N; low at the
    // floor and at depth, high in the middle where acceptance is marginal
    static int climb_threshold(int depth) {
        return climb_threshold(depth, climb_override());
    }

    // the same, against an explicit override table (empty: the built-in one)
    static int climb_threshold(int depth, const std::vector<int> & table) {
        if (!table.empty()) {
            const size_t i = (size_t) std::max(1, depth) - 1;
            return table[std::min(i, table.size() - 1)];
        }
        switch (depth) {
            case 1: return 2;
            case 2: return 4;
            case 3: return 10; // hardened 3->4 barrier: keeps prose/reasoning pinned
            case 4: return 6;
            case 5: return 3;
            case 6: return 2;
            default: return 2; // depth >= 7
        }
    }

    // LLAMA_SPEC_ADAPTIVE_CLIMB, read once; empty means the built-in table
    static const std::vector<int> & climb_override() {
        static const std::vector<int> table = parse_climb(std::getenv("LLAMA_SPEC_ADAPTIVE_CLIMB"));
        return table;
    }

    // "2,4,3" -> {2, 4, 3}; null or empty -> {}; anything else that is not a list
    // of positive integers aborts
    static std::vector<int> parse_climb(const char * s) {
        std::vector<int> table;
        if (s == nullptr || *s == '\0') {
            return table;
        }
        const char * p = s;
        while (true) {
            char * end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (end == p || v <= 0 || v > 1000000) {
                fprintf(stderr, "LLAMA_SPEC_ADAPTIVE_CLIMB=\"%s\": expected positive integers separated by commas\n", s);
                std::abort();
            }
            table.push_back((int) v);
            if (*end == '\0') {
                break;
            }
            if (*end != ',') {
                fprintf(stderr, "LLAMA_SPEC_ADAPTIVE_CLIMB=\"%s\": expected positive integers separated by commas\n", s);
                std::abort();
            }
            p = end + 1;
        }
        return table;
    }

    // accumulated (n_draft - n_accepted) needed to drop one step from depth N;
    // scaled by depth, with a floor so shallow depths do not collapse too fast
    static int drop_pressure(int depth) {
        return std::max(depth * 5, 20);
    }

    // reset to the floor max(1, n_min_adaptive), bounded by the ceiling n_max;
    // the controller climbs from there once acceptance feedback arrives
    void reset(int n_max, int n_min_adaptive) {
        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        n_cur   = std::min(floor, cap);
        n_climb = 0;
        n_drop  = 0;
    }

    // feed one verification result: n_draft is the number of tokens this
    // implementation drafted, n_accepted the number the target accepted
    void update(int n_draft, int n_accepted, int n_max, int n_min_adaptive) {
        if (n_draft <= 0) {
            return;
        }

        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        if (n_accepted == n_draft) {
            n_drop = 0;

            // full acceptance: reset the drop pressure, accumulate the climb streak
            if (n_cur < cap && ++n_climb >= climb_threshold(n_cur)) {
                n_cur++;
                n_climb = 0;
            }
        } else {
            n_climb = 0;

            // any miss adds (n_draft - n_accepted) to the drop pressure; drop one
            // step when the accumulated pressure reaches the depth-scaled budget
            if (n_cur > floor) {
                n_drop += n_draft - n_accepted;
                if (n_drop >= drop_pressure(n_cur)) {
                    n_cur--;
                    n_drop = 0;
                }
            }
        }
    }
};

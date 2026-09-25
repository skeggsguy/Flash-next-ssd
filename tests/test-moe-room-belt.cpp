// The reading room's belt (src/llama-moe-room-belt.cpp): where parts go, and when their memory may be
// used again.
//
// The belt fails silently when it is wrong: a part placed over one the GPU still reads, or memory handed
// out while a runner's read is still landing in it, gives the GEMM the wrong books and changes the words
// without crashing. So besides the hand-made cases, seeded random sequences of every call are checked
// against an interval-set oracle after each step: held parts never overlap, sit inside the belt at a
// multiple of their stride, and a part is let go only when it is handed back (or cancelled), is the
// oldest, and has no read landing.

#include "testing.h"

#include "../src/llama-moe-room-belt.h"

#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

// the test's own account of one part, kept from what it told the belt, not from the belt's answers
struct shadow {
    size_t   offs = 0, bytes = 0, stride = 0;
    int32_t  to_begin = 0; // reads not yet begun
    int32_t  pending  = 0; // reads not yet landed
    int32_t  inflight = 0;
    bool     gone     = false; // reclaimed
    llama_moe_part_state state = LLAMA_MOE_PART_FILLING;
};

// held parts never overlap, stay inside the belt, and start at a multiple of their stride
bool intervals_ok(const llama_moe_belt & belt) {
    std::map<size_t, size_t> iv; // offs -> end
    for (const auto & p : belt.parts) {
        if (p.offs % p.stride != 0 || p.offs + p.bytes > belt.size || p.bytes != (size_t) p.n_records*p.stride) {
            return false;
        }
        auto it = iv.lower_bound(p.offs);
        if ((it != iv.end() && it->first < p.offs + p.bytes) || (it != iv.begin() && std::prev(it)->second > p.offs)) {
            return false;
        }
        iv[p.offs] = p.offs + p.bytes;
    }
    return true;
}

void random_sequence(testing & t, uint32_t seed) {
    std::mt19937 rng(seed);
    const size_t strides[] = { 1024, 1536, 3072 };
    llama_moe_belt belt(24*3072);
    std::map<uint64_t, shadow> sh;
    uint64_t last_gone = 0;
    size_t   max_bytes = 0;
    bool     ok        = true;

    auto pick = [&](auto pred) -> uint64_t {
        std::vector<uint64_t> c;
        for (auto & [seq, s] : sh) {
            if (!s.gone && pred(s)) {
                c.push_back(seq);
            }
        }
        return c.empty() ? 0 : c[rng() % c.size()];
    };

    for (int step = 0; step < 3000 && ok; step++) {
        const uint32_t op = rng() % 100;
        if (op < 20) {
            const size_t   stride = strides[rng() % 3];
            const uint32_t n      = 1 + rng() % 8;
            const int32_t  reads  = (int32_t) n*2;
            max_bytes = std::max(max_bytes, (size_t) n*stride);
            size_t where = 0;
            const bool fits = belt.place(n, stride, where);
            llama_moe_part * p = belt.push(0, 0, n, stride, reads);
            ok = ok && fits == (p != nullptr) && (!p || p->offs == where);
            if (p) {
                sh[p->seq] = { p->offs, p->bytes, stride, reads, reads, 0, false, LLAMA_MOE_PART_FILLING };
            } else {
                // it failed on a belt holding parts, and not for want of more than two parts' waste
                ok = ok && !belt.parts.empty() && belt.held_bytes() + 2*max_bytes + (belt.parts.size() + 1)*3072 >= belt.size;
            }
        } else if (op < 45) {
            const uint64_t seq = pick([](const shadow & s) { return s.to_begin > 0 && s.state == LLAMA_MOE_PART_FILLING; });
            if (seq) {
                ok = ok && belt.begin_read(seq) != nullptr;
                sh[seq].to_begin--;
                sh[seq].inflight++;
            }
        } else if (op < 70) {
            const uint64_t seq = pick([](const shadow & s) { return s.inflight > 0; });
            if (seq) {
                shadow & s = sh[seq];
                // pending counts reads that landed, so after a failed one it never reaches 0
                const bool read_ok = rng() % 50 != 0;
                const bool want    = read_ok && s.state == LLAMA_MOE_PART_FILLING && s.pending == 1;
                ok = ok && belt.end_read(seq, read_ok) == want;
                s.inflight--;
                if (s.state == LLAMA_MOE_PART_FILLING && read_ok) {
                    s.pending--;
                }
                if (want) {
                    s.state = LLAMA_MOE_PART_READY;
                }
            }
        } else if (op < 78) {
            const uint64_t seq = pick([](const shadow & s) { return s.state == LLAMA_MOE_PART_READY; });
            if (seq) {
                belt.find(seq)->state = LLAMA_MOE_PART_IN_USE; // what the part op does
                sh[seq].state = LLAMA_MOE_PART_IN_USE;
            }
        } else if (op < 86) {
            const uint64_t through = pick([](const shadow & s) {
                return s.state == LLAMA_MOE_PART_READY || s.state == LLAMA_MOE_PART_IN_USE; });
            if (through) {
                belt.release_through(through);
                for (auto & [seq, s] : sh) {
                    if (seq <= through && (s.state == LLAMA_MOE_PART_READY || s.state == LLAMA_MOE_PART_IN_USE)) {
                        s.state = LLAMA_MOE_PART_RELEASED;
                    }
                }
            }
        } else if (op < 88) {
            belt.cancel_all();
            for (auto & [seq, s] : sh) {
                if (!s.gone && s.state != LLAMA_MOE_PART_RELEASED) {
                    s.state = LLAMA_MOE_PART_CANCELLED;
                }
            }
        } else if (op < 98) {
            belt.reclaim();
        } else {
            // a read queued for a part that is gone or cancelled must not happen
            const uint64_t seq = pick([](const shadow & s) { return s.state == LLAMA_MOE_PART_CANCELLED; });
            if (seq) {
                ok = ok && belt.begin_read(seq) == nullptr;
            }
        }

        // the oracle: what the belt holds against what the test knows
        ok = ok && intervals_ok(belt);
        for (auto & [seq, s] : sh) {
            if (s.gone) {
                ok = ok && belt.find(seq) == nullptr;
                continue;
            }
            const llama_moe_part * p = belt.find(seq);
            if (p == nullptr) {
                // let go: only when handed back or cancelled, with no read landing, oldest first
                ok = ok && (s.state == LLAMA_MOE_PART_RELEASED || s.state == LLAMA_MOE_PART_CANCELLED) &&
                     s.inflight == 0 && seq > last_gone;
                s.gone    = true;
                last_gone = seq;
                continue;
            }
            ok = ok && p->state == s.state && p->inflight == s.inflight && p->offs == s.offs;
        }
        if (!ok) {
            t.assert_true("seed " + std::to_string(seed) + " step " + std::to_string(step) + ": belt matches the oracle", false);
        }
    }
    t.assert_true("seed " + std::to_string(seed), ok);
}

} // namespace

int main(int argc, char ** argv) {
    testing t;
    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    t.verbose = verbose && std::string(verbose) == "1";
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("alignment", [](testing & t) {
        llama_moe_belt belt(100000);
        llama_moe_part * a = belt.push(0, 0, 3, 1000, 6);
        llama_moe_part * b = belt.push(1, 0, 2, 3000, 4); // a q5_K floor after a q4_K one
        t.assert_equal("first at 0", (size_t) 0, a->offs);
        t.assert_equal("next at the first multiple of its stride past 3000", (size_t) 3000, b->offs);
        t.assert_equal("its records are whole indices of its floor's view", 1u, b->first_record());
    });

    t.test("wrap and FIFO", [](testing & t) {
        llama_moe_belt belt(10000);
        const uint64_t s1 = belt.push(0, 0, 4, 1000, 0)->seq;  // [0, 4000), no reads: READY at once
        const uint64_t s2 = belt.push(0, 1, 4, 1000, 0)->seq;  // [4000, 8000)
        t.assert_true("no room for 3000 more", belt.push(0, 2, 3, 1000, 0) == nullptr);
        belt.release_through(s2 - 1);
        belt.find(s2)->state = LLAMA_MOE_PART_IN_USE;
        t.assert_equal("the handed-back oldest part is let go", (size_t) 1, belt.reclaim());
        llama_moe_part * p = belt.push(0, 2, 3, 1000, 0);
        t.assert_true("wraps to the start", p != nullptr && p->offs == 0);
        t.assert_true("full again", belt.push(0, 3, 2, 1000, 0) == nullptr);

        if (p == nullptr) {
            return;
        }
        belt.release_through(p->seq); // p is newer than s2, which is in use: both handed back
        t.assert_true("in use too", belt.find(s2)->state == LLAMA_MOE_PART_RELEASED);
        t.assert_equal("oldest first, then the rest", (size_t) 2, belt.reclaim());
        t.assert_true("s1 is gone for good", belt.find(s1) == nullptr && belt.begin_read(s1) == nullptr);
        const llama_moe_part * q = belt.push(0, 3, 10, 1000, 0);
        t.assert_true("empty belt starts over", q != nullptr && q->offs == 0);
    });

    t.test("FIFO: a newer part handed back waits for the older", [](testing & t) {
        llama_moe_belt belt(10000);
        llama_moe_part * a = belt.push(0, 0, 2, 1000, 0);
        const uint64_t   b = belt.push(0, 1, 2, 1000, 0)->seq;
        a->state = LLAMA_MOE_PART_IN_USE;
        belt.find(b)->state = LLAMA_MOE_PART_RELEASED;
        t.assert_equal("nothing let go", (size_t) 0, belt.reclaim());
        t.assert_equal("both still held", (size_t) 4000, belt.held_bytes());
    });

    t.test("a cancelled part keeps its memory while a read lands", [](testing & t) {
        llama_moe_belt belt(10000);
        const uint64_t s = belt.push(0, 0, 2, 1000, 4)->seq;
        t.assert_true("read begins", belt.begin_read(s) != nullptr);
        belt.cancel_all();
        t.assert_true("a queued read of it is stale", belt.begin_read(s) == nullptr);
        t.assert_equal("held while the read lands", (size_t) 0, belt.reclaim());
        t.assert_true("nothing new over it", belt.push(1, 0, 10, 1000, 0) == nullptr);
        t.assert_true("the landing read never makes it ready", !belt.end_read(s, true));
        t.assert_equal("then let go", (size_t) 1, belt.reclaim());
    });

    t.test("a failed read never makes a part ready", [](testing & t) {
        llama_moe_belt belt(10000);
        const uint64_t s = belt.push(0, 0, 1, 1000, 2)->seq;
        belt.begin_read(s);
        belt.begin_read(s);
        t.assert_true("first lands", !belt.end_read(s, true));
        t.assert_true("second fails", !belt.end_read(s, false));
        t.assert_true("still filling", belt.find(s)->state == LLAMA_MOE_PART_FILLING);
        const uint64_t r = belt.push(0, 1, 1, 1000, 1)->seq;
        belt.begin_read(r);
        t.assert_true("last read makes it ready", belt.end_read(r, true) && belt.find(r)->state == LLAMA_MOE_PART_READY);
        t.assert_true("an end without a begin is ignored", !belt.end_read(r, true));
    });

    t.test("random sequences against the oracle", [](testing & t) {
        for (uint32_t seed = 1; seed <= 200; seed++) {
            random_sequence(t, seed);
        }
    });

    return t.summary();
}

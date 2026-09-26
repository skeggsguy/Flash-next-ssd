// The lent belt's bookkeeping (src/llama-moe-room-lend.cpp): where copies of put-back books go on the
// belt, which ones are dropped to make room, and which one a later trip may copy back.
//
// The lent belt fails silently when it is wrong: a copy placed over one a restore still reads, a copy
// found before its slabs are all copied, or a copy found after a newer one was written over it hands the
// desk the wrong bytes for a book, and the words change without a crash. So besides the hand-made cases,
// seeded random sequences of every call are checked against an interval oracle after each step: copies
// never overlap, sit inside the belt at 256 B, go first in first out, a busy copy is never dropped, and a
// copy that is found still holds exactly the bytes its own slabs wrote (a byte-level shadow of the belt).

#include "testing.h"

#include "../src/llama-moe-room-lend.h"

#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

// the test's own account of one copy, kept from what it told the ring, not from the ring's answers
struct shadow {
    int32_t il = -1, book = -1;
    size_t  offs = 0, bytes = 0;
    int32_t n_slabs = 0;
    uint8_t slab[LLAMA_MOE_LEND_SLABS_MAX] = {};
    int32_t pins = 0;
    bool    gone = false;

    bool complete() const {
        for (int32_t i = 0; i < n_slabs; i++) {
            if (slab[i] != LLAMA_MOE_LEND_DONE) {
                return false;
            }
        }
        return true;
    }
    bool busy() const { return pins > 0 || !complete(); }
    // slab i's bytes: the record cut into n_slabs pieces, as the manager lays a book's weights out
    size_t slab_begin(int32_t i) const { return offs + bytes*i/n_slabs; }
    size_t slab_end(int32_t i)   const { return offs + bytes*(i + 1)/n_slabs; }
};

bool overlap(size_t a0, size_t a1, size_t b0, size_t b1) {
    return a0 < b1 && b0 < a1;
}

void random_sequence(testing & t, uint32_t seed) {
    std::mt19937 rng(seed);
    const size_t  size      = 40000 + (seed % 7)*1000; // not always a multiple of 256
    const size_t  records[] = { 1000, 1536, 3072, 4500 };
    llama_moe_lend lend(size);
    std::map<uint64_t, shadow> sh;
    std::vector<uint64_t> mem(size, 0); // which copy's slab last wrote each byte of the belt
    size_t o_head = 0;                  // the end of the newest copy the oracle placed
    bool   ok     = true;
    std::string why_failed;

    auto fail = [&](const std::string & what) {
        if (ok) {
            why_failed = what;
        }
        ok = false;
    };
    auto live = [&](auto pred) -> uint64_t {
        std::vector<uint64_t> c;
        for (auto & [seq, s] : sh) {
            if (!s.gone && pred(s)) {
                c.push_back(seq);
            }
        }
        return c.empty() ? 0 : c[rng() % c.size()];
    };

    for (int step = 0; step < 4000 && ok; step++) {
        const uint32_t op = rng() % 100;
        if (op < 30) {
            const int32_t il = rng() % 3, book = rng() % 12, n_slabs = 2 + rng() % 2;
            const size_t  bytes = rng() % 60 == 0 ? size + 1 : records[rng() % 4];

            // the oracle: held, or else where it goes, what it drops and whether a busy copy blocks it
            uint64_t held = 0;
            for (auto & [seq, s] : sh) {
                held = !s.gone && s.il == il && s.book == book ? seq : held;
            }
            size_t at = (o_head + 255)/256*256, skip = size;
            if (at + bytes > size) {
                skip = std::min(at, size);
                at   = 0;
            }
            std::vector<uint64_t> in_way;
            bool blocked = bytes > size;
            for (auto & [seq, s] : sh) {
                if (!s.gone && (overlap(s.offs, s.offs + s.bytes, at, at + bytes) || overlap(s.offs, s.offs + s.bytes, skip, size))) {
                    in_way.push_back(seq);
                    blocked = blocked || s.busy();
                }
            }
            const size_t n_before = lend.ring.size();
            llama_moe_lend_keep why;
            llama_moe_lend_copy * c = lend.keep(il, book, 5, bytes, n_slabs, why);
            if (held) {
                if (c != nullptr || why != LLAMA_MOE_LEND_HELD) {
                    fail("a held book is not copied twice");
                }
            } else if (blocked) {
                if (c != nullptr || why != LLAMA_MOE_LEND_BLOCKED || lend.ring.size() != n_before) {
                    fail("a busy copy in the way blocks the keep and changes nothing");
                }
            } else if (c == nullptr || why != LLAMA_MOE_LEND_KEPT || c->offs != at || c->bytes != bytes) {
                fail("kept at the oracle's place");
            } else {
                for (const uint64_t seq : in_way) {
                    sh[seq].gone = true;
                }
                shadow s;
                s.il = il; s.book = book; s.offs = at; s.bytes = bytes; s.n_slabs = n_slabs;
                sh[c->seq] = s;
                o_head = at + bytes;
            }
        } else if (op < 50) {
            // whoever copies a slab claims it first; a second claim of the same slab is refused
            const uint64_t seq = live([](const shadow & s) { return !s.complete(); });
            if (seq) {
                shadow & s = sh[seq];
                int32_t i = rng() % s.n_slabs;
                const bool want = s.slab[i] == LLAMA_MOE_LEND_PENDING;
                if (lend.claim(seq, i) != want) {
                    fail("claim takes a PENDING slab only");
                }
                if (want) {
                    s.slab[i] = LLAMA_MOE_LEND_RUNNING;
                }
            }
        } else if (op < 70) {
            const uint64_t seq = live([](const shadow & s) {
                for (int32_t i = 0; i < s.n_slabs; i++) {
                    if (s.slab[i] == LLAMA_MOE_LEND_RUNNING) {
                        return true;
                    }
                }
                return false;
            });
            if (seq) {
                shadow & s = sh[seq];
                int32_t i = 0;
                while (s.slab[i] != LLAMA_MOE_LEND_RUNNING) {
                    i++;
                }
                std::fill(mem.begin() + s.slab_begin(i), mem.begin() + s.slab_end(i), seq); // the copy itself
                s.slab[i] = LLAMA_MOE_LEND_DONE;
                if (lend.finish(seq, i) != s.complete()) {
                    fail("finish says when the copy is complete");
                }
            }
        } else if (op < 78) {
            const uint64_t seq = live([](const shadow & s) { return s.complete(); }); // a restore reads a complete copy
            if (seq) {
                lend.pin(seq);
                sh[seq].pins++;
            }
        } else if (op < 86) {
            const uint64_t seq = live([](const shadow & s) { return s.pins > 0; });
            if (seq) {
                lend.unpin(seq);
                sh[seq].pins--;
            }
        } else if (op < 98) {
            // a trip looks the book up: only a complete copy, and its bytes are its own
            const int32_t il = rng() % 3, book = rng() % 12;
            uint64_t want = 0;
            for (auto & [seq, s] : sh) {
                want = !s.gone && s.il == il && s.book == book && s.complete() ? seq : want;
            }
            const llama_moe_lend_copy * c = lend.find(il, book);
            if ((c ? c->seq : 0) != want) {
                fail("find gives the book's complete copy, and nothing else");
            } else if (c != nullptr) {
                for (size_t b = c->offs; b < c->offs + c->bytes; b++) {
                    if (mem[b] != c->seq) {
                        fail("a found copy holds its own bytes");
                        break;
                    }
                }
            }
        } else if (!lend.busy()) {
            // the room takes the belt back: nothing may be busy, and afterwards nothing is found
            lend.clear();
            for (auto & [seq, s] : sh) {
                s.gone = true;
            }
            o_head = 0;
            for (int32_t il = 0; il < 3; il++) {
                for (int32_t book = 0; book < 12; book++) {
                    if (lend.find(il, book) != nullptr) {
                        fail("nothing is found after a clear");
                    }
                }
            }
        }

        // the oracle: the ring holds exactly the live copies, in order, at 256 B, inside the belt, apart
        size_t n_live = 0, n_complete = 0;
        bool   busy = false;
        for (auto & [seq, s] : sh) {
            const llama_moe_lend_copy * c = lend.at(seq);
            if (s.gone) {
                if (c != nullptr) {
                    fail("a dropped copy is gone for good");
                }
                continue;
            }
            n_live++;
            n_complete += s.complete();
            busy = busy || s.busy();
            if (c == nullptr) {
                fail(s.busy() ? "a busy copy is never dropped" : "a copy is dropped only when one is placed over it");
                continue;
            }
            if (c->offs != s.offs || c->pins != s.pins || c->offs % LLAMA_MOE_LEND_ALIGN != 0 || c->offs + c->bytes > size) {
                fail("a live copy stays where it was put, at 256 B, inside the belt");
            }
            for (int32_t i = 0; i < s.n_slabs; i++) {
                if (c->slab[i] != s.slab[i]) {
                    fail("slab states as the oracle's");
                }
            }
        }
        for (size_t i = 0; i < lend.ring.size(); i++) {
            const llama_moe_lend_copy & a = lend.ring[i];
            if (i > 0 && a.seq != lend.ring[i - 1].seq + 1) {
                fail("the ring's seqs are consecutive, oldest first");
            }
            for (size_t j = i + 1; j < lend.ring.size(); j++) {
                if (overlap(a.offs, a.offs + a.bytes, lend.ring[j].offs, lend.ring[j].offs + lend.ring[j].bytes)) {
                    fail("copies never overlap");
                }
            }
        }
        if (lend.ring.size() != n_live || lend.held() != n_complete || lend.busy() != busy) {
            fail("held, busy and the ring's size agree with the oracle");
        }
    }
    t.assert_true("seed " + std::to_string(seed) + (ok ? "" : ": " + why_failed), ok);
}

} // namespace

int main(int argc, char ** argv) {
    testing t;
    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    t.verbose = verbose && std::string(verbose) == "1";
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    auto done = [](llama_moe_lend & lend, llama_moe_lend_copy * c) {
        for (int32_t i = 0; c && i < c->n_slabs; i++) {
            lend.claim(c->seq, i);
            lend.finish(c->seq, i);
        }
    };

    t.test("placement at 256 B, first in first out", [&](testing & t) {
        llama_moe_lend lend(10000);
        llama_moe_lend_keep why;
        llama_moe_lend_copy * a = lend.keep(0, 1, 0, 1000, 3, why);
        t.assert_true("kept at 0", a && why == LLAMA_MOE_LEND_KEPT && a->offs == 0);
        done(lend, a);
        llama_moe_lend_copy * b = lend.keep(1, 1, 0, 3000, 2, why);
        t.assert_true("next at the first 256 B past 1000", b && b->offs == 1024);
        done(lend, b);
        const uint64_t sa = a->seq, sb = b->seq;
        llama_moe_lend_copy * c = lend.keep(0, 2, 0, 5000, 3, why); // [4096, 9096)
        done(lend, c);
        t.assert_true("fits before the end", c && c->offs == 4096 && lend.at(sa) && lend.at(sb));
        llama_moe_lend_copy * d = lend.keep(0, 3, 0, 1500, 3, why); // wraps: [9216, 10000) is given up
        t.assert_true("wraps to the start", d && d->offs == 0);
        t.assert_true("dropping the oldest in its way only", !lend.at(sa) && !lend.at(sb) && lend.at(c->seq));
        t.assert_equal("two dropped", (int64_t) 2, lend.n_dropped);
        t.assert_true("a dropped book is not found", lend.find(0, 1) == nullptr);
    });

    t.test("only a complete copy is found, and a book is never copied twice", [&](testing & t) {
        llama_moe_lend lend(10000);
        llama_moe_lend_keep why;
        llama_moe_lend_copy * a = lend.keep(2, 7, 4, 2000, 2, why);
        const uint64_t s = a->seq;
        t.assert_true("pending: not found", lend.find(2, 7) == nullptr);
        t.assert_true("a pending book is not kept again", !lend.keep(2, 7, 9, 2000, 2, why) && why == LLAMA_MOE_LEND_HELD);
        t.assert_true("claim once", lend.claim(s, 0) && !lend.claim(s, 0));
        t.assert_true("half copied", !lend.finish(s, 0) && lend.find(2, 7) == nullptr);
        t.assert_true("finish needs a claim", !lend.finish(s, 1));
        t.assert_true("claim, then the last slab completes it", lend.claim(s, 1) && lend.finish(s, 1));
        t.assert_true("found", lend.find(2, 7) == lend.at(s) && lend.at(s)->slot == 4);
        t.assert_true("restored books keep their copy: put back again, not copied", !lend.keep(2, 7, 1, 2000, 2, why) && why == LLAMA_MOE_LEND_HELD);
        t.assert_true("another floor's book of the same number is its own", lend.keep(3, 7, 1, 2000, 2, why) != nullptr);
    });

    t.test("a busy copy is never dropped", [&](testing & t) {
        llama_moe_lend lend(4000);
        llama_moe_lend_keep why;
        llama_moe_lend_copy * a = lend.keep(0, 1, 0, 2000, 2, why);
        const uint64_t s = a->seq;
        llama_moe_lend_copy * b = lend.keep(0, 2, 0, 1500, 2, why);
        done(lend, b);
        t.assert_true("a keep still copying blocks", !lend.keep(0, 3, 0, 2000, 2, why) && why == LLAMA_MOE_LEND_BLOCKED);
        t.assert_true("and changes nothing", lend.ring.size() == 2 && lend.busy());
        done(lend, lend.at(s));
        lend.pin(s);
        t.assert_true("a restore reading it blocks too", !lend.keep(0, 3, 0, 2000, 2, why) && why == LLAMA_MOE_LEND_BLOCKED);
        lend.unpin(s);
        t.assert_true("then it goes", lend.keep(0, 3, 0, 2000, 2, why) != nullptr && lend.at(s) == nullptr);
        t.assert_true("bigger than the belt: blocked", !lend.keep(0, 4, 0, 4001, 2, why) && why == LLAMA_MOE_LEND_BLOCKED);
    });

    t.test("clear is an epoch", [&](testing & t) {
        llama_moe_lend lend(10000);
        llama_moe_lend_keep why;
        llama_moe_lend_copy * a = lend.keep(0, 1, 0, 3000, 2, why);
        done(lend, a);
        const uint64_t s = a->seq;
        lend.keep(0, 2, 0, 3000, 2, why);
        t.assert_true("busy while a keep is pending", lend.busy());
        done(lend, lend.at(s + 1));
        t.assert_true("not busy once copied", !lend.busy() && lend.held() == 2);
        t.assert_equal("two dropped", (size_t) 2, lend.clear());
        t.assert_true("the stale index finds nothing", lend.find(0, 1) == nullptr && lend.at(s) == nullptr);
        llama_moe_lend_copy * b = lend.keep(0, 1, 0, 3000, 2, why);
        t.assert_true("the same book is kept anew, from the start", b && b->offs == 0 && b->seq > s + 1);
        t.assert_true("its new copy is not found until copied", lend.find(0, 1) == nullptr);
        done(lend, b);
        t.assert_true("then found", lend.find(0, 1) == b);
    });

    t.test("the per-call cap and the load line's capacity, at the paperback's records", [](testing & t) {
        const size_t belt = 1542528000, strides = 43*3072000ull + 4*3584000ull + 3993600ull; // 48 floors
        t.assert_equal("a floor's share of the belt: 10 a call", 10u, llama_moe_lend_cap(10, belt, strides));
        t.assert_equal("never under the books a word reads", 16u, llama_moe_lend_cap(16, belt, strides));
        t.assert_equal("~492 books", 492u, llama_moe_lend_capacity(belt, strides, 48));
        t.assert_equal("no floors: the word's books", 8u, llama_moe_lend_cap(8, belt, 0));
    });

    t.test("random sequences against the oracle", [](testing & t) {
        for (uint32_t seed = 1; seed <= 200; seed++) {
            random_sequence(t, seed);
        }
    });

    return t.summary();
}

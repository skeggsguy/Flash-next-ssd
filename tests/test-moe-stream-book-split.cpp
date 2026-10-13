// One missing book across both runners (LLAMA_MOE_STREAM_SPLIT, src/llama-moe-stream-split.h), without a model:
//
//   - the switch: off unless set, an empty value off, clamped to 0..2
//   - the cut of one slab (llama_moe_split_slab) over generated slabs: level 0 is today's job exactly (whole, no
//     runner chosen, the alternation untouched); level 1 one whole job on alternating runners; level 2 two jobs that
//     cover the slab once, on both runners, cut on a 4096-byte file boundary when there is one
//   - the demand queue (llama_moe_stream_queue_demand_locked): the jobs it queues and slot_pending counting them, at
//     each level, with one runner, for a restore from the lent belt (whole slabs, no runner) and at 4 slabs (8 jobs)
//   - the runners themselves on two files that differ on purpose: every half lands at its own offset in the slot,
//     from the runner its job names, the slot turns RESIDENT only with every byte in place, each runner's bytes
//     are counted, on the staging path and with direct reads

#include "testing.h"

#include "llama.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "../src/llama-moe-stream.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

static std::string tmp_path(const char * name) {
    const char * d = std::getenv("TMPDIR");
    return (d && *d ? std::string(d) : std::string("/tmp")) + "/" + name;
}

static void write_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    GGML_ASSERT(f != nullptr && fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size());
    fclose(f);
}

// a floor of n_expert books with these slab sizes, its desk a CPU buffer of n_slots slots; the file's books start
// at `base` and each weight's slabs follow the last weight's (as a GGUF lays out the exps tensors)
struct floor_fixture {
    llama_moe_stream_layer sl;
    ggml_context *         ctx = nullptr;
    ggml_backend_buffer_t  buf = nullptr;

    floor_fixture(uint32_t n_expert, uint32_t n_slots, const std::vector<size_t> & nb, size_t base) {
        ggml_init_params ip = { ggml_tensor_overhead()*(nb.size() + 1), nullptr, true };
        ctx = ggml_init(ip);
        sl.n_expert = n_expert;
        sl.n_slots  = n_slots;
        size_t offs = base;
        for (const size_t n : nb) {
            ggml_tensor * cache = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) (n*n_slots));
            sl.weights.push_back({ cache, 0, offs, n });
            offs += n*n_expert + 32; // the next tensor, 32-aligned as GGUF pads them
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        sl.slot_expert.assign(n_slots, -1);
        sl.slot_state.assign(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        sl.slot_pending.assign(n_slots, 0);
        sl.slot_gen.assign(n_slots, 0);
    }
    ~floor_fixture() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    size_t file_end() const {
        const auto & w = sl.weights.back();
        return w.offs + w.nb_expert*sl.n_expert;
    }
    // reserve slot s for book e, as reserve_slot_locked leaves it
    void reserve(int32_t e, int32_t s) {
        sl.slot_expert[s] = e;
        sl.slot_state[s]  = LLAMA_MOE_STREAM_SLOT_LOADING;
        sl.slot_gen[s]++;
    }
};

static std::vector<llama_moe_stream_work> take_queue(llama_moe_stream & mgr) {
    std::vector<llama_moe_stream_work> q(mgr.q_demand.begin(), mgr.q_demand.end());
    mgr.q_demand.clear();
    return q;
}

int main(int argc, char ** argv) {
    testing t;
    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    t.verbose = verbose && std::string(verbose) == "1";
    if (!t.verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    const char * env = "LLAMA_MOE_STREAM_SPLIT";

    t.test("the_switch_is_off_unless_set", [&](testing & t) {
        unsetenv(env);
        t.assert_equal("unset: 0", 0, llama_moe_stream_split_env());
        for (const auto & [v, want] : std::vector<std::pair<const char *, int>>{
                { "", 0 }, { "0", 0 }, { "1", 1 }, { "2", 2 }, { "3", 2 }, { "-1", 0 }, { "x", 0 } }) {
            setenv(env, v, 1);
            t.assert_equal(std::string("\"") + v + "\"", want, llama_moe_stream_split_env());
        }
        setenv(env, "2", 1);
        {
            llama_moe_stream mgr(1, 8, 1, false);
            t.assert_equal("the manager reads it at load", 2, mgr.split.level);
        }
        unsetenv(env);
        llama_moe_stream mgr(1, 8, 1, false);
        t.assert_equal("unset: the manager is at level 0", 0, mgr.split.level);
    });

    t.test("the_cut_over_generated_slabs", [&](testing & t) {
        std::mt19937_64 rng(11);
        int bad0 = 0, bad1 = 0, bad2 = 0, n_aligned = 0, n_middle = 0;
        for (int i = 0; i < 20000; i++) {
            // the real slab (1,024,000 B at a 32-aligned offset) most of the time, any size and offset otherwise
            const bool real = i % 2 == 0;
            const size_t nb   = real ? 1024000 : (size_t) (rng() % (i % 7 == 1 ? 9000 : 3000000));
            const size_t offs = real ? (size_t) (rng() % (1ull << 36)) & ~(size_t) 31 : (size_t) (rng() % (1ull << 40));
            llama_moe_split_piece p[2];
            uint32_t rr = (uint32_t) rng();
            const uint32_t rr0 = rr;

            int np = llama_moe_split_slab(0, offs, nb, rr, p);
            bad0 += np != 1 || p[0].offs != 0 || p[0].len != 0 || p[0].drive != -1 || rr != rr0;

            np = llama_moe_split_slab(1, offs, nb, rr, p);
            bad1 += np != 1 || p[0].offs != 0 || p[0].len != 0 || p[0].drive != (int8_t) (rr0 & 1) || rr != rr0 + 1;

            np = llama_moe_split_slab(2, offs, nb, rr, p);
            const int8_t first = (int8_t) ((rr0 + 1) & 1);
            if (nb < 2) {
                bad2 += np != 1 || p[0].len != 0 || p[0].drive != first;
                continue;
            }
            const size_t cut = p[0].len;
            const size_t at  = (offs + nb/2) & ~(size_t) 4095;
            bool ok = np == 2 && p[0].offs == 0 && cut > 0 && p[1].offs == cut && p[1].len == nb - cut &&
                      p[0].drive == first && p[1].drive == 1 - first && rr == rr0 + 2;
            if (at > offs) {
                ok = ok && cut == at - offs && (offs + cut) % 4096 == 0 && nb/2 - cut < 4096;
                n_aligned++;
            } else {
                ok = ok && cut == nb/2;
                n_middle++;
            }
            bad2 += !ok;
        }
        t.assert_equal("level 0: today's job, whole, no runner, the alternation untouched", 0, bad0);
        t.assert_equal("level 1: one whole job, on the alternation's runner", 0, bad1);
        t.assert_equal("level 2: two jobs covering the slab once, both runners, the cut on a file block", 0, bad2);
        t.assert_true("both kinds of cut were generated (" + std::to_string(n_aligned) + " on a block, " +
                std::to_string(n_middle) + " at the middle)", n_aligned > 1000 && n_middle > 100);

        // the real book: three slabs of 1,024,000 B, so 6 halves of ~500 KiB, alternate halves first on each runner
        uint32_t rr = 0;
        llama_moe_split_piece p[2];
        for (int s = 0; s < 3; s++) {
            llama_moe_split_slab(2, 4096*1000 + s*1024000, 1024000, rr, p);
            t.assert_true("slab " + std::to_string(s) + ": its first half from runner " + std::to_string(s & 1),
                    p[0].drive == (s & 1) && p[1].drive == 1 - (s & 1) && p[0].len == 512000 && p[1].len == 512000);
        }
    });

    t.test("the_demand_queue", [&](testing & t) {
        floor_fixture F(8, 4, { 1024000, 1024000, 1024000 }, 4096*3 + 96);
        llama_moe_stream mgr(1, 4, 1, false);
        std::unique_lock<std::mutex> lk(mgr.mtx);

        // one runner: every level queues today's jobs
        for (int level = 0; level <= 2; level++) {
            mgr.split.level = level;
            F.reserve(5, 1);
            llama_moe_stream_queue_demand_locked(mgr, F.sl, 5, 1);
            const auto q = take_queue(mgr);
            bool same = q.size() == 3 && F.sl.slot_pending[1] == 3;
            for (size_t i = 0; same && i < q.size(); i++) {
                same = q[i].sl == &F.sl && q[i].expert == 5 && q[i].slot == 1 && q[i].widx == (int32_t) i &&
                       q[i].gen == F.sl.slot_gen[1] && q[i].piece_len == 0 && q[i].piece_offs == 0 && q[i].drive == -1 &&
                       q[i].lent == 0 && q[i].save == 0 && q[i].slot >= 0;
            }
            t.assert_true("one runner, level " + std::to_string(level) + ": a whole job per slab, pending 3", same);
        }

        const std::string probe = tmp_path("moe-book-split-probe");
        write_file(probe, { 1 });
        mgr.files_alt.emplace_back(new llama_file(probe.c_str(), "rb", false)); // any open file: nothing reads it
        // level 0 with two runners: still today's jobs, carrying the lent belt's keep
        mgr.split.level = 0;
        F.reserve(6, 2);
        llama_moe_stream_queue_demand_locked(mgr, F.sl, 6, 2, 0, 77);
        auto q = take_queue(mgr);
        bool ok = q.size() == 3 && F.sl.slot_pending[2] == 3;
        for (size_t i = 0; ok && i < q.size(); i++) {
            ok = q[i].widx == (int32_t) i && q[i].piece_len == 0 && q[i].drive == -1 && q[i].save == 77;
        }
        t.assert_true("level 0, two runners: today's jobs, the keep carried", ok);
        t.assert_true("level 0 counts nothing", mgr.split.n_spread == 0 && mgr.split.n_halved == 0 && mgr.split.rr == 0);

        // level 1: whole jobs on alternating runners, carried on from book to book
        mgr.split.level = 1;
        std::vector<int> drives;
        for (int e : { 3, 4 }) {
            F.reserve(e, 0);
            llama_moe_stream_queue_demand_locked(mgr, F.sl, e, 0);
            for (const auto & w : take_queue(mgr)) {
                ok = ok && w.piece_len == 0 && w.expert == e;
                drives.push_back(w.drive);
            }
            ok = ok && F.sl.slot_pending[0] == 3;
        }
        t.assert_true("level 1: whole jobs, pending 3 a book", ok);
        t.assert_true("level 1: runners 0 1 0 | 1 0 1", drives == std::vector<int>({ 0, 1, 0, 1, 0, 1 }));
        t.assert_equal("level 1: 6 slabs spread", (int64_t) 6, mgr.split.n_spread);

        // level 2: two halves a slab, both runners, pending counts the halves
        mgr.split.level = 2;
        F.reserve(7, 3);
        llama_moe_stream_queue_demand_locked(mgr, F.sl, 7, 3, 0, 9);
        q = take_queue(mgr);
        ok = q.size() == 6 && F.sl.slot_pending[3] == 6;
        for (size_t i = 0; ok && i < q.size(); i += 2) {
            const size_t nb = F.sl.weights[q[i].widx].nb_expert;
            ok = q[i].widx == (int32_t) (i/2) && q[i + 1].widx == q[i].widx && q[i].piece_offs == 0 &&
                 q[i + 1].piece_offs == q[i].piece_len && q[i].piece_len + q[i + 1].piece_len == nb &&
                 q[i].drive + q[i + 1].drive == 1 && q[i].save == 9 && q[i + 1].save == 9 && q[i].gen == F.sl.slot_gen[3];
        }
        t.assert_true("level 2: halves covering each slab, one on each runner, pending 6, the keep on both", ok);
        t.assert_equal("level 2: 3 slabs halved", (int64_t) 3, mgr.split.n_halved);

        // a restore from the lent belt: copied from RAM, whole slabs (one pin each), no runner
        F.reserve(2, 1);
        llama_moe_stream_queue_demand_locked(mgr, F.sl, 2, 1, 42, 0);
        q = take_queue(mgr);
        ok = q.size() == 3 && F.sl.slot_pending[1] == 3;
        for (const auto & w : q) {
            ok = ok && w.lent == 42 && w.piece_len == 0 && w.drive == -1;
        }
        t.assert_true("level 2, a restore: whole slabs, no runner, pending 3", ok);

        // the most slabs a book has (gate, up, down + one): 8 halves, still a uint8_t count
        floor_fixture G(8, 2, { 9000, 9000, 9000, 9000 }, 0);
        G.reserve(1, 0);
        llama_moe_stream_queue_demand_locked(mgr, G.sl, 1, 0);
        t.assert_true("4 slabs: 8 jobs, pending 8", take_queue(mgr).size() == 8 && G.sl.slot_pending[0] == 8);
        mgr.files_alt.clear();
        remove(probe.c_str());
    });

    // the runners on a real file and a second one whose bytes differ on purpose (an alt copy is identical in use;
    // here the difference names which runner served each byte)
    for (const bool direct : { false, true }) {
        t.test(direct ? "the_runners_direct" : "the_runners_staging", [&](testing & t) {
            floor_fixture F(4, 3, { 40000, 24576 + 100 }, 4096*3 + 96);
            std::vector<uint8_t> a(F.file_end() + 4096), b(a.size());
            std::mt19937 rng(direct ? 5 : 6);
            for (size_t i = 0; i < a.size(); i++) {
                a[i] = (uint8_t) rng();
                b[i] = a[i] ^ 0x5A;
            }
            const std::string pa = tmp_path(direct ? "moe-book-split-a-d" : "moe-book-split-a");
            const std::string pb = tmp_path(direct ? "moe-book-split-b-d" : "moe-book-split-b");
            write_file(pa, a);
            write_file(pb, b);

            for (int level = 0; level <= 2; level++) {
                llama_moe_stream mgr(1, 3, /*n_io_threads =*/ 1, direct);
                mgr.files.emplace_back(new llama_file(pa.c_str(), "rb", direct));
                mgr.files_alt.emplace_back(new llama_file(pb.c_str(), "rb", direct));
                mgr.max_nb_expert = 40000;
                mgr.alt_split     = 50; // book 3 of 4 is the alt runner's by id
                mgr.split.level   = level;
                mgr.test_land_delay_us = 2000; // one runner, slow: an early publish would be seen

                std::unique_lock<std::mutex> lk(mgr.mtx);
                mgr.start_workers_locked();
                std::string why;
                for (const int32_t e : { 3, 0, 2 }) {
                    const int32_t s = e % 3;
                    F.reserve(e, s);
                    llama_moe_stream_queue_demand_locked(mgr, F.sl, e, s);
                    std::vector<llama_moe_stream_work> jobs(mgr.q_demand.begin(), mgr.q_demand.end());
                    mgr.cv_done.wait(lk, [&] { return F.sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT; });
                    // the moment it is RESIDENT, every byte of every job is in place, from the runner it names
                    for (const auto & w : jobs) {
                        const auto & wt  = F.sl.weights[w.widx];
                        const size_t len = w.piece_len ? w.piece_len : wt.nb_expert;
                        const bool   alt = w.drive >= 0 ? w.drive == 1 : mgr.use_alt(e, 4); // idle runners: the id split
                        const uint8_t * want = (alt ? b : a).data() + wt.offs + (size_t) e*wt.nb_expert + w.piece_offs;
                        std::vector<uint8_t> got(len);
                        ggml_backend_tensor_get(wt.cache, got.data(), (size_t) s*wt.nb_expert + w.piece_offs, len);
                        if (memcmp(got.data(), want, len) != 0) {
                            why += " book " + std::to_string(e) + " slab " + std::to_string(w.widx) + " @" +
                                   std::to_string(w.piece_offs) + ";";
                        }
                    }
                    t.assert_equal("level " + std::to_string(level) + ", book " + std::to_string(e) + ": " +
                            std::to_string(jobs.size()) + " jobs, nothing pending", 0, (int) F.sl.slot_pending[s]);
                }
                t.assert_true("level " + std::to_string(level) + ": every job's bytes in place at RESIDENT, from its runner" +
                        why, why.empty());
                const int64_t books = 3*(40000 + 24576 + 100);
                t.assert_equal("level " + std::to_string(level) + ": the runners' bytes add up to 3 books",
                        books, mgr.stats.n_bytes_file + mgr.stats.n_bytes_alt);
                if (level == 2) {
                    t.assert_true("level 2: both runners read, near half each",
                            mgr.stats.n_bytes_file > books/3 && mgr.stats.n_bytes_alt > books/3);
                    t.assert_equal("level 2: 12 read jobs", (int64_t) 12, mgr.stats.n_slabs_read);
                } else {
                    t.assert_equal("6 read jobs", (int64_t) 6, mgr.stats.n_slabs_read);
                }
            }
            remove(pa.c_str());
            remove(pb.c_str());
        });
    }

    return t.summary();
}

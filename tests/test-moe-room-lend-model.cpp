// The lent belt (src/llama-moe-room-lend-ops.cpp) on the 64-book fixtures: while writing, the reading
// room's belt keeps copies of the books the writing remap puts back, and a later miss for one of them
// copies it back instead of reading it. Its promise is that nothing but where a miss's bytes come from
// changes, so every scenario compares byte for byte (logits, and every ffn_moe_out where the eval
// callback runs) against the same model with lending off (LLAMA_MOE_ROOM_LEND=0) and with no streaming
// (the harness: test-moe-room-model.h), and each asserts the belt was really lent, or it would pass by
// never lending. The desk and the drive's slab count are compared with the lookahead prefetch off
// (LLAMA_MOE_STREAM_LOOKAHEAD=0): with it on, a wrong guess still loading is skipped as a victim, so the
// desk's choices follow the runners' timing and two runs with lending off end with different desks about
// half the time; the bytes do not depend on it, and "writing" and "desks" also run with it on.
//   writing   300 tokens read in through the room, then 48 written with a 40-slot desk: books kept and
//             copied back; byte-identical to lending off and to no streaming; the same desk as lending
//             off; and the drive read exactly one book's slabs fewer per book copied back. The lines it
//             writes, and GPU_SLOT=3 (on the GPU) loading with lending off
//   read-in   the same, then a second read-in and 48 more written, with LLAMA_MOE_ROOM_POISON=1 (copies
//             the room takes back are filled with 0xFF; every slab copied back is checked against a read
//             of the file) and no eval callback, which would make every split synchronous
//   desks     prompt B after A and after C (each written on, so lent), on two fresh models: B's bytes the
//             same, and the same as no streaming
//   paths     direct I/O (a staging read, then the upload the gate holds) with two runners on a twin copy,
//             against the zero-copy path of "writing" (the read lands in the slot, the gate holds the read)

#include "test-moe-room-model.h"

namespace {

std::string g_lend_log; // what loading said about the lent belt, the window lines and the run's total

const char * g_lend_env = "LLAMA_MOE_ROOM_LEND";

// the harness's own log capture (setup), plus the lent belt's lines
void capture_logs() {
    llama_log_set([](ggml_log_level, const char * text, void *) {
        if (strstr(text, "moe stream: reading room = ") != nullptr) {
            g_summary = text;
        }
        if (strstr(text, "reading room") != nullptr || strstr(text, "moe-stream-room") != nullptr) {
            g_room_log += text;
        }
        if (strstr(text, "lent belt") != nullptr) {
            g_lend_log += text;
        }
    }, nullptr);
}

struct lend_run {
    outputs out;
    std::vector<int32_t> desk;
    llama_moe_lend_stats lent;
    int64_t slabs  = 0;  // slabs the runners read (desk and belt)
    size_t  n_w    = 0;  // weights a book has on these floors
    bool    ok     = false;
};

const char * g_lookahead_env = "LLAMA_MOE_STREAM_LOOKAHEAD";

// segs on a fresh model of config c, lending on or off
lend_run run_lend(const config & c, bool lend, const std::vector<segment> & segs, bool capture,
        const std::function<void(llama_model *)> & also = {}) {
    if (lend) {
        unsetenv(g_lend_env);
    } else {
        setenv(g_lend_env, "0", 1);
    }
    lend_run r;
    llama_model * model = load(c);
    r.ok = model != nullptr && room_of(model) != nullptr && run(model, 128, segs, r.out, capture);
    if (r.ok) {
        llama_moe_stream * ms = model->moe_stream();
        r.desk  = desk_of(model);
        r.lent  = room_of(model)->lstats;
        r.slabs = ms->stats.n_slabs_read;
        for (const auto & sl : ms->layers) {
            r.n_w = r.n_w == 0 && sl ? sl->weights.size() : r.n_w;
        }
        check(room_of(model)->lend_on == lend, std::string("lending is ") + (lend ? "on" : "off") + " as asked");
        if (also) {
            also(model);
        }
    }
    llama_model_free(model);
    unsetenv(g_lend_env);
    return r;
}

std::string counts(const lend_run & r) {
    return std::to_string(r.lent.n_kept) + " kept, " + std::to_string(r.lent.n_lent) + " of " +
           std::to_string(r.lent.n_miss) + " misses copied back";
}

} // namespace

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    capture_logs();
    setenv("LLAMA_MOE_STREAM_STATS_MS", "1", 1); // a stats window every millisecond, so the lent line is written
    setenv(g_lookahead_env, "0", 1);              // a desk that does not follow the runners' timing (see the top)
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(ref_model));
    std::mt19937 rng(4321);
    const segment A = { random_tokens(rng, 300, n_vocab), 32, false };
    const segment B = { random_tokens(rng, 300, n_vocab), 48, true };
    const segment C = { random_tokens(rng, 300, n_vocab), 32, false };
    const segment W = { random_tokens(rng, 300, n_vocab), 48, true };

    outputs ref_b, ref_bw;
    bool ok = run(ref_model, 128, { B }, ref_b, true) && run(ref_model, 128, { B, W }, ref_bw, false);
    llama_model_free(ref_model);
    if (!ok) {
        fprintf(stderr, "the reference run failed\n");
        return 1;
    }
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);

    const config room4; // 55 slots with auto: a desk of 40 of 64 books, the belt beside it
    {
        g_lend_log.clear();
        const lend_run on  = run_lend(room4, true, { B }, true, [](llama_model * model) {
            g_summary.clear();
            llama_moe_stream_print_stats(model);
        });
        const lend_run off = run_lend(room4, false, { B }, true);
        check(on.ok && off.ok, "writing: ran with lending on and off");
        check(on.lent.n_kept > 0 && on.lent.n_lent > 0, "writing: books kept and copied back (" + counts(on) + ")");
        check(off.lent.n_kept == 0 && off.lent.n_miss == 0, "writing: lending off keeps nothing");
        check(same(ref_b.logits, on.out.logits) && same(ref_b.moe, on.out.moe) && nonzero(ref_b.moe),
                "writing: logits and every ffn_moe_out byte-identical to no streaming");
        check(same(off.out.logits, on.out.logits) && same(off.out.moe, on.out.moe), "writing: and to lending off");
        check(!on.desk.empty() && on.desk == off.desk, "writing: the desk the same as with lending off");
        check(off.slabs - on.slabs == (int64_t) on.n_w*on.lent.n_lent,
                "writing: the drive read " + std::to_string(off.slabs - on.slabs) + " slabs fewer = " +
                std::to_string(on.n_w) + " x " + std::to_string(on.lent.n_lent) + " books copied back");
        check(on.lent.n_bytes_out > 0 && on.lent.n_bytes_in > 0 && on.lent.n_taken_back == 0,
                "writing: bytes copied both ways, the belt never taken back");

        const size_t at_load  = g_lend_log.find("lent belt: while writing, the belt keeps up to ~");
        const size_t at_line  = g_lend_log.find("moe stream: lent belt: ");
        const size_t at_total = g_lend_log.find("moe stream: lent belt = ");
        const size_t at_off   = g_lend_log.find("lent belt: off; LLAMA_MOE_ROOM_LEND=0");
        check(at_load != std::string::npos && at_line != std::string::npos && at_total != std::string::npos &&
              at_off != std::string::npos, "writing: the load line, the window line, the total and the off line");
        check(g_lend_log.find("reading room:") == std::string::npos &&
              g_lend_log.find("moe stream: room") == std::string::npos && g_lend_log.find("moe stream: drives") == std::string::npos,
              "writing: none of them reads as the room's lines to the study's parser");
        // printed: the load line, the window that copied the most books back, and the run's total
        size_t at_busiest = at_line;
        int    most       = -1;
        for (size_t at = at_line; at != std::string::npos; at = g_lend_log.find("moe stream: lent belt: ", at + 1)) {
            int x = 0;
            const size_t bar = g_lend_log.find("| ", at);
            if (bar != std::string::npos && sscanf(g_lend_log.c_str() + bar + 2, "%d of", &x) == 1 && x > most) {
                most       = x;
                at_busiest = at;
            }
        }
        for (const size_t at : { at_load, at_busiest, at_total }) {
            if (at != std::string::npos) {
                printf("  | %s", g_lend_log.substr(at, g_lend_log.find('\n', at) - at + 1).c_str());
            }
        }

        // GPU_SLOT=3 moves the desk's bookkeeping to the GPU, where the remap that lends never runs: off, saying why
        if (g_ngl > 0 && llama_supports_gpu_offload()) {
            setenv("LLAMA_MOE_STREAM_GPU_SLOT", "3", 1);
            g_lend_log.clear();
            llama_model * m = load(room4);
            check(m != nullptr && room_of(m) != nullptr && !room_of(m)->lend_on &&
                  g_lend_log.find("lent belt: off; LLAMA_MOE_STREAM_GPU_SLOT=3") != std::string::npos,
                  "writing, GPU_SLOT=3: the room without lending, and the load line says why");
            llama_model_free(m);
            unsetenv("LLAMA_MOE_STREAM_GPU_SLOT");
        }

        // the study's default: the lookahead prefetch on (it neither keeps nor restores, v1)
        unsetenv(g_lookahead_env);
        const lend_run la = run_lend(room4, true, { B }, true);
        setenv(g_lookahead_env, "0", 1);
        check(la.ok && la.lent.n_lent > 0 && same(ref_b.logits, la.out.logits) && same(ref_b.moe, la.out.moe),
                "writing, lookahead on: lent (" + counts(la) + "), byte-identical to no streaming");
    }
    {
        // a read-in after lending: the room takes the belt back and reads over the copies; POISON fills them
        // with 0xFF as it does, so a copy found after that gives wrong words, and checks each slab copied
        // back against the file; no eval callback, which would make every split synchronous
        setenv("LLAMA_MOE_ROOM_POISON", "1", 1);
        const lend_run on  = run_lend(room4, true, { B, W }, false);
        const lend_run off = run_lend(room4, false, { B, W }, false);
        unsetenv("LLAMA_MOE_ROOM_POISON");
        check(on.ok && off.ok, "read-in after lending, poison: ran");
        check(on.lent.n_taken_back == 1 && on.lent.n_cleared > 0 && on.lent.n_lent > 0,
                "read-in after lending, poison: taken back once, " + std::to_string(on.lent.n_cleared) + " copies dropped, " + counts(on));
        check(same(ref_bw.logits, on.out.logits), "read-in after lending, poison: logits byte-identical to no streaming");
        check(same(off.out.logits, on.out.logits) && on.desk == off.desk, "read-in after lending, poison: and to lending off, the same desk");
        check(off.slabs - on.slabs == (int64_t) on.n_w*on.lent.n_lent, "read-in after lending, poison: slabs fewer by the books copied back");
    }
    {
        // B after two different histories, each written on so the belt was lent: different desks and belts,
        // the same bytes for B (and B's own writing lends too); the lookahead on, as the study runs
        unsetenv(g_lookahead_env);
        const lend_run a = run_lend(room4, true, { A, B }, true);
        const lend_run c = run_lend(room4, true, { C, B }, true);
        setenv(g_lookahead_env, "0", 1);
        size_t n_diff = 0;
        for (size_t i = 0; i < a.desk.size() && i < c.desk.size(); i++) {
            n_diff += a.desk[i] != c.desk[i];
        }
        check(a.ok && c.ok && n_diff > 0, "desks: after A and after C the desks differ (" + std::to_string(n_diff) + " slots)");
        check(a.lent.n_lent > 0 && c.lent.n_lent > 0 && a.lent.n_taken_back == 1 && c.lent.n_taken_back == 1,
                "desks: both lent, both taken back for B (" + counts(a) + "; " + counts(c) + ")");
        check(same(a.out.logits, c.out.logits) && same(a.out.moe, c.out.moe), "desks: B after A == B after C, byte for byte");
        check(same(ref_b.logits, a.out.logits) && same(ref_b.moe, a.out.moe), "desks: and == no streaming");
    }
    {
        // the study's path: direct I/O, where a book's new slab lands in staging and the gate holds its upload,
        // with two runners on a byte-identical copy (the zero-copy path, where the gate holds the read into the
        // slot itself, is "writing" above)
        config prod = room4;
        prod.direct = true;
        prod.alt    = g_path + ".lend-twin-ngl" + std::to_string(g_ngl) + ".gguf";
        std::filesystem::copy_file(g_path, prod.alt, std::filesystem::copy_options::overwrite_existing);
        const lend_run on = run_lend(prod, true, { B }, true, [](llama_model * model) {
            const llama_moe_stream * ms = model->moe_stream();
            check(ms->use_direct_io && !ms->files_alt.empty() && ms->stats.n_bytes_file > 0 && ms->stats.n_bytes_alt > 0,
                    "direct I/O, two runners: direct reads, and both runners read");
        });
        const lend_run off = run_lend(prod, false, { B }, true);
        std::filesystem::remove(prod.alt);
        check(on.ok && off.ok && on.lent.n_kept > 0 && on.lent.n_lent > 0, "direct I/O, two runners: lent (" + counts(on) + ")");
        check(same(ref_b.logits, on.out.logits) && same(ref_b.moe, on.out.moe), "direct I/O, two runners: byte-identical to no streaming");
        check(same(off.out.logits, on.out.logits) && on.desk == off.desk, "direct I/O, two runners: and to lending off, the same desk");
        check(off.slabs - on.slabs == (int64_t) on.n_w*on.lent.n_lent, "direct I/O, two runners: slabs fewer by the books copied back");
    }
    return finish();
}

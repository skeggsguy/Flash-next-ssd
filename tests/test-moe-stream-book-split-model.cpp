// One missing book across both runners (LLAMA_MOE_STREAM_SPLIT, src/llama-moe-stream-split.h) on the 64-book
// fixtures, with two runners on a byte-identical twin copy. The split moves which runner reads which bytes of a
// demand trip, never the bytes, so every scenario compares byte for byte (logits, and every ffn_moe_out) against
// no streaming and against level 0, and checks the split really happened, or it would pass by never splitting.
// The lookahead prefetch is off (LLAMA_MOE_STREAM_LOOKAHEAD=0) where the desk and the read counts are compared:
// with it on, the desk follows the runners' timing (test-moe-room-lend-model.cpp says why).
//   levels     0, 1 and 2 on the study's path (direct reads through staging) and on the zero-copy path (the read
//              lands in the slot): 300 tokens read in through the room, 48 written on a 40-slot desk with the
//              lent belt on. The same desk at every level; the same bytes off the drives; level 2 reads one more
//              job per halved slab; level 0 splits nothing; both runners read; every slot settled
//   room       a read-in alone at level 2: the room's reads (its belt and its cold desk fills) stay whole
//   lent belt  at level 2 books are kept and copied back (whole slabs: a restore is a copy, no runner), and the
//              keep's gate holds each half's write until the old slab is copied out
//   lookahead  on, as the study runs: halves only for the writing remap's trips, never for a guess
//   one runner level 2 asked for, nothing split, the load line saying why
//   GPU slot   GPU_SLOT=3 (-ngl > 0): the slot servicer's trips halved, byte for byte
//   waves      the room off: a read-in's waves queue demand trips too; level 2 against level 0, byte for byte

#include "test-moe-room-model.h"

namespace {

std::string g_split_log; // what loading said about the split (open_files' "book split level" line)

struct split_run {
    outputs out;
    std::vector<int32_t> desk;
    llama_moe_lend_stats lent;
    llama_moe_stream_split split;
    int64_t slabs = 0, bytes_file = 0, bytes_alt = 0, misses = 0, room_books = 0;
    size_t  n_w = 0;
    bool    settled = false; // no slot LOADING, nothing pending
    bool    ok = false;
    std::string log; // the load line naming the level
};

split_run run_split(const config & c, int level, const std::vector<segment> & segs, bool capture) {
    setenv("LLAMA_MOE_STREAM_SPLIT", std::to_string(level).c_str(), 1);
    split_run r;
    g_split_log.clear();
    llama_model * model = load(c);
    r.log = g_split_log;
    r.ok = model != nullptr && run(model, 128, segs, r.out, capture);
    if (r.ok) {
        llama_moe_stream * ms = model->moe_stream();
        std::lock_guard<std::mutex> lk(ms->mtx);
        r.desk       = desk_of(model);
        r.split      = ms->split;
        r.slabs      = ms->stats.n_slabs_read;
        r.bytes_file = ms->stats.n_bytes_file;
        r.bytes_alt  = ms->stats.n_bytes_alt;
        r.misses     = ms->stats.n_miss;
        r.lent       = ms->room ? ms->room->lstats : llama_moe_lend_stats();
        r.room_books = ms->room ? ms->room->stats.n_books : 0;
        r.settled    = ms->q_demand.empty() && ms->q_spec.empty();
        for (const auto & sl : ms->layers) {
            r.n_w = r.n_w == 0 && sl ? sl->weights.size() : r.n_w;
            for (uint32_t s = 0; sl && s < sl->n_slots; s++) {
                r.settled = r.settled && sl->slot_pending[s] == 0 && sl->slot_state[s] != LLAMA_MOE_STREAM_SLOT_LOADING;
            }
        }
        r.ok = r.ok && ms->split.level == level && ms->files_alt.empty() == c.alt.empty();
    }
    llama_model_free(model);
    unsetenv("LLAMA_MOE_STREAM_SPLIT");
    return r;
}

std::string counts(const split_run & r) {
    return std::to_string(r.split.n_halved) + " slabs halved, " + std::to_string(r.split.n_spread) + " spread, " +
           std::to_string(r.slabs) + " read jobs, " + std::to_string(r.lent.n_kept) + " kept, " +
           std::to_string(r.lent.n_lent) + " copied back";
}

} // namespace

int main(int argc, char ** argv) {
    llama_model * ref_model = setup(argc, argv);
    if (ref_model == nullptr) {
        return 1;
    }
    setenv("LLAMA_MOE_STREAM_LOOKAHEAD", "0", 1);
    llama_log_set([](ggml_log_level, const char * text, void *) { // silent, but the split's load line is kept
        if (strstr(text, "book split level") != nullptr) {
            g_split_log += text;
        }
    }, nullptr);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(ref_model));
    std::mt19937 rng(2611);
    const segment B = { random_tokens(rng, 300, n_vocab), 48, true };
    const segment R = { random_tokens(rng, 300, n_vocab), 0, true }; // a read-in, nothing written

    outputs ref_b;
    if (!run(ref_model, 128, { B }, ref_b, true)) {
        fprintf(stderr, "the reference run failed\n");
        return 1;
    }
    llama_model_free(ref_model);
    printf("%s, -ngl %d\n", g_path.c_str(), g_ngl);

    config direct; // 55 slots with auto: a desk of 40 of 64 books, the room and the lent belt beside it
    direct.direct = true;
    direct.alt    = g_path + ".book-split-twin-ngl" + std::to_string(g_ngl) + ".gguf";
    std::filesystem::copy_file(g_path, direct.alt, std::filesystem::copy_options::overwrite_existing);
    config zero = direct;
    zero.direct = false;

    for (const config * c : { &direct, &zero }) {
        const std::string path = c->direct ? "direct" : "zero-copy";
        const split_run l0 = run_split(*c, 0, { B }, true);
        check(l0.ok && same(ref_b.logits, l0.out.logits) && same(ref_b.moe, l0.out.moe) && nonzero(ref_b.moe),
                path + ", level 0: byte-identical to no streaming");
        check(l0.split.n_halved == 0 && l0.split.n_spread == 0 && l0.split.rr == 0, path + ", level 0: nothing split");
        int64_t spread = -1; // level 1's slabs spread: the trips' slabs, which level 2 must halve, every one
        for (const int level : { 1, 2 }) {
            const std::string name = path + ", level " + std::to_string(level);
            const split_run r = run_split(*c, level, { B }, true);
            check(r.ok && same(ref_b.logits, r.out.logits) && same(ref_b.moe, r.out.moe),
                    name + ": logits and every ffn_moe_out byte-identical to no streaming (" + counts(r) + ")");
            check(same(l0.out.logits, r.out.logits) && same(l0.out.moe, r.out.moe) && !r.desk.empty() && r.desk == l0.desk,
                    name + ": and to level 0, the same desk");
            check(r.bytes_file + r.bytes_alt == l0.bytes_file + l0.bytes_alt && r.bytes_file > 0 && r.bytes_alt > 0,
                    name + ": the same bytes off the drives as level 0, both runners reading");
            check(r.slabs == l0.slabs + r.split.n_halved, name + ": " + std::to_string(r.slabs) + " read jobs = level 0's " +
                    std::to_string(l0.slabs) + " + one per halved slab");
            check(level == 1 ? r.split.n_spread > 0 && r.split.n_halved == 0 : r.split.n_halved > 0 && r.split.n_spread == 0,
                    name + (level == 1 ? ": slabs spread, none halved" : ": slabs halved"));
            check(r.settled, name + ": every slot settled, nothing pending");
            const std::string want = "MoE expert streaming book split level " + std::to_string(level) + ": ";
            check(r.log.find(want) != std::string::npos && r.log.find("one runner") == std::string::npos &&
                    r.log.find("room") == std::string::npos && r.log.find("drives") == std::string::npos,
                    name + ": the load line names the level: " + r.log.substr(0, r.log.find('\n')));
            if (level == 2) {
                // the lent belt: books kept as the desk put them back (each half's write held by the keep's gate),
                // and trips copied back from the belt instead (whole slabs, no runner: the restore asserts it)
                check(r.lent.n_kept > 0 && r.lent.n_lent > 0 && r.lent.n_lent == l0.lent.n_lent,
                        name + ": the lent belt kept and copied back as at level 0 (" + counts(r) + ")");
                // the room's misses (its cold desk fills) are read whole off q_room, so the count of trips' slabs
                // is level 1's spread: the same desk, so the same trips through the same queue
                check(r.split.n_halved == spread && r.split.n_halved % (int64_t) r.n_w == 0,
                        name + ": every slab of every trip halved, no other (" + std::to_string(r.split.n_halved) +
                        " halved, level 1 spread " + std::to_string(spread) + ", whole books of " +
                        std::to_string(r.n_w) + " slabs)");
            } else {
                spread = r.split.n_spread;
            }
        }
    }
    {
        // one runner: nothing to split across, and the load line says so
        config one = direct;
        one.alt.clear();
        const split_run r = run_split(one, 2, { B }, true);
        check(r.ok && same(ref_b.logits, r.out.logits) && r.split.n_halved == 0 && r.split.n_spread == 0 &&
                r.bytes_alt == 0 && r.settled && r.log.find("book split level 0: ") != std::string::npos &&
                r.log.find("one runner, nothing to split across (LLAMA_MOE_STREAM_SPLIT=2)") != std::string::npos,
                "one runner, level 2: nothing split, byte-identical, and the load line says why");
    }
    {
        // a read-in alone: the room's reads stay whole jobs
        const split_run r = run_split(direct, 2, { R }, true);
        check(r.ok && r.room_books > 0 && r.slabs > 0 && r.split.n_halved == 0 && r.split.n_spread == 0 && r.settled,
                "room, level 2: " + std::to_string(r.room_books) + " books on the belt, " + std::to_string(r.slabs) +
                " read jobs, none split");
    }
    {
        // the lookahead on: its guesses keep whole jobs, so the halves are the trips' slabs and no more
        unsetenv("LLAMA_MOE_STREAM_LOOKAHEAD");
        const split_run r = run_split(direct, 2, { B }, true);
        setenv("LLAMA_MOE_STREAM_LOOKAHEAD", "0", 1);
        check(r.ok && same(ref_b.logits, r.out.logits) && same(ref_b.moe, r.out.moe) && r.settled,
                "lookahead on, level 2: byte-identical to no streaming (" + counts(r) + ")");
        check(r.split.n_halved > 0 && r.split.n_halved % (int64_t) r.n_w == 0 &&
                r.split.n_halved <= (int64_t) r.n_w*(r.misses - r.lent.n_lent),
                "lookahead on, level 2: halves only for the trips' slabs, whole books of them"); // misses bound trips
    }
    if (g_ngl > 0 && llama_supports_gpu_offload()) {
        // GPU_SLOT=3: the GPU picks the desk and the slot servicer queues the trips (the fourth demand site)
        setenv("LLAMA_MOE_STREAM_GPU_SLOT", "3", 1);
        const split_run g0 = run_split(direct, 0, { B }, true);
        const split_run g2 = run_split(direct, 2, { B }, true);
        unsetenv("LLAMA_MOE_STREAM_GPU_SLOT");
        check(g0.ok && g2.ok && same(ref_b.logits, g2.out.logits) && same(ref_b.moe, g2.out.moe) && g2.settled,
                "GPU_SLOT=3, level 2: byte-identical to no streaming, every slot settled (" + counts(g2) + ")");
        check(g2.split.n_halved > 0 && g2.slabs == g0.slabs + g2.split.n_halved &&
                g2.bytes_file + g2.bytes_alt == g0.bytes_file + g0.bytes_alt,
                "GPU_SLOT=3, level 2: halved, one more job per halved slab, the same bytes as level 0");
    }
    {
        // waves: the room off, so the read-in's books are demand trips on the desk
        config waves = direct;
        waves.room_mode = LLAMA_MOE_ROOM_OFF;
        waves.slots     = 40;
        const split_run w0 = run_split(waves, 0, { B }, true);
        const split_run w2 = run_split(waves, 2, { B }, true);
        check(w0.ok && w2.ok && same(w0.out.logits, w2.out.logits) && same(w0.out.moe, w2.out.moe) && nonzero(w0.out.moe),
                "waves, level 2: byte-identical to level 0 (" + counts(w2) + ")");
        check(w2.split.n_halved > 0 && w2.slabs == w0.slabs + w2.split.n_halved && w2.desk == w0.desk && w2.settled,
                "waves, level 2: halved, one more job per halved slab, the same desk");
    }
    std::filesystem::remove(direct.alt);
    return finish();
}

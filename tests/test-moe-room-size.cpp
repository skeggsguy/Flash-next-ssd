// The reading room's size (src/llama-moe-room-size.cpp): how --moe-stream-room and --moe-stream-room-parts
// carve a belt out of the desk's budget, every refusal and warning, and the startup line word for word.
//
// A wrong size does not crash anything: a room that silently took too much leaves writing a smaller desk,
// and one sized from the wrong floor leaves a part that cannot fit. So the arithmetic is pinned here with
// numbers worked out by hand, and the auto size is checked against its own definition (a fixed point).

#include "testing.h"

#include "../src/llama-moe-room-size.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr uint64_t MiB = 1024ull*1024ull;

// 256 books a floor, 8 read a word, 2 floors, each book 2 MiB on a floor (1 MiB gate_up + 1 MiB down):
// a slot on every floor costs 4 MiB, so a 400 MiB budget is a desk of 100 slots with the room off
llama_moe_room_books small_books() {
    llama_moe_room_books b;
    b.budget           = 400*MiB;
    b.desk_slots       = 100;
    b.book_bytes       = 4*MiB;
    b.stride_min       = 2*MiB;
    b.stride_max       = 2*MiB;
    b.n_expert         = 256;
    b.n_expert_used    = 8;
    b.sweep_min_tokens = llama_moe_room_sweep_min_tokens(256, 8);
    return b;
}

// the paperback's shape: 512 books a floor, 10 read, 48 floors, ~3.06 MiB a book per floor, desk 32 GiB
llama_moe_room_books library_books() {
    llama_moe_room_books b;
    b.n_expert         = 512;
    b.n_expert_used    = 10;
    b.stride_min       = b.stride_max = 3211264;
    b.book_bytes       = 48ull*3211264;
    b.budget           = 32ull*1024*MiB;
    b.desk_slots       = (uint32_t) (b.budget/b.book_bytes);
    b.sweep_min_tokens = llama_moe_room_sweep_min_tokens(512, 10);
    return b;
}

llama_moe_room_request request(int32_t mode, double value, int32_t parts = 4) {
    llama_moe_room_request r;
    r.mode  = mode;
    r.value = value;
    r.parts = parts;
    return r;
}

bool contains(const std::string & s, const std::string & what) {
    return s.find(what) != std::string::npos;
}

} // namespace

int main(int argc, char ** argv) {
    testing t;
    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    t.verbose = verbose && std::string(verbose) == "1";
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("sweep threshold", [](testing & t) {
        // ceil(20 slips per book x n / k): RR-room's 1,024 tokens for the library, 160 for the 64-book fixtures
        t.assert_equal("10 of 512", 1024u, llama_moe_room_sweep_min_tokens(512, 10));
        t.assert_equal("8 of 64", 160u, llama_moe_room_sweep_min_tokens(64, 8));
        t.assert_equal("8 of 256", 640u, llama_moe_room_sweep_min_tokens(256, 8));
        t.assert_equal("7 of 64 rounds up", 183u, llama_moe_room_sweep_min_tokens(64, 7)); // 182.9 slips' worth
        t.assert_equal("every book read", 1u, llama_moe_room_sweep_min_tokens(8, 8));
        t.assert_equal("no slips", 1u, llama_moe_room_sweep_min_tokens(8, 0));
        // it scales with the model: twice the books, twice the tokens for the same slips per book
        t.assert_equal("scales with n", 2*llama_moe_room_sweep_min_tokens(512, 10), llama_moe_room_sweep_min_tokens(1024, 10));
        // the most books llama loads (LLAMA_MAX_EXPERTS) reading one a word is the largest reachable value
        t.assert_equal("1 of 1024", 20480u, llama_moe_room_sweep_min_tokens(1024, 1));
        // a shape past uint32 saturates (never takes the room) rather than wrapping to a small number or 0
        t.assert_equal("past uint32 never takes the room", UINT32_MAX, llama_moe_room_sweep_min_tokens(1u << 31, 1));
    });

    t.test("sweep threshold from the environment", [](testing & t) {
        // LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS wins whatever the model's shape; unset (null) is the default
        t.assert_equal("unset", 1024u, llama_moe_room_sweep_min_tokens_env(nullptr, 512, 10));
        t.assert_equal("set", 234u, llama_moe_room_sweep_min_tokens_env("234", 512, 10));
        t.assert_equal("set, fixtures", 35u, llama_moe_room_sweep_min_tokens_env("35", 64, 8));
        t.assert_equal("above the default", 4096u, llama_moe_room_sweep_min_tokens_env("4096", 64, 8));
        // at least 1: a read-in of 0 tokens is no read-in, and garbage reads as 0
        t.assert_equal("0", 1u, llama_moe_room_sweep_min_tokens_env("0", 512, 10));
        t.assert_equal("negative", 1u, llama_moe_room_sweep_min_tokens_env("-5", 512, 10));
        t.assert_equal("not a number", 1u, llama_moe_room_sweep_min_tokens_env("abc", 512, 10));
        t.assert_equal("empty", 1u, llama_moe_room_sweep_min_tokens_env("", 512, 10));
        // read the way the manager always read it (atoi): leading blanks skipped, trailing junk ignored
        t.assert_equal("as atoi reads it", 42u, llama_moe_room_sweep_min_tokens_env(" 42x", 512, 10));
    });

    t.test("record stride", [](testing & t) {
        const size_t nb[] = { 1000, 256, 257 }; // each weight starts 256-aligned in the record
        t.assert_equal("padded sum", (size_t) (1024 + 256 + 512), llama_moe_room_record_stride(nb, 3));
    });

    t.test("room 0 keeps today's desk", [](testing & t) {
        const llama_moe_room_layout lay = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_OFF, 0.0), small_books());
        t.assert_true("off", !lay.on);
        t.assert_true("no error", lay.error.empty());
        t.assert_equal("desk slots", 100u, lay.desk_slots);
        t.assert_equal("no room", (uint64_t) 0, lay.room_bytes);
        t.assert_equal("off line", std::string("reading room: off; reading in uses waves on the desk (100 slots per floor)"),
                llama_moe_room_describe(lay));
    });

    t.test("GiB", [](testing & t) {
        // 256 MiB of room leaves (400 - 256)/4 = 36 slots, so 220 books a floor are missing: 0.58 floors
        // (a warning), parts of ceil(220/4) = 55 books = 110 MiB, 128 records of the belt per floor
        const llama_moe_room_layout lay = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_GIB, 0.25), small_books());
        t.assert_true("on", lay.on);
        t.assert_true("no error: " + lay.error, lay.error.empty());
        t.assert_equal("room", 256*MiB, lay.room_bytes);
        t.assert_equal("desk slots", 36u, lay.desk_slots);
        t.assert_equal("desk bytes", 144*MiB, lay.desk_bytes);
        t.assert_equal("largest part", 110*MiB, lay.part_bytes_max);
        t.assert_equal("records", 128u, lay.n_records_max);
        t.assert_true("less than a floor warns", contains(lay.warning, "0.58 floors of look-ahead, less than one"));
        t.assert_equal("startup line", std::string("reading room: 0.25 GiB = 0.58 floors of look-ahead, each floor in "
                "4 parts (up to 110 MiB); desk 0.1 GiB (36 slots per floor); reading in of 640+ tokens uses the room, "
                "shorter uses waves"), llama_moe_room_describe(lay));
    });

    t.test("floors", [](testing & t) {
        // by hand: desk 100 -> room 156 MiB -> desk 61 -> 195 -> 51 -> 205 -> 48 -> 208 -> 48, the fixed point;
        // half of the 208 books a floor then leaves are 104 books = 208 MiB, so two parts fill the room exactly
        const llama_moe_room_layout lay = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_FLOORS, 0.5, 2), small_books());
        t.assert_true("on", lay.on);
        t.assert_equal("room", 208*MiB, lay.room_bytes);
        t.assert_equal("desk slots", 48u, lay.desk_slots);
        t.assert_true("exactly half a floor", std::fabs(lay.floors - 0.5) < 1e-12);
        t.assert_equal("a part fills the room", 208*MiB, lay.part_bytes_max);

        // one floor with a 1000 MiB budget: desk 250 -> 244 slots, 12 books (24 MiB) missing, all in one part
        llama_moe_room_books b = small_books();
        b.budget     = 1000*MiB;
        b.desk_slots = 250;
        const llama_moe_room_layout one = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_FLOORS, 1.0, 1), b);
        t.assert_true("one floor on", one.on && one.warning.empty());
        t.assert_equal("one floor's desk", 244u, one.desk_slots);
        t.assert_true("one part", contains(llama_moe_room_describe(one), "= 1.00 floors of look-ahead, each floor in 1 part (up to 24 MiB)"));
    });

    t.test("auto on the library", [](testing & t) {
        const llama_moe_room_books b = library_books();
        const llama_moe_room_layout lay = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_AUTO, 0.0), b);
        t.assert_true("on", lay.on && lay.error.empty() && lay.warning.empty());
        t.assert_true("1.25 floors", lay.floors >= 1.25 && lay.floors < 1.2501);
        // the fixed point's two equations hold together
        t.assert_equal("desk is what the room leaves", (uint32_t) ((b.budget - lay.room_bytes)/b.book_bytes), lay.desk_slots);
        const double want = std::ceil(1.25*(512 - lay.desk_slots)*(double) b.stride_max);
        t.assert_true("room is 1.25 of the floor that desk leaves", lay.room_bytes >= want && lay.room_bytes < want + 256);
        t.assert_true("desk and room within the budget", lay.desk_bytes + lay.room_bytes <= b.budget);
        t.assert_true("about 1.1 GiB", lay.room_bytes > 1100*MiB && lay.room_bytes < 1150*MiB);
        t.assert_true("1024+ tokens", contains(llama_moe_room_describe(lay), "reading in of 1024+ tokens uses the room"));
    });

    t.test("refusals", [](testing & t) {
        const llama_moe_room_books b = small_books();
        auto error = [&](const llama_moe_room_request & r) {
            const llama_moe_room_layout lay = llama_moe_room_resolve(r, b);
            return lay.on ? std::string() : lay.error;
        };
        t.assert_true("parts 0", contains(error(request(LLAMA_MOE_ROOM_AUTO, 0.0, 0)), "must be between 1 and 16 (got 0)"));
        t.assert_true("parts 17", contains(error(request(LLAMA_MOE_ROOM_AUTO, 0.0, 17)), "must be between 1 and 16 (got 17)"));
        t.assert_true("room as big as the budget", contains(error(request(LLAMA_MOE_ROOM_GIB, 0.5)), "takes the whole desk budget"));
        t.assert_true("floors bigger than the budget", contains(error(request(LLAMA_MOE_ROOM_FLOORS, 2.0)), "takes the whole desk budget"));
        // 0.35 GiB leaves (400 - 358.4)/4 = 10 slots, under 3 x 8
        t.assert_true("desk too small", contains(error(request(LLAMA_MOE_ROOM_GIB, 0.35)), "leaves the desk 10 slots per floor, fewer than the 24"));
        // one part of 220 books is 440 MiB, more than the 256 MiB room
        t.assert_true("part bigger than the room", contains(error(request(LLAMA_MOE_ROOM_GIB, 0.25, 1)), "up to 440 MiB, bigger than the whole reading room (256 MiB)"));
        t.assert_true("no size", contains(error(request(LLAMA_MOE_ROOM_GIB, 0.0)), "cannot make a reading room"));
        t.assert_true("unknown mode", contains(error(request(7, 1.0)), "cannot make a reading room"));
        llama_moe_room_books whole = b; // a budget that seats every book: nothing to sweep, so no room
        whole.desk_slots = whole.n_expert;
        t.assert_true("desk holds every book", contains(llama_moe_room_resolve(request(LLAMA_MOE_ROOM_AUTO, 0.0), whole).error,
                "the desk already holds every book (256 slots per floor for 256 books)"));
        llama_moe_room_books none = b;
        none.book_bytes = 0;
        t.assert_true("no streamed books", contains(llama_moe_room_resolve(request(LLAMA_MOE_ROOM_AUTO, 0.0), none).error,
                "no streamed books"));

        llama_moe_room_books tiny = b; // 64 KiB books on some floor: 4096 of them fit in 256 MiB
        tiny.stride_min = 64*1024;
        const llama_moe_room_layout lay = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_GIB, 0.25), tiny);
        t.assert_true("more than 1024 records", !lay.on && contains(lay.error, "holds 4096 books of one floor, but the GPU indexes at most 1024"));
    });

    t.test("the default is auto where the room can be made", [](testing & t) {
        // not asked for (the flag absent): the same room auto makes, and nothing to warn about
        const llama_moe_room_books b = library_books();
        const llama_moe_room_layout asked = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_AUTO, 0.0), b);
        const llama_moe_room_layout dflt  = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_DEFAULT, 0.0), b);
        t.assert_true("on", dflt.on && dflt.error.empty() && dflt.warning.empty());
        t.assert_equal("auto's room", asked.room_bytes, dflt.room_bytes);
        t.assert_equal("auto's desk", asked.desk_slots, dflt.desk_slots);
        t.assert_equal("auto's parts", asked.parts, dflt.parts);
        t.assert_equal("auto's startup line", llama_moe_room_describe(asked), llama_moe_room_describe(dflt));
        // parts still count when the room is the default one
        t.assert_equal("parts 2", 2, llama_moe_room_resolve(request(LLAMA_MOE_ROOM_DEFAULT, 0.0, 2), b).parts);
    });

    t.test("the default room steps back to off where asked-for refuses", [](testing & t) {
        // Each thing that stops an asked-for room: the default one leaves today's desk, room off, and says
        // why in a warning, never an error (an error stops the load)
        const llama_moe_room_books lib = library_books();
        llama_moe_room_books arch = lib, whole = lib, none = lib, tiny = lib, ub512 = lib;
        arch.refusal     = "--moe-stream-room: llama4 weights each book's input before the expert maths";
        whole.desk_slots = whole.n_expert;
        none.book_bytes  = 0;
        tiny.stride_min  = 64*1024; // 1.25 floors of the largest books hold far more than 1024 of the smallest
        ub512.ubatch     = 512;     // llama.cpp's own default -ub, under the library's 1,024-token threshold
        struct refused { const char * what; llama_moe_room_books books; int32_t parts; const char * reason; };
        for (const refused & r : std::vector<refused>{
                { "the arch or the devices",  arch,          4,  "llama4 weights each book's input" },
                { "a budget too small",       small_books(), 4,  "takes the whole desk budget" },
                { "a desk that seats every book", whole,     4,  "the desk already holds every book" },
                { "no streamed books",        none,          4,  "no streamed books" },
                { "more than 1024 records",   tiny,          4,  "but the GPU indexes at most 1024" },
                { "a batch under the threshold", ub512,      4,  "batches of 512 tokens" },
                { "parts 0 through the API",  lib,           0,  "must be between 1 and 16 (got 0)" } }) {
            const llama_moe_room_layout asked = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_AUTO, 0.0, r.parts), r.books);
            const llama_moe_room_layout dflt  = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_DEFAULT, 0.0, r.parts), r.books);
            const std::string w = r.what;
            t.assert_true(w + ": asked for, it refuses: " + asked.error, !asked.on && contains(asked.error, r.reason));
            t.assert_true(w + ": the default loads, room off", !dflt.on && dflt.error.empty());
            t.assert_equal(w + ": today's desk", r.books.desk_slots, dflt.desk_slots);
            t.assert_equal(w + ": no room", (uint64_t) 0, dflt.room_bytes);
            t.assert_true(w + ": the off startup line", contains(llama_moe_room_describe(dflt), "reading room: off;"));
            t.assert_true(w + ": the warning says so", contains(dflt.warning, "on by default but cannot be made here, so it is off"));
            t.assert_true(w + ": and gives asked-for's reason", contains(dflt.warning, asked.error));
        }
        // room 0 asked for on a model the room refuses is simply off, with nothing to say
        const llama_moe_room_layout off = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_OFF, 0.0), arch);
        t.assert_true("room 0 on a refusing model", !off.on && off.error.empty() && off.warning.empty());
        // any size asked for refuses on the arch alone
        t.assert_true("GiB asked for refuses", contains(llama_moe_room_resolve(request(LLAMA_MOE_ROOM_GIB, 1.0), arch).error, "llama4"));
        t.assert_true("floors asked for refuse", contains(llama_moe_room_resolve(request(LLAMA_MOE_ROOM_FLOORS, 1.0), arch).error, "llama4"));
    });

    t.test("a batch that can never reach the threshold makes no room", [](testing & t) {
        // The room takes a reading-in batch of sweep_min_tokens and up (1,024 on the library), and a read-in
        // is cut into batches of -ub first. llama.cpp's own default is -ub 512, so a default user would have
        // ~1.1 GiB carved out of the desk for a room that never runs: the default room stays off and says
        // so, a room asked for refuses (the same class as a desk that seats every book: the room would only
        // shrink the desk). 0 is "not known" (an API caller that did not say) and leaves the old rule: the
        // room is made and the context warns when its batch turns out too small.
        llama_moe_room_books b = library_books();
        for (const uint32_t ub : { 0u, 1024u, 4096u }) {
            b.ubatch = ub;
            const std::string w = "-ub " + std::to_string(ub);
            t.assert_true(w + ": the default room is on", llama_moe_room_resolve(request(LLAMA_MOE_ROOM_DEFAULT, 0.0), b).on);
            t.assert_true(w + ": auto is on", llama_moe_room_resolve(request(LLAMA_MOE_ROOM_AUTO, 0.0), b).on);
        }
        for (const uint32_t ub : { 1u, 512u, 1023u }) {
            b.ubatch = ub;
            const std::string w = "-ub " + std::to_string(ub);
            const llama_moe_room_layout dflt = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_DEFAULT, 0.0), b);
            t.assert_true(w + ": the default room is off, and loads", !dflt.on && dflt.error.empty());
            t.assert_equal(w + ": the whole desk", b.desk_slots, dflt.desk_slots);
            t.assert_true(w + ": the warning names the batch and the threshold",
                    contains(dflt.warning, "batches of " + std::to_string(ub) + " tokens (-ub), fewer than the 1024"));
            t.assert_true(w + ": the warning says what to do", contains(dflt.warning, "raise -ub to 1024 or more"));
            for (const int32_t mode : { LLAMA_MOE_ROOM_AUTO, LLAMA_MOE_ROOM_GIB, LLAMA_MOE_ROOM_FLOORS }) {
                const llama_moe_room_layout asked = llama_moe_room_resolve(request(mode, 1.0), b);
                t.assert_true(w + ": asked for (mode " + std::to_string(mode) + ") refuses",
                        !asked.on && contains(asked.error, "the room would never be used"));
            }
            // "reading room:" is the startup line's mark, and the runner takes the first line that carries
            // it as the room, so no warning may carry it
            t.assert_true(w + ": the warning is not a startup line", !contains(dflt.warning, "reading room:"));
        }
        // the rung's own threshold (LLAMA_MOE_ROOM_SWEEP_MIN_TOKENS) is what the batch is held against
        b.ubatch = 512;
        b.sweep_min_tokens = 234;
        t.assert_true("a lowered threshold under -ub 512 keeps the room", llama_moe_room_resolve(request(LLAMA_MOE_ROOM_DEFAULT, 0.0), b).on);
        b.sweep_min_tokens = 513;
        t.assert_true("a raised one above it does not", !llama_moe_room_resolve(request(LLAMA_MOE_ROOM_DEFAULT, 0.0), b).on);
        // room 0 has nothing to say about the batch
        b.sweep_min_tokens = 1024;
        const llama_moe_room_layout off = llama_moe_room_resolve(request(LLAMA_MOE_ROOM_OFF, 0.0), b);
        t.assert_true("room 0 at -ub 512", !off.on && off.error.empty() && off.warning.empty());
    });

    return t.summary();
}

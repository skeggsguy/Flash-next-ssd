// The put-back rule's load line (LLAMA_MOE_STREAM_LRU, pick_victim_locked), without a model: open_files names the
// rule a desk runs, once per desk, and the rule it names is the one the desk then uses.
//
//   - on (any nonzero value): "moe stream: put-back rule: lru, ..."; unset, empty, "0" or not a number: "... hotness"
//   - one line per desk, so a server's two desks (the library's and the apprentice's) print two
//   - the words after "moe stream:" never start with "room" or "drives" (the study's runner/roomstats.py takes its
//     window lines by those first words), and neither line holds the other's rule word (the study's chains grep -F)
//   - the line tells the truth: on a desk where the book idle longest is not the book used least, the slot the desk
//     puts back is the one the line's rule names

#include "testing.h"

#include "llama.h"

#include "../src/llama-moe-stream.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

static const char * const LRU_LINE =
    "open_files: moe stream: put-back rule: lru, the book idle longest goes back (LLAMA_MOE_STREAM_LRU on)\n";
static const char * const HOTNESS_LINE =
    "open_files: moe stream: put-back rule: hotness, the book used least lately goes back (LLAMA_MOE_STREAM_LRU off)\n";

static std::vector<std::string> g_lines; // every put-back rule line the engine logged

static std::string tmp_file() {
    const char * d = std::getenv("TMPDIR");
    const std::string path = (d && *d ? std::string(d) : std::string("/tmp")) + "/moe-put-back-" +
                             std::to_string(getpid()) + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    GGML_ASSERT(f != nullptr);
    const std::vector<uint8_t> bytes(8192, 0x5A);
    GGML_ASSERT(fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size());
    fclose(f);
    return path;
}

// one floor's desk of four books, all landed: slot 0 holds the book used least (hotness 1) and slot 1 the book idle
// longest (last use 1), so the two rules put back different slots
struct desk {
    std::unique_ptr<llama_moe_stream> mgr;
    llama_moe_stream_layer * sl = nullptr;

    desk() {
        mgr = std::make_unique<llama_moe_stream>(/*n_layer =*/ 1, /*n_slots =*/ 4, /*n_io_threads =*/ 1, /*direct =*/ false);
        mgr->layers[0] = std::make_unique<llama_moe_stream_layer>();
        sl = mgr->layers[0].get();
        sl->mgr      = mgr.get();
        sl->n_expert = 8;
        sl->n_slots  = 4;
        sl->slot_expert   = { 0, 1, 2, 3 };
        sl->slot_state.assign(4, LLAMA_MOE_STREAM_SLOT_RESIDENT);
        sl->slot_pending.assign(4, 0);
        sl->slot_gen.assign(4, 1);
        sl->slot_last_use = { 10, 1, 20, 30 };
        sl->route_hotness = { 1, 9, 5, 5, 0, 0, 0, 0 };
    }

    int32_t victim() {
        std::lock_guard<std::mutex> lk(mgr->mtx);
        return mgr->pick_victim_locked(*sl, nullptr);
    }
};

// open one desk's files with LLAMA_MOE_STREAM_LRU set to value (nullptr: unset); what it logged, and its victim
struct opened {
    std::vector<std::string> lines;
    int32_t victim = -1;
};

static opened open_with(const char * value, const std::string & path) {
    if (value) {
        setenv("LLAMA_MOE_STREAM_LRU", value, 1);
    } else {
        unsetenv("LLAMA_MOE_STREAM_LRU");
    }
    g_lines.clear();
    desk d;
    d.mgr->open_files({ path });
    unsetenv("LLAMA_MOE_STREAM_LRU");
    return { g_lines, d.victim() };
}

// the words after "moe stream: " must not start with a word the stats parser takes as a window line
static bool parser_safe(const std::string & line) {
    const std::string tag = "moe stream: ";
    const size_t at = line.find(tag);
    if (at == std::string::npos) {
        return false;
    }
    const std::string rest = line.substr(at + tag.size());
    return rest.rfind("room", 0) != 0 && rest.rfind("reading room", 0) != 0 && rest.rfind("drives", 0) != 0 &&
           rest.rfind("lent belt", 0) != 0 && rest.rfind("read ", 0) != 0;
}

int main(int argc, char ** argv) {
    testing t;
    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    t.verbose = verbose && std::string(verbose) == "1";
    llama_log_set([](ggml_log_level, const char * text, void * verbose) {
        if (strstr(text, "put-back rule") != nullptr) {
            g_lines.emplace_back(text);
        }
        if (*(bool *) verbose) {
            fputs(text, stderr);
        }
    }, &t.verbose);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    const std::string path = tmp_file();

    t.test("lru_when_on", [&](testing & t) {
        for (const char * v : { "1", "2" }) {
            const opened o = open_with(v, path);
            t.assert_equal(std::string("\"") + v + "\": one line", (size_t) 1, o.lines.size());
            if (!o.lines.empty()) {
                t.assert_equal(std::string("\"") + v + "\": the lru line", std::string(LRU_LINE), o.lines[0]);
            }
            t.assert_equal(std::string("\"") + v + "\": the desk puts back the book idle longest (slot 1)", 1, o.victim);
        }
    });

    t.test("hotness_otherwise", [&](testing & t) {
        for (const char * v : { (const char *) nullptr, "", "0", "x" }) {
            const std::string name = v ? std::string("\"") + v + "\"" : std::string("unset");
            const opened o = open_with(v, path);
            t.assert_equal(name + ": one line", (size_t) 1, o.lines.size());
            if (!o.lines.empty()) {
                t.assert_equal(name + ": the hotness line", std::string(HOTNESS_LINE), o.lines[0]);
            }
            t.assert_equal(name + ": the desk puts back the book used least (slot 0)", 0, o.victim);
        }
    });

    t.test("one_line_a_desk", [&](testing & t) {
        // a server with the apprentice opens two desks, each its own manager: the library's and the apprentice's
        setenv("LLAMA_MOE_STREAM_LRU", "1", 1);
        g_lines.clear();
        desk library, apprentice;
        library.mgr->open_files({ path });
        apprentice.mgr->open_files({ path });
        unsetenv("LLAMA_MOE_STREAM_LRU");
        t.assert_equal("two desks, two lines", (size_t) 2, g_lines.size());
        t.assert_true("both name lru", g_lines.size() == 2 && g_lines[0] == LRU_LINE && g_lines[1] == LRU_LINE);
    });

    t.test("no_parser_mistakes_it", [&](testing & t) {
        for (const std::string line : { LRU_LINE, HOTNESS_LINE }) {
            t.assert_true("not a window line to roomstats: " + line.substr(0, 50), parser_safe(line));
        }
        const std::string lru_mark = "moe stream: put-back rule: lru";
        const std::string hot_mark = "moe stream: put-back rule: hotness";
        t.assert_true("the lru mark is only in the lru line",
                std::string(LRU_LINE).find(lru_mark) != std::string::npos &&
                std::string(HOTNESS_LINE).find(lru_mark) == std::string::npos);
        t.assert_true("the hotness mark is only in the hotness line",
                std::string(HOTNESS_LINE).find(hot_mark) != std::string::npos &&
                std::string(LRU_LINE).find(hot_mark) == std::string::npos);
    });

    remove(path.c_str());
    return t.summary();
}

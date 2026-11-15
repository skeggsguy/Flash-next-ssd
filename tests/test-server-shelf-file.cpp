// The bookmark shelf's files (test-server-shelf's second file, tools/server/server-shelf-io.h and server-shelf.h):
//
// - a file round trip in odd-sized writes and reads across the 8 MiB pieces, its end refused, a missing file refused
// - the file cache is bypassed both ways: on macOS no page of a file written and read back straight through is left
//   in the file cache (mincore), where the same file through the cache leaves it all
// - the record: the library's and the apprentice's parts passed through by their owners, the carry state and the
//   copies come back as written; its size counted exactly; a part moving the wrong number of bytes, a cut file, a
//   wrong magic and a size past the file's end are refused
// - the folder: a missing folder, a file and a folder that takes no file are refused with why; the shelf's old files
//   (and nothing else) are deleted at open; the limit is capped to the drive; a dropped bookmark's file is deleted

#include "test-server-shelf.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

namespace fs = std::filesystem;

static const uint64_t MiB = 1024ull * 1024ull;
static const uint64_t GiB = 1024ull * MiB;

static fs::path test_dir(const char * name) {
    const fs::path d = fs::temp_directory_path() / ("test-server-shelf-" + std::to_string((long) getpid())) / name;
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d;
}

static std::vector<uint8_t> noise(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(n);
    for (auto & b : v) {
        b = (uint8_t) rng();
    }
    return v;
}

// odd chunk sizes, so pieces and calls never line up
static const size_t CHUNKS[] = { 1, 7, 4096, 1 * 1024 * 1024 + 3, 3 * 1024 * 1024 + 11 };

static bool write_odd(server_shelf_file & f, const std::vector<uint8_t> & v, size_t n) {
    size_t done = 0;
    for (int i = 0; done < n; i++) {
        const size_t k = std::min(CHUNKS[i % 5], n - done);
        if (!f.write(v.data() + done, k)) {
            return false;
        }
        done += k;
    }
    return true;
}

static bool read_odd(server_shelf_file & f, std::vector<uint8_t> & v, size_t n) {
    v.resize(n);
    size_t done = 0;
    for (int i = 0; done < n; i++) {
        const size_t k = std::min(CHUNKS[(i + 2) % 5], n - done);
        if (!f.read(v.data() + done, k)) {
            return false;
        }
        done += k;
    }
    return true;
}

#ifdef __APPLE__
// pages of the file in the file cache
static size_t resident_pages(const std::string & path) {
    const int fd = open(path.c_str(), O_RDONLY);
    struct stat st = {};
    fstat(fd, &st);
    const size_t n = (size_t) st.st_size;
    void * m = mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
    const size_t pg = (size_t) getpagesize();
    std::vector<char> vec((n + pg - 1) / pg);
    mincore(m, n, vec.data());
    size_t res = 0;
    for (char c : vec) {
        res += c & 1;
    }
    munmap(m, n);
    close(fd);
    return res;
}
#endif

void test_files() {
    const fs::path d = test_dir("files");
    const std::string path = (d / "a.shelf").string();

    // 20 MiB and a bit, across pieces, both ways
    const size_t n = 20 * MiB + 123;
    const auto data = noise(n, 1);
    {
        server_shelf_file f;
        CHECK(f.open_write(path));
        CHECK(write_odd(f, data, n));
        CHECK(f.close());
        CHECK(f.n_bytes() == n);
        CHECK(fs::file_size(path) == n);
    }
    {
        server_shelf_file f;
        std::vector<uint8_t> back;
        CHECK(f.open_read(path));
        CHECK(f.remaining() == n);
        CHECK(read_odd(f, back, n));
        CHECK(back == data);
        CHECK(f.remaining() == 0);
        uint8_t one = 0;
        CHECK(!f.read(&one, 1)); // past the end
        CHECK(f.error().find("ends early") != std::string::npos);
    }
    {
        server_shelf_file f;
        CHECK(!f.open_read((d / "missing.shelf").string()));
        CHECK(f.error().find("missing.shelf") != std::string::npos);
        server_shelf_file g;
        CHECK(g.open_write((d / "b.shelf").string()));
        CHECK(g.close());
        CHECK(!g.write("x", 1)); // closed
    }

#ifdef __APPLE__
    // straight through: nothing of the file left in the file cache, after the write or the read
    {
        const auto big = noise(16 * MiB + 1000, 2);
        server_shelf_file f;
        CHECK(f.open_write(path, true));
        CHECK(f.nocache());
        CHECK(write_odd(f, big, big.size()));
        CHECK(f.close());
        CHECK(resident_pages(path) == 0);

        server_shelf_file r;
        std::vector<uint8_t> back;
        CHECK(r.open_read(path, true));
        CHECK(read_odd(r, back, big.size()));
        CHECK(back == big);
        r.close();
        CHECK(resident_pages(path) == 0);

        // through the cache, the same file stays there: the check above can see a cached page
        const std::string path2 = (d / "cached.shelf").string();
        server_shelf_file c;
        CHECK(c.open_write(path2, false));
        CHECK(!c.nocache());
        CHECK(write_odd(c, big, big.size()));
        CHECK(c.close());
        CHECK(resident_pages(path2) > (16 * MiB / getpagesize()) / 2);
    }
#endif
    std::error_code ec;
    fs::remove_all(d, ec);
}

// a part owned elsewhere (a llama state), moved in odd chunks
static server_shelf_part put_part(const std::vector<uint8_t> & v, size_t short_by = 0) {
    return [&v, short_by](server_shelf_file & f, uint64_t size) {
        return size == v.size() && write_odd(f, v, v.size() - short_by);
    };
}

static server_shelf_part get_part(std::vector<uint8_t> & v, int & calls, size_t short_by = 0) {
    return [&v, &calls, short_by](server_shelf_file & f, uint64_t size) {
        calls++;
        return read_odd(f, v, size - short_by);
    };
}

static std::list<common_prompt_checkpoint> copies() {
    std::list<common_prompt_checkpoint> ckpts;
    for (int i = 0; i < 3; i++) {
        common_prompt_checkpoint c;
        c.n_tokens  = 4211 + 1000 * i;
        c.id_task   = 40 + i;
        c.pinned    = i == 0;
        c.pos_min   = (llama_pos) c.n_tokens - 1;
        c.pos_max   = (llama_pos) c.n_tokens - 1;
        c.data_tgt  = noise(1 * MiB + 17 * i, 10 + i);
        c.data_dft  = i == 1 ? std::vector<uint8_t>() : noise(1000 + i, 20 + i); // a copy without the apprentice
        c.data_spec = noise(8 * i, 30 + i);
        ckpts.push_back(std::move(c));
    }
    return ckpts;
}

static bool same_copies(const std::list<common_prompt_checkpoint> & a, const std::list<common_prompt_checkpoint> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (auto i = a.begin(), j = b.begin(); i != a.end(); ++i, ++j) {
        if (i->n_tokens != j->n_tokens || i->id_task != j->id_task || i->pinned != j->pinned ||
            i->pos_min != j->pos_min || i->pos_max != j->pos_max || i->data_tgt != j->data_tgt ||
            i->data_dft != j->data_dft || i->data_spec != j->data_spec) {
            return false;
        }
    }
    return true;
}

void test_record() {
    const fs::path d = test_dir("record");
    const std::string path = (d / "r.shelf").string();

    const auto tgt  = noise(9 * MiB + 5, 3); // crosses a piece
    const auto dft  = noise(1 * MiB + 9, 4);
    const auto spec = noise(4 + 64, 5);
    const auto ckpts = copies();

    for (bool with_dft : { true, false }) {
        server_shelf_head head;
        head.n_tokens    = 51234;
        head.spec_synced = !with_dft;
        const uint64_t size_dft = with_dft ? dft.size() : 0;
        const uint64_t want = server_shelf_record_size(tgt.size(), size_dft, spec.size(), ckpts);
        {
            server_shelf_file f;
            CHECK(f.open_write(path));
            CHECK(server_shelf_write_record(f, head, tgt.size(), put_part(tgt), size_dft, put_part(dft), spec, ckpts));
            CHECK(f.close());
            CHECK(f.n_bytes() == want);
            CHECK(fs::file_size(path) == want);
        }
        {
            server_shelf_file f;
            server_shelf_head back;
            std::vector<uint8_t> tgt_back, dft_back, spec_back;
            std::list<common_prompt_checkpoint> ckpts_back;
            uint64_t size_dft_back = 99;
            int calls_tgt = 0, calls_dft = 0;
            CHECK(f.open_read(path));
            CHECK(server_shelf_read_record(f, back, get_part(tgt_back, calls_tgt), get_part(dft_back, calls_dft),
                    size_dft_back, spec_back, ckpts_back));
            CHECK(back.n_tokens == 51234 && back.spec_synced == !with_dft);
            CHECK(tgt_back == tgt && calls_tgt == 1);
            CHECK(size_dft_back == size_dft);
            CHECK(with_dft ? (dft_back == dft && calls_dft == 1) : calls_dft == 0);
            CHECK(spec_back == spec);
            CHECK(same_copies(ckpts_back, ckpts));
        }
    }

    // a part that moves fewer bytes than it said: the write is refused
    {
        server_shelf_file f;
        CHECK(f.open_write(path));
        CHECK(!server_shelf_write_record(f, {}, tgt.size(), put_part(tgt, 1), 0, put_part(dft), spec, ckpts));
    }

    // a good file, then the ways it can go wrong
    {
        server_shelf_file f;
        CHECK(f.open_write(path));
        CHECK(server_shelf_write_record(f, {}, tgt.size(), put_part(tgt), dft.size(), put_part(dft), spec, ckpts));
        CHECK(f.close());
    }
    const auto read_back = [&](const std::string & p, size_t short_by = 0) {
        server_shelf_file f;
        server_shelf_head h;
        std::vector<uint8_t> a, b, s;
        std::list<common_prompt_checkpoint> c;
        uint64_t sd = 0;
        int ca = 0, cb = 0;
        return f.open_read(p) && server_shelf_read_record(f, h, get_part(a, ca, short_by), get_part(b, cb), sd, s, c);
    };
    CHECK(read_back(path));
    CHECK(!read_back(path, 1)); // the library's part read short

    const std::string cut = (d / "cut.shelf").string();
    fs::copy_file(path, cut, fs::copy_options::overwrite_existing);
    fs::resize_file(cut, fs::file_size(path) - 1);
    CHECK(!read_back(cut));

    const auto poke = [&](const std::string & p, uint64_t off, const void * src, size_t n) {
        fs::copy_file(path, p, fs::copy_options::overwrite_existing);
        std::fstream s(p, std::ios::in | std::ios::out | std::ios::binary);
        s.seekp((std::streamoff) off);
        s.write((const char *) src, (std::streamsize) n);
    };
    const std::string bad = (d / "bad.shelf").string();
    poke(bad, 0, "XKSHELF1", 8);
    CHECK(!read_back(bad));
    // the carry state's size, past the file's end: refused before anything is allocated
    const uint64_t huge = 1ull << 40;
    poke(bad, 32 + 8 + tgt.size() + 8 + dft.size(), &huge, sizeof(huge));
    CHECK(!read_back(bad));
    // the library's part's size, past the file's end
    poke(bad, 32, &huge, sizeof(huge));
    CHECK(!read_back(bad));

    std::error_code ec;
    fs::remove_all(d, ec);
}

static void touch(const fs::path & p) {
    std::ofstream(p.string()) << "x";
}

void test_folder() {
    const fs::path d = test_dir("folder");

    CHECK(server_shelf_check_dir((d / "missing").string()).find("does not exist") != std::string::npos);
    touch(d / "plain-file");
    CHECK(server_shelf_check_dir((d / "plain-file").string()) == "not a folder");
    CHECK(server_shelf_check_dir("") == "no folder given");
    CHECK(server_shelf_check_dir(d.string()).empty());
    CHECK(!fs::exists(d / ".bookmark-shelf-probe")); // the probe file is gone again
#ifndef _WIN32
    const fs::path ro = d / "read-only";
    fs::create_directories(ro);
    fs::permissions(ro, fs::perms::owner_read | fs::perms::owner_exec);
    if (getuid() != 0) {
        CHECK(server_shelf_check_dir(ro.string()).find("cannot write") != std::string::npos);
    }
    fs::permissions(ro, fs::perms::owner_all);
#endif

    // the shelf's own files only
    touch(d / "bookmark-1.shelf");
    touch(d / "bookmark-22.shelf");
    touch(d / "bookmark-.shelf");
    touch(d / "bookmark-3.shelf.keep");
    touch(d / "notes.txt");
    CHECK(server_shelf_clear_dir(d.string()) == 2);
    CHECK(!fs::exists(d / "bookmark-1.shelf") && !fs::exists(d / "bookmark-22.shelf"));
    CHECK(fs::exists(d / "bookmark-.shelf") && fs::exists(d / "bookmark-3.shelf.keep") && fs::exists(d / "notes.txt"));
    CHECK(server_shelf_file_name(7) == "bookmark-7.shelf");

    // open: old bookmarks deleted, the limit capped to the drive (60 GiB free: 54 usable, 6 kept free)
    touch(d / "bookmark-5.shelf");
    {
        server_shelf shelf;
        shelf.free_bytes = [](uint64_t & out) { out = 60 * GiB; return true; };
        add_entry(shelf, 9, toks(0, 10, 0, 0), 1); // a stale entry: open starts empty
        CHECK(shelf.open(d.string(), 100 * GiB).empty());
        CHECK(!fs::exists(d / "bookmark-5.shelf"));
        CHECK(shelf.entries.empty());
        CHECK(shelf.lim.capped && shelf.lim.limit == 54 * GiB && shelf.lim.reserve == 6 * GiB);
        CHECK(shelf.free_at_open == 60 * GiB && shelf.limit_asked == 100 * GiB);
        CHECK(shelf.path_for(3) == (d / "bookmark-3.shelf").string());

        // a dropped bookmark's file is deleted
        touch(shelf.path_for(3));
        add_entry(shelf, 3, toks(0, 10, 0, 0), 1);
        shelf.entries.back().path = shelf.path_for(3);
        shelf.drop(3);
        CHECK(shelf.entries.empty() && !fs::exists(shelf.path_for(3)));
    }
    {
        // the real drive: under the free space, not capped
        server_shelf shelf;
        CHECK(shelf.open(d.string(), 1 * MiB).empty());
        CHECK(!shelf.lim.capped && shelf.lim.limit == 1 * MiB && shelf.free_at_open > 0);
    }
    {
        server_shelf shelf;
        CHECK(!shelf.open((d / "missing").string(), GiB).empty());
        server_shelf full;
        full.free_bytes = [](uint64_t & out) { out = 0; return true; };
        CHECK(full.open(d.string(), GiB).find("limit of 0") != std::string::npos);
        server_shelf blind;
        blind.free_bytes = [](uint64_t &) { return false; };
        CHECK(!blind.open(d.string(), GiB).empty());
    }

    std::error_code ec;
    fs::remove_all(d.parent_path(), ec);
}

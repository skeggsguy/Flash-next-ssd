#include "server-shelf-io.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

#include <fcntl.h>

#ifdef _WIN32
#include <io.h>
#include <malloc.h>
#define SHELF_OPEN(p, fl, mode) _open((p), (fl) | _O_BINARY, (mode))
#define SHELF_READ  _read
#define SHELF_WRITE _write
#define SHELF_CLOSE _close
#else
#include <unistd.h>
#define SHELF_OPEN(p, fl, mode) ::open((p), (fl), (mode))
#define SHELF_READ  ::read
#define SHELF_WRITE ::write
#define SHELF_CLOSE ::close
#endif

static constexpr size_t SHELF_ALIGN = 16384; // Apple silicon's page size; a multiple of every other's

static uint8_t * shelf_alloc(size_t n) {
#ifdef _WIN32
    return (uint8_t *) _aligned_malloc(n, SHELF_ALIGN);
#else
    void * p = nullptr;
    return posix_memalign(&p, SHELF_ALIGN, n) == 0 ? (uint8_t *) p : nullptr;
#endif
}

static void shelf_free(uint8_t * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// the file cache bypass: F_NOCACHE (macOS) is set at open and covers reads and writes; elsewhere posix_fadvise
// drops the pages a piece leaves behind (shelf_drop_pages)
static bool shelf_set_nocache(int fd) {
#if defined(F_NOCACHE)
    return fcntl(fd, F_NOCACHE, 1) != -1;
#elif defined(POSIX_FADV_DONTNEED)
    (void) fd;
    return true;
#else
    (void) fd;
    return false;
#endif
}

static void shelf_drop_pages(int fd, bool writing) {
#if !defined(F_NOCACHE) && defined(POSIX_FADV_DONTNEED)
    if (writing) {
        fdatasync(fd); // dirty pages are dropped only once written
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#else
    (void) fd;
    (void) writing;
#endif
}

server_shelf_file::~server_shelf_file() {
    close();
}

bool server_shelf_file::fail(const std::string & what, bool with_errno) {
    failed = true;
    if (err.empty()) {
        err = what + " (" + path + ")" + (with_errno ? std::string(": ") + std::strerror(errno) : std::string());
    }
    return false;
}

bool server_shelf_file::open_write(const std::string & p, bool nocache) {
    path    = p;
    writing = true;
    fd = SHELF_OPEN(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return fail("cannot create the file");
    }
    nocache_on = nocache && shelf_set_nocache(fd);
    buf = shelf_alloc(PIECE);
    if (buf == nullptr) {
        return fail("cannot allocate the file's buffer");
    }
    return true;
}

bool server_shelf_file::open_read(const std::string & p, bool nocache) {
    path    = p;
    writing = false;
    fd = SHELF_OPEN(p.c_str(), O_RDONLY, 0);
    if (fd < 0) {
        return fail("cannot open the file");
    }
    std::error_code ec;
    n_file = (uint64_t) std::filesystem::file_size(p, ec);
    if (ec) {
        return fail("cannot read the file's size", false);
    }
    nocache_on = nocache && shelf_set_nocache(fd);
    buf = shelf_alloc(PIECE);
    if (buf == nullptr) {
        return fail("cannot allocate the file's buffer");
    }
    return true;
}

bool server_shelf_file::flush_piece() {
    size_t done = 0;
    while (done < buf_pos) {
        const auto n = SHELF_WRITE(fd, buf + done, (unsigned) (buf_pos - done));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return fail("write failed");
        }
        if (n == 0) {
            return fail("write failed: nothing written", false);
        }
        done += (size_t) n;
    }
    if (nocache_on) {
        shelf_drop_pages(fd, true);
    }
    buf_pos = 0;
    return true;
}

bool server_shelf_file::fill_piece() {
    buf_len = 0;
    buf_pos = 0;
    while (buf_len < PIECE && !eof) {
        const auto n = SHELF_READ(fd, buf + buf_len, (unsigned) (PIECE - buf_len));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return fail("read failed");
        }
        if (n == 0) {
            eof = true;
        }
        buf_len += (size_t) n;
    }
    if (nocache_on) {
        shelf_drop_pages(fd, false);
    }
    return true;
}

bool server_shelf_file::write(const void * src, size_t size) {
    if (fd < 0 || !writing || failed) {
        return false;
    }
    const uint8_t * p = (const uint8_t *) src;
    while (size > 0) {
        const size_t n = std::min(size, PIECE - buf_pos);
        std::memcpy(buf + buf_pos, p, n);
        buf_pos += n;
        p       += n;
        size    -= n;
        n_done  += n;
        if (buf_pos == PIECE && !flush_piece()) {
            return false;
        }
    }
    return true;
}

bool server_shelf_file::read(void * dst, size_t size) {
    if (fd < 0 || writing || failed) {
        return false;
    }
    uint8_t * p = (uint8_t *) dst;
    while (size > 0) {
        if (buf_pos == buf_len) {
            if (eof || !fill_piece()) {
                return failed ? false : fail("the file ends early", false);
            }
            if (buf_len == 0) {
                return fail("the file ends early", false);
            }
        }
        const size_t n = std::min(size, buf_len - buf_pos);
        std::memcpy(p, buf + buf_pos, n);
        buf_pos += n;
        p       += n;
        size    -= n;
        n_done  += n;
    }
    return true;
}

bool server_shelf_file::close() {
    bool ok = !failed;
    if (fd >= 0) {
        if (writing && !failed && buf_pos > 0) {
            ok = flush_piece();
        }
        if (SHELF_CLOSE(fd) != 0 && writing && ok) {
            ok = fail("close failed");
        }
        fd = -1;
    }
    if (buf != nullptr) {
        shelf_free(buf);
        buf = nullptr;
    }
    return ok;
}

std::string server_shelf_check_dir(const std::string & dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (dir.empty()) {
        return "no folder given";
    }
    if (!fs::exists(dir, ec)) {
        return "the folder does not exist (is its drive connected?)";
    }
    if (!fs::is_directory(dir, ec)) {
        return "not a folder";
    }
    const std::string probe = (fs::path(dir) / ".bookmark-shelf-probe").string();
    server_shelf_file f;
    const bool ok = f.open_write(probe) && f.write("shelf", 5) && f.close();
    const std::string why = f.error();
    fs::remove(probe, ec);
    if (!ok) {
        return "cannot write a file there: " + why;
    }
    return "";
}

bool server_shelf_free_bytes(const std::string & dir, uint64_t & out) {
    std::error_code ec;
    const auto sp = std::filesystem::space(dir, ec);
    if (ec) {
        return false;
    }
    out = (uint64_t) sp.available;
    return true;
}

std::string server_shelf_file_name(uint64_t id) {
    return "bookmark-" + std::to_string(id) + ".shelf";
}

size_t server_shelf_clear_dir(const std::string & dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    size_t n = 0;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        const bool ours = name.size() > 15 && name.rfind("bookmark-", 0) == 0 &&
                          name.compare(name.size() - 6, 6, ".shelf") == 0;
        if (ours && it->is_regular_file(ec) && fs::remove(it->path(), ec)) {
            n++;
        }
    }
    return n;
}

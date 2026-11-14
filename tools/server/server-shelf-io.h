#pragma once

// The bookmark shelf's files (--cache-disk, server-shelf.h): one file per saved conversation, written and read
// straight through to the drive. A bookmark at 200K tokens is ~6 GB: it goes through one 8 MiB page-aligned buffer,
// so it never needs RAM of its own size, and with the file cache bypassed (F_NOCACHE on macOS, posix_fadvise
// elsewhere) it never leaves GBs of clean pages in macOS's file cache either, which on a full 64 GB machine is what
// pushes the rest of memory toward swap.

#include <cstddef>
#include <cstdint>
#include <string>

class server_shelf_file {
public:
    static constexpr size_t PIECE = 8u*1024u*1024u; // bytes per read or write call to the drive

    server_shelf_file() = default;
    ~server_shelf_file();

    server_shelf_file(const server_shelf_file &) = delete;
    server_shelf_file & operator=(const server_shelf_file &) = delete;

    // nocache: bypass the file cache (the shelf always does; tests compare with it off)
    bool open_write(const std::string & path, bool nocache = true);
    bool open_read (const std::string & path, bool nocache = true);

    bool write(const void * src, size_t size);
    bool read (void * dst, size_t size);

    // writing: the last partial piece goes to the drive; false if any write failed. Idempotent; the destructor
    // closes too, without a report
    bool close();

    uint64_t n_bytes() const { return n_done; } // bytes passed through write() or read()
    uint64_t remaining() const { return n_file > n_done ? n_file - n_done : 0; } // reading: bytes not yet read
    bool     nocache() const { return nocache_on; } // the file cache is bypassed for this file
    const std::string & error() const { return err; }

private:
    bool fail(const std::string & what, bool with_errno = true);
    bool flush_piece();
    bool fill_piece();

    int       fd         = -1;
    bool      writing    = false;
    bool      nocache_on = false;
    bool      failed     = false;
    uint8_t * buf        = nullptr; // PIECE bytes, page-aligned
    size_t    buf_len    = 0;       // reading: valid bytes in buf
    size_t    buf_pos    = 0;       // writing: bytes waiting; reading: bytes already served
    bool      eof        = false;
    uint64_t  n_done     = 0;
    uint64_t  n_file     = 0;       // reading: the file's size
    std::string path;
    std::string err;
};

// "" when dir is an existing folder that takes a new file (one is created and deleted), else why not
std::string server_shelf_check_dir(const std::string & dir);

// the bytes free to this process on dir's volume (statvfs's f_bavail); false if it cannot be read
bool server_shelf_free_bytes(const std::string & dir, uint64_t & out);

// a bookmark's file name in the shelf's folder: bookmark-<id>.shelf
std::string server_shelf_file_name(uint64_t id);

// deletes the shelf's own files in dir (bookmark-*.shelf, nothing else); returns how many
size_t server_shelf_clear_dir(const std::string & dir);

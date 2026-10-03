#pragma once

// Private to the book manager's own files (src/llama-moe-stream*.cpp): the constants and the read
// helpers they share. Not for includers of llama-moe-stream.h.

#include "llama-mmap.h"

#include <cstddef>
#include <cstdint>

static const uint32_t MOE_STREAM_IO_THREADS_DEFAULT = 9;
static const uint32_t MOE_STREAM_IO_THREADS_MAX     = 18;
// Route-hotness halves every this many tokens. 1024, not the original 64: at 64 an expert accumulates
// only ~1.5 uses between halvings (256 experts, 6 picked per token), so counters sit at 0-3, cannot
// rank experts, and eviction degenerates into plain LRU. Measured on decode, 3 runs each:
//   64 -> 7.54 / 7.61 / 7.82 t/s, miss ~6.8%     1024 -> 9.06 / 8.85 / 9.11 t/s, miss ~5.4%
static const int64_t  MOE_STREAM_HOT_DECAY_TOKENS   = 1024;

// O_DIRECT alignment: 4096 is a multiple of any device logical block size (512/4096), so it is
// universally valid, and reading a few extra KB of head/tail padding per slab is negligible
#if defined(__APPLE__)
static const char * const MOE_STREAM_DIRECT_NAME = "F_NOCACHE";
#else
static const char * const MOE_STREAM_DIRECT_NAME = "O_DIRECT";
#endif

static const size_t MOE_STREAM_DIRECT_ALIGN = 4096;

// saturating increment - route-hotness counters accumulate over a whole run and must not wrap
static inline uint32_t sat_inc(uint32_t & c) {
    if (c < UINT32_MAX - 1) {
        c++;
    }
    return c;
}

// A desk slot's state is guarded by the manager's lock, with one exception: the remap's quick path (no lock when
// nothing is missing, llama-moe-stream-quick.cpp) reads it on the graph thread without the lock. The two stores
// that make a slot RESIDENT (a runner's last slab, and a restore from the lent belt) publish with release and the
// quick path reads with acquire, so a slot it sees RESIDENT has every byte of its book in place. Every other
// access is under the lock, as before. Without the GCC/Clang atomic builtins the quick path is compiled out.
#if defined(__GNUC__) || defined(__clang__)
static const bool MOE_STREAM_QUICK_BUILT = true;
static inline uint8_t moe_slot_state_peek(const uint8_t & s) { return __atomic_load_n(&s, __ATOMIC_ACQUIRE); }
static inline void    moe_slot_state_publish(uint8_t & s, uint8_t v) { __atomic_store_n(&s, v, __ATOMIC_RELEASE); }
#else
static const bool MOE_STREAM_QUICK_BUILT = false;
static inline uint8_t moe_slot_state_peek(const uint8_t & s) { return s; }
static inline void    moe_slot_state_publish(uint8_t & s, uint8_t v) { s = v; }
#endif

// page-aligned allocation and positional read, defined beside the I/O workers
void *          moe_aligned_alloc(size_t n);
void            moe_aligned_free(void * p);
const uint8_t * llama_moe_stream_pread(llama_file & file, uint8_t * staging, size_t len, size_t offs, bool direct);

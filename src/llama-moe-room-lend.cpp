#include "llama-moe-room-lend.h"

#include <algorithm>

static size_t align_up(size_t n) {
    return (n + LLAMA_MOE_LEND_ALIGN - 1)/LLAMA_MOE_LEND_ALIGN*LLAMA_MOE_LEND_ALIGN;
}

static bool overlaps(size_t a0, size_t a1, size_t b0, size_t b1) {
    return a0 < b1 && b0 < a1;
}

bool llama_moe_lend_copy::complete() const {
    for (int32_t i = 0; i < n_slabs; i++) {
        if (slab[i] != LLAMA_MOE_LEND_DONE) {
            return false;
        }
    }
    return n_slabs > 0;
}

bool llama_moe_lend::place(size_t bytes, size_t & offs, size_t & n_drop) const {
    if (bytes == 0 || bytes > size) {
        return false;
    }
    // after the newest copy, or from the start when the end is too short; then [skip, size) is given up
    // too, so the copies there (the oldest) go, and the ring stays first in, first out
    size_t at   = align_up(head);
    size_t skip = size;
    if (at + bytes > size) {
        skip = std::min(at, size);
        at   = 0;
    }
    // Copies sit in the ring's order around the belt, so the ones in the way are always the oldest: drop
    // from the front until one is clear of the new copy, and every newer one is clear too.
    n_drop = 0;
    for (const auto & c : ring) {
        if (!overlaps(c.offs, c.offs + c.bytes, at, at + bytes) && !overlaps(c.offs, c.offs + c.bytes, skip, size)) {
            break;
        }
        if (c.busy()) {
            return false; // a keep still being copied, or a restore still reading it
        }
        n_drop++;
    }
    offs = at;
    return true;
}

uint64_t & llama_moe_lend::index_of(int32_t il, int32_t book) {
    if ((size_t) il >= index.size()) {
        index.resize(il + 1);
    }
    if ((size_t) book >= index[il].size()) {
        index[il].resize(book + 1, 0);
    }
    return index[il][book];
}

llama_moe_lend_copy * llama_moe_lend::keep(int32_t il, int32_t book, int32_t slot, size_t bytes, int32_t n_slabs,
        llama_moe_lend_keep & why) {
    why = LLAMA_MOE_LEND_BLOCKED;
    if (il < 0 || book < 0 || n_slabs < 1 || n_slabs > LLAMA_MOE_LEND_SLABS_MAX) {
        return nullptr;
    }
    // held, complete or still being copied: a book is never copied twice, and a restored book keeps its copy
    const llama_moe_lend_copy * old = at(index_of(il, book));
    if (old != nullptr && old->il == il && old->book == book) {
        why = LLAMA_MOE_LEND_HELD;
        return nullptr;
    }
    size_t offs = 0, n_drop = 0;
    if (!place(bytes, offs, n_drop)) {
        return nullptr;
    }
    for (size_t i = 0; i < n_drop; i++) {
        ring.pop_front();
    }
    n_dropped += (int64_t) n_drop;

    llama_moe_lend_copy c;
    c.seq     = next_seq++;
    c.il      = il;
    c.book    = book;
    c.slot    = slot;
    c.offs    = offs;
    c.bytes   = bytes;
    c.n_slabs = n_slabs;
    ring.push_back(c);
    head = offs + bytes;
    index_of(il, book) = c.seq;
    why = LLAMA_MOE_LEND_KEPT;
    return &ring.back();
}

llama_moe_lend_copy * llama_moe_lend::at(uint64_t seq) {
    if (ring.empty() || seq < ring.front().seq || seq > ring.back().seq) {
        return nullptr;
    }
    return &ring[seq - ring.front().seq];
}

llama_moe_lend_copy * llama_moe_lend::find(int32_t il, int32_t book) {
    if (il < 0 || book < 0 || (size_t) il >= index.size() || (size_t) book >= index[il].size()) {
        return nullptr;
    }
    llama_moe_lend_copy * c = at(index[il][book]);
    return c != nullptr && c->il == il && c->book == book && c->complete() ? c : nullptr;
}

bool llama_moe_lend::claim(uint64_t seq, int32_t slab) {
    llama_moe_lend_copy * c = at(seq);
    if (c == nullptr || slab < 0 || slab >= c->n_slabs || c->slab[slab] != LLAMA_MOE_LEND_PENDING) {
        return false;
    }
    c->slab[slab] = LLAMA_MOE_LEND_RUNNING;
    return true;
}

bool llama_moe_lend::finish(uint64_t seq, int32_t slab) {
    llama_moe_lend_copy * c = at(seq);
    if (c == nullptr || slab < 0 || slab >= c->n_slabs || c->slab[slab] != LLAMA_MOE_LEND_RUNNING) {
        return false;
    }
    c->slab[slab] = LLAMA_MOE_LEND_DONE;
    return c->complete();
}

void llama_moe_lend::pin(uint64_t seq) {
    if (llama_moe_lend_copy * c = at(seq)) {
        c->pins++;
    }
}

void llama_moe_lend::unpin(uint64_t seq) {
    llama_moe_lend_copy * c = at(seq);
    if (c != nullptr && c->pins > 0) {
        c->pins--;
    }
}

size_t llama_moe_lend::clear() {
    const size_t n = ring.size();
    ring.clear(); // the index is left as it is: its seqs are all older than the ring's from here on
    head = 0;
    return n;
}

bool llama_moe_lend::busy() const {
    return std::any_of(ring.begin(), ring.end(), [](const llama_moe_lend_copy & c) { return c.busy(); });
}

size_t llama_moe_lend::held() const {
    return (size_t) std::count_if(ring.begin(), ring.end(), [](const llama_moe_lend_copy & c) { return c.complete(); });
}

uint32_t llama_moe_lend_cap(uint32_t n_expert_used, size_t belt_bytes, size_t sum_strides) {
    const size_t share = sum_strides > 0 ? belt_bytes/sum_strides : 0;
    return (uint32_t) std::max<size_t>(n_expert_used, share);
}

uint32_t llama_moe_lend_capacity(size_t belt_bytes, size_t sum_strides, size_t n_floors) {
    return sum_strides > 0 ? (uint32_t) (belt_bytes*n_floors/sum_strides) : 0;
}

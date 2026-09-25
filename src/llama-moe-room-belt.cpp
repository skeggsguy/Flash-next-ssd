#include "llama-moe-room-belt.h"

static size_t align_up(size_t n, size_t a) {
    return (n + a - 1)/a*a;
}

bool llama_moe_belt::place(uint32_t n_records, size_t stride, size_t & offs) const {
    const size_t bytes = (size_t) n_records*stride;
    if (stride == 0 || bytes == 0 || bytes > size) {
        return false;
    }
    if (parts.empty()) {
        offs = 0;
        return true;
    }
    const size_t head    = parts.front().offs;
    const size_t tail    = parts.back().offs + parts.back().bytes;
    const bool   wrapped = parts.back().offs < head; // the newest part sits before the oldest

    const size_t at = align_up(tail, stride);
    if (!wrapped) {
        if (at + bytes <= size) {  // after the newest part
            offs = at;
            return true;
        }
        if (bytes <= head) {       // wrap: from the start up to the oldest part
            offs = 0;
            return true;
        }
        return false;
    }
    if (at + bytes <= head) {      // between the newest part and the oldest
        offs = at;
        return true;
    }
    return false;
}

llama_moe_part * llama_moe_belt::push(int32_t il, int32_t index, uint32_t n_records, size_t stride, int32_t pending) {
    size_t offs = 0;
    if (!place(n_records, stride, offs)) {
        return nullptr;
    }
    llama_moe_part p;
    p.seq       = next_seq++;
    p.il        = il;
    p.index     = index;
    p.offs      = offs;
    p.bytes     = (size_t) n_records*stride;
    p.stride    = stride;
    p.n_records = n_records;
    p.pending   = pending;
    p.state     = pending > 0 ? LLAMA_MOE_PART_FILLING : LLAMA_MOE_PART_READY;
    parts.push_back(p);
    return &parts.back();
}

llama_moe_part * llama_moe_belt::find(uint64_t seq) {
    for (auto & p : parts) {
        if (p.seq == seq) {
            return &p;
        }
    }
    return nullptr;
}

llama_moe_part * llama_moe_belt::begin_read(uint64_t seq) {
    llama_moe_part * p = find(seq);
    if (p == nullptr || p->state != LLAMA_MOE_PART_FILLING) {
        return nullptr;
    }
    p->inflight++;
    return p;
}

bool llama_moe_belt::end_read(uint64_t seq, bool ok) {
    llama_moe_part * p = find(seq);
    if (p == nullptr || p->inflight <= 0) {
        return false; // a read is only ended after begin_read, which keeps the part until it lands
    }
    p->inflight--;
    if (!ok || p->state != LLAMA_MOE_PART_FILLING || p->pending <= 0) {
        return false;
    }
    if (--p->pending == 0) {
        p->state = LLAMA_MOE_PART_READY;
        return true;
    }
    return false;
}

void llama_moe_belt::release_through(uint64_t through) {
    for (auto & p : parts) {
        if (p.seq <= through && (p.state == LLAMA_MOE_PART_READY || p.state == LLAMA_MOE_PART_IN_USE)) {
            p.state = LLAMA_MOE_PART_RELEASED;
        }
    }
}

void llama_moe_belt::cancel_all() {
    for (auto & p : parts) {
        if (p.state != LLAMA_MOE_PART_RELEASED) {
            p.state = LLAMA_MOE_PART_CANCELLED;
        }
    }
}

size_t llama_moe_belt::reclaim() {
    size_t n = 0;
    while (!parts.empty() && parts.front().inflight == 0 &&
           (parts.front().state == LLAMA_MOE_PART_RELEASED || parts.front().state == LLAMA_MOE_PART_CANCELLED)) {
        parts.pop_front();
        n++;
    }
    return n;
}

size_t llama_moe_belt::held_bytes() const {
    size_t n = 0;
    for (const auto & p : parts) {
        n += p.bytes;
    }
    return n;
}

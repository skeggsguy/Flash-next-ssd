#include "llama-moe-room-plan.h"

#include "ggml.h"

llama_moe_room_floor_plan llama_moe_room_plan_floor(uint32_t n_expert, const std::vector<int32_t> & slot_expert,
        const std::vector<uint8_t> & is_alt, int32_t n_parts) {
    GGML_ASSERT(n_parts >= 1);
    GGML_ASSERT(is_alt.size() == n_expert);

    std::vector<uint8_t> on_desk(n_expert, 0);
    std::vector<int32_t> empty; // EMPTY desk slots, in slot order
    for (size_t s = 0; s < slot_expert.size(); s++) {
        const int32_t e = slot_expert[s];
        if (e < 0) {
            empty.push_back((int32_t) s);
        } else {
            GGML_ASSERT((uint32_t) e < n_expert && !on_desk[e]);
            on_desk[e] = 1;
        }
    }

    // the missing books of each runner, in id order (so each runner reads its file front to back)
    std::vector<int32_t> home, alt;
    for (uint32_t e = 0; e < n_expert; e++) {
        if (!on_desk[e]) {
            (is_alt[e] ? alt : home).push_back((int32_t) e);
        }
    }

    // Interleave: after m books, floor(m*H/N) of them are the first runner's. Any run of the order - a
    // part - then holds each runner's books in proportion to within one book, so neither runner idles
    // while the other works through a part alone.
    const size_t H = home.size();
    const size_t N = home.size() + alt.size();
    std::vector<int32_t> order;
    order.reserve(N);
    for (size_t m = 0, h = 0, a = 0; m < N; m++) {
        if ((m + 1)*H/N > h) {
            order.push_back(home[h++]);
        } else {
            order.push_back(alt[a++]);
        }
    }

    llama_moe_room_floor_plan plan;

    // a cold desk takes the first books into its EMPTY slots; a warm one is never evicted
    size_t next = 0;
    for (; next < order.size() && next < empty.size(); next++) {
        plan.fills.emplace_back(order[next], empty[next]);
    }

    // the rest in n_parts parts, the first (rest % n_parts) of them one book longer
    const size_t rest = order.size() - next;
    plan.parts.resize(n_parts);
    for (int32_t p = 0; p < n_parts; p++) {
        const size_t len = rest/n_parts + ((size_t) p < rest % n_parts ? 1 : 0);
        plan.parts[p].assign(order.begin() + next, order.begin() + next + len);
        next += len;
    }
    GGML_ASSERT(next == order.size());

    return plan;
}

int64_t llama_moe_room_emit(const int32_t * ids, int64_t n, const std::vector<int32_t> & where, int32_t * out) {
    int64_t owned = 0;
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (size_t) e < where.size());
        out[i] = where[e];
        owned += out[i] >= 0;
    }
    return owned;
}

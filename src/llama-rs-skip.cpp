#include "llama-rs-skip.h"

#include "llama-flashnext.h"
#include "llama-impl.h"

#include <cstring>

bool llama_rs_skip_slots_parse(const char * value) {
    return llama_flashnext_rs_skip_slots(value); // unset: on, the fork's default
}

bool llama_rs_skip_slots_init(const char * value, uint32_t n_rs_seq) {
    const bool on = llama_rs_skip_slots_parse(value);
    if (value != nullptr && !on && std::strcmp(value, "0") != 0) {
        LLAMA_LOG_WARN("%s: LLAMA_RS_SKIP_SLOTS=%s is not 0 or 1: off\n", __func__, value);
    }
    if (on) {
        LLAMA_LOG_WARN("%s: undo copies: only the slots a rollback can reach, n_rs_seq = %u "
                "(LLAMA_RS_SKIP_SLOTS=1; a rollback past the last round is refused)\n", __func__, n_rs_seq);
    } else {
        LLAMA_LOG_WARN("%s: undo copies: every slot each round, n_rs_seq = %u "
                "(LLAMA_RS_SKIP_SLOTS=1 writes only the reachable ones)\n", __func__, n_rs_seq);
    }
    return on;
}

// The fork's proven defaults: what each switch means unset, and set. The rules are in llama-flashnext.h.

#include "llama-flashnext.h"

#include "llama-impl.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// the everyday values (serve.py's launch, 2026-10-15), each the unset meaning
static const int64_t  FLASHNEXT_ALLOC_CHUNK_MIB = 4096;
static const uint32_t FLASHNEXT_DEPTH2          = 4;
static const float    FLASHNEXT_CONF            = 0.2f;

// "" and "0" are off, anything else on (the lookahead's switches have always read this way)
static bool flashnext_flag(const char * value) {
    return *value != '\0' && strcmp(value, "0") != 0;
}

bool llama_flashnext_lru(const char * value) {
    return value == nullptr || atoi(value) != 0; // an EMPTY value must mean off, not on
}

bool llama_flashnext_rs_skip_slots(const char * value) {
    return value == nullptr || strcmp(value, "1") == 0;
}

int64_t llama_flashnext_alloc_chunk_mib(const char * value) {
    return value == nullptr ? FLASHNEXT_ALLOC_CHUNK_MIB : std::max<int64_t>(0, atoll(value));
}

bool llama_flashnext_desk_first(const char * value) {
    return value == nullptr || flashnext_flag(value);
}

bool llama_flashnext_guesser_named(const char * guesser_value) {
    return guesser_value != nullptr && *guesser_value != '\0';
}

bool llama_flashnext_lookahead_all(const char * value, bool guesser) {
    return value == nullptr ? guesser : flashnext_flag(value);
}

uint32_t llama_flashnext_lookahead_depth2(const char * value, bool guesser) {
    if (value == nullptr) {
        return guesser ? FLASHNEXT_DEPTH2 : 0;
    }
    return (uint32_t) std::max(0, atoi(value));
}

bool llama_flashnext_lookahead_keep(const char * value, bool guesser) {
    return value == nullptr ? guesser : flashnext_flag(value);
}

float llama_flashnext_lookahead_conf(const char * value, bool guesser) {
    if (value == nullptr) {
        return guesser ? FLASHNEXT_CONF : 0.0f;
    }
    if (!flashnext_flag(value)) {
        return 0.0f;
    }
    char * end = nullptr;
    errno = 0;
    const float cut = std::strtof(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !(cut > 0.0f && cut <= 1.0f)) {
        throw std::runtime_error(format("LLAMA_MOE_STREAM_LOOKAHEAD_CONF=%s is not a confidence cut in (0, 1]", value));
    }
    return cut;
}

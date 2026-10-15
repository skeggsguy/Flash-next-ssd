// The residency keep-alive's switch. See ggml-metal-keep-alive.h.

#include "ggml-metal-keep-alive.h"

#include <cstdlib>

int ggml_metal_keep_alive_parse(const char * s) {
    if (s == nullptr) {
        return GGML_METAL_KEEP_ALIVE_DEFAULT_S;
    }
    const int v = atoi(s);
    return v > 0 ? v : GGML_METAL_KEEP_ALIVE_OLD_S;
}

// The small-batch mat-vec up to 16 columns. See ggml-metal-mv-ext.h.

#include "ggml-metal-mv-ext.h"

#include <cerrno>
#include <cstdlib>

int ggml_metal_mv_ext_max_parse(const char * s, bool * bad) {
    if (bad) {
        *bad = false;
    }
    if (s == nullptr || *s == '\0') {
        return GGML_METAL_MV_EXT_MAX_NEW;
    }
    char * end = nullptr;
    errno = 0;
    const long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno != 0) {
        if (bad) {
            *bad = true;
        }
        return GGML_METAL_MV_EXT_MAX_NEW;
    }
    if (v <= GGML_METAL_MV_EXT_MAX_OLD) {
        return GGML_METAL_MV_EXT_MAX_OLD; // "0" is off, and nothing narrower than today
    }
    return v >= GGML_METAL_MV_EXT_MAX_NEW ? GGML_METAL_MV_EXT_MAX_NEW : (int) v;
}

int ggml_metal_mv_ext_limit(int64_t ne01, int mv_ext_max) {
    // test-backend-ops perf (M5 Pro, 2026-09-28, every small-batch type at the model's shapes): 9-16 columns
    // beat mul_mm on every matrix of 2,560 rows or fewer (0.2-0.9x the time); on 6,144 rows and more
    // mul_mv_ext's time grows with each round of threadgroups while mul_mm's barely moves, so 9-10 columns
    // still win (0.6-0.9x) and 11-12 are even to 1.2x, 13-16 up to 1.5x
    if (ne01 >= GGML_METAL_MV_EXT_ROWS_BIG) {
        return mv_ext_max < GGML_METAL_MV_EXT_MAX_BIG ? mv_ext_max : GGML_METAL_MV_EXT_MAX_BIG;
    }
    return mv_ext_max;
}

int ggml_metal_mv_ext_r1ptg_wide(int ne11) {
    // the fewest rounds of threadgroups (upstream's rule for 2-8: 6 -> 3, 7 and 8 -> 4), evenly filled on a
    // tie: 9-10 in two rounds of 5, 11-12 in three of 4, 13-15 in three of 5, 16 in four of 4. Measured
    // against the writing analysis' prototype (88205cfbb: 9 and 12 -> 3, 13 and 14 -> 4): as fast or faster
    // at every count and shape (9 columns on 6,144-12,288 Q8_0 rows: 0.67-0.76x mul_mm's time, against
    // 0.68-0.78x)
    switch (ne11) {
        case  9: case 10:                     return 5;
        case 11: case 12: case 16:            return 4;
        case 13: case 14: case 15:            return 5;
        default:                              return 0;
    }
}

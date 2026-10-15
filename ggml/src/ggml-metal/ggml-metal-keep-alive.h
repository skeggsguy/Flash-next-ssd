// How long Metal keeps the library's buffers resident after the last request (GGML_METAL_RESIDENCY_KEEP_ALIVE_S).
//
// The residency sets are re-requested every few ms while work comes in and released after keep_alive seconds of
// quiet; the next request then pins every buffer again at once (~41 GiB for the Flash-Next library, a small rush).
// The fork keeps them for 10,000,000 s (~115 days, near the loop counter's int limit) unless told otherwise: the
// Flash-Next study's everyday server ran so from 2026-09-23 (its runner/idle_test.py measured the release), and it
// became the engine's default on 2026-10-15 (the fork's proven defaults, src/llama-flashnext.h).
//
//   unset                    10000000 s
//   "0", a negative or not a number   180 s, upstream's 3 minutes (the old way)
//   N > 0                    N s

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_METAL_KEEP_ALIVE_DEFAULT_S 10000000
#define GGML_METAL_KEEP_ALIVE_OLD_S     (3*60)

// the keep-alive in seconds for the switch's value (NULL: unset)
int ggml_metal_keep_alive_parse(const char * s);

#ifdef __cplusplus
}
#endif

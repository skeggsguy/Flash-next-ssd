#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// commit one tiny command buffer on a queue of its own, so an idle GPU does not power down (ggml-metal-awake.m)
void    ggml_metal_keep_awake(void);
// pings committed so far
int64_t ggml_metal_keep_awake_count(void);

#ifdef __cplusplus
}
#endif

// Keep the GPU awake while a floor waits on a trip (study patch, C2: cheaper floor stops).
//
// When the GPU has had nothing to do for ~1.75-2 ms, the next command buffer takes ~450-650 us from commit to
// its first instruction instead of ~90 us: the GPU has powered down and must come back up. A floor that waits
// on a trip from the stacks often leaves it idle that long. Measured on the M5 Pro with a standalone Metal
// program (no model): launch after a 1750 us gap 94 us, after 2000 us 483 us, after 3000 us 657 us; with a
// tiny command buffer committed every 1 ms during the gap, 94-124 us out to 5 ms gaps.
//
// This is that tiny command buffer: one blit that fills 4 bytes of a 256-byte private buffer of its own, on a
// queue of its own. It touches no tensor and no buffer of the model, waits for nothing and is waited for by
// nothing, and the backend's GGML_METAL_CBLOG does not log it, so it cannot change a word. The book manager
// (src/llama-moe-stream-remap.cpp) calls it through the proc address "ggml_backend_metal_keep_awake".

#import "ggml-metal-awake.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>

static id<MTLCommandQueue> g_awake_queue = nil;
static id<MTLBuffer>       g_awake_buf   = nil;
static int64_t             g_awake_n     = 0; // pings committed, for tests

void ggml_metal_keep_awake(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        // the device ggml-metal itself uses (ggml-metal-device.m takes the system default device)
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev != nil) {
            g_awake_queue = [dev newCommandQueue];
            g_awake_buf   = [dev newBufferWithLength:256 options:MTLResourceStorageModePrivate];
        }
    });
    if (g_awake_queue == nil || g_awake_buf == nil) {
        return;
    }
    @autoreleasepool {
        id<MTLCommandBuffer>      cb = [g_awake_queue commandBufferWithUnretainedReferences];
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        [be fillBuffer:g_awake_buf range:NSMakeRange(0, 4) value:0];
        [be endEncoding];
        [cb commit];
    }
    __atomic_fetch_add(&g_awake_n, 1, __ATOMIC_RELAXED);
}

int64_t ggml_metal_keep_awake_count(void) {
    return __atomic_load_n(&g_awake_n, __ATOMIC_RELAXED);
}

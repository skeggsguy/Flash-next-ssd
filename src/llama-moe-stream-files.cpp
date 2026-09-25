#include "llama-moe-stream.h"
#include "llama-moe-stream-impl.h"

#include "llama-impl.h"

#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// The alt copy's shard list, from its first shard. This has to agree with how the split loader
// resolves -m's siblings (llama_get_list_splits in llama-model-loader.cpp): same prefix helper,
// same path format, same count. A single-file model has no suffix to strip and its alt list is
// just the one path.
std::vector<std::string> llama_moe_stream_alt_paths(const std::string & first, size_t n_split) {
    if (n_split <= 1) {
        return { first };
    }

    std::vector<char> buf(llama_path_max(), 0);

    const int32_t n = llama_split_prefix(buf.data(), buf.size(), first.c_str(), 0, (int32_t) n_split);
    if (n == 0) {
        throw std::runtime_error(format(
                "--moe-stream-alt-path must be the FIRST shard of a %zu-way split, named "
                "<prefix>-%05d-of-%05d.gguf; got %s", n_split, 1, (int) n_split, first.c_str()));
    }
    const std::string prefix(buf.data(), n);

    std::vector<std::string> out;
    for (size_t i = 0; i < n_split; i++) {
        const int32_t w = llama_split_path(buf.data(), buf.size(), prefix.c_str(), (int32_t) i, (int32_t) n_split);
        if (w == 0) {
            throw std::runtime_error(format("cannot build alt shard %zu of %zu from %s", i + 1, n_split, prefix.c_str()));
        }
        out.emplace_back(buf.data(), w);
    }
    return out;
}

void llama_moe_stream::open_files(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        if (path.empty()) {
            throw std::runtime_error("MoE expert streaming requires a file-based model (not a stream/file descriptor)");
        }
    }

    std::vector<std::string> alt_paths;
    if (!alt_path.empty()) {
        alt_paths = llama_moe_stream_alt_paths(alt_path, paths.size());
    }

    auto open_all = [&](bool direct) {
        files.clear();
        for (const auto & path : paths) {
            files.emplace_back(new llama_file(path.c_str(), "rb", direct));
        }
        files_alt.clear();
        for (const auto & path : alt_paths) {
            files_alt.emplace_back(new llama_file(path.c_str(), "rb", direct));
        }
    };

    open_all(use_direct_io);

    // wt.offs is carried over from the model's own shards unchanged, which is only correct if the
    // alt shards are byte-identical copies. Sizes are the cheap half of that check and they catch
    // the mistake that actually happens - pointing at a different edition, or at a half-copied
    // file. A mismatch is a hard failure: reading an expert from the wrong offset would not crash,
    // it would quietly produce different words.
    for (size_t i = 0; i < files_alt.size(); i++) {
        const size_t want = files[i]->size();
        const size_t got  = files_alt[i]->size();
        if (want != got) {
            throw std::runtime_error(format(
                    "MoE expert streaming: alt shard %s is %zu bytes but its twin %s is %zu; the "
                    "two runners must hold byte-identical copies",
                    alt_paths[i].c_str(), got, paths[i].c_str(), want));
        }
    }

    if (!files_alt.empty()) {
        LLAMA_LOG_WARN("%s: MoE expert streaming uses two runners: expert ids below %d%% of a layer "
                       "from %s, the rest from %s\n",
                __func__, alt_split, paths[0].c_str(), alt_paths[0].c_str());
    }

    // fall back to buffered when O_DIRECT is unusable: either the open did not honor it (macOS,
    // Windows, unsupported filesystems), or it opened but a probe read fails (some network/overlay
    // filesystems accept the flag then reject aligned reads). reopening is needed because O_DIRECT
    // is a property of the fd. done here, single-threaded, before any worker starts.
    if (use_direct_io) {
        // both sets: the two runners can be different filesystems, and direct I/O working on one
        // says nothing about the other. Either both bypass the page cache or neither does - a
        // half-direct run would have two different read costs in one measurement.
        bool ok = !files.empty() && files.front()->has_direct_io() &&
                  (files_alt.empty() || files_alt.front()->has_direct_io());
        if (ok) {
            uint8_t * probe = (uint8_t *) moe_aligned_alloc(MOE_STREAM_DIRECT_ALIGN);
            GGML_ASSERT(probe != nullptr);
            ok = llama_moe_stream_pread(*files.front(), probe, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true) != nullptr;
            if (ok && !files_alt.empty()) {
                ok = llama_moe_stream_pread(*files_alt.front(), probe, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true) != nullptr;
            }
            moe_aligned_free(probe);
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: %s not usable, falling back to buffered streaming reads\n",
                    __func__, MOE_STREAM_DIRECT_NAME);
            use_direct_io = false;
            open_all(false);
        }
    }

    // Named rather than hardcoded: macOS reaches this through F_NOCACHE, not O_DIRECT, and a log
    // line claiming the wrong mechanism is worse than none - it cannot be told from a real bypass.
    if (use_direct_io) {
        LLAMA_LOG_WARN("%s: MoE expert streaming uses %s (page cache bypassed)\n",
                __func__, MOE_STREAM_DIRECT_NAME);
    }

    no_zerocopy = std::getenv("LLAMA_MOE_STREAM_NO_ZEROCOPY") != nullptr;

    // whether reads land in the cache slot directly or stage through a bounce buffer - worth a line
    // because it silently changes the per-miss cost and depends on the backend's buffer type
    for (const auto & sl : layers) {
        if (sl && !sl->weights.empty()) {
            const bool zc = !use_direct_io && !no_zerocopy &&
                    ggml_backend_tensor_get_host_ptr(sl->weights[0].cache) != nullptr;
            LLAMA_LOG_WARN("%s: MoE expert streaming reads %s\n",
                    __func__, zc ? "directly into the expert cache (no staging copy)"
                                 : "through a staging buffer");
            break;
        }
    }


    // one token drives ~one remap per streamed layer, so decaying every 64 tokens is
    //   64 * n_streamed_layers remap calls (computed once here, off the hot path)
    int64_t n_streamed = 0;
    for (const auto & sl : layers) {
        n_streamed += sl != nullptr;
    }
    // Eviction is hotness-with-decay: every hot_decay_interval remap calls, all counters halve, so
    // recent routing outweighs old routing. The 64-token constant has never been measured - and the
    // miss RATE is now the lever that matters, because the read path itself is close to the drive's
    // practical limit. LLAMA_MOE_STREAM_HOT_DECAY sweeps it without a rebuild.
    int64_t decay_tokens = MOE_STREAM_HOT_DECAY_TOKENS;
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_HOT_DECAY")) {
        decay_tokens = std::max<int64_t>(0, std::atoll(s));   // 0 = never decay (pure cumulative)
    }
    hot_decay_interval = decay_tokens * n_streamed;

}

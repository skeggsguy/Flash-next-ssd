# Notes for coding agents

This is Flash-Next SSD, a llama.cpp fork that streams the experts of Qwen3.8-Flash-Next from SSD on
Apple Silicon. Read [README.md](README.md) and [docs/flashnext/](docs/flashnext/README.md) first.

## Where the fork's code lives

- Expert streaming and the expert cache: `src/llama-moe-stream*.cpp`, `src/llama-graph-moe-*.cpp`.
- The reading room: `src/llama-moe-room*.cpp`.
- Union attention (the sparse-attention pick lists): `src/llama-qsa-picks.cpp`.
- The draft head's recording and token list: `src/llama-mtp-record.cpp`, `src/llama-mtp-vocab.cpp`.
- The n-gram shelf: `src/llama-ple-shelf*.cpp`.
- The chat window: `tools/server/server-lane.cpp`.
- Command-line options: `common/arg.cpp`. Every flag in the guides must exist there.
- Tests: `tests/test-moe-*.cpp`, `tests/test-qsa-*.cpp`, `tests/test-*mtp*.cpp`, `tests/test-server-*.cpp`.

Everything else follows upstream llama.cpp; keep changes to upstream files small so merges stay easy.

## Working rules

- Build with `cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`,
  and run the tests with `ctest --test-dir build`. Check Metal operations with
  `./build/bin/test-backend-ops -b MTL0`.
- Run one model process at a time: two servers on the same Mac void each other's timings and can
  push it into swap.
- Never commit model files (`*.gguf`).
- A speed change that should not change the output must give byte-identical answers at
  temperature 0 on fixed prompts; one that does change the output needs a perplexity check.
- Keep `LICENSE` and `licenses/` as they are.

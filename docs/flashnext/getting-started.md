# Getting started

This page walks through the [quick start](../../README.md#quick-start) step by step: what you need,
where the files go, the optional second drive, and how to tell that the server started well. Every
option is explained in [Settings](settings.md).

## What you need

| | | Why |
|---|---|---|
| **Mac** | Apple Silicon with 64 GB of unified memory | The settings here were measured on a Mac mini M5 Pro, 64 GB. For 48 GB, see [Memory sizing](memory-sizing.md). |
| **Disk** | About 115 GB free on a fast internal SSD | Experts are read from the SSD for every token. A slow or USB drive will be far slower. |
| **Second SSD** (optional) | Another ~112 GB on a fast external SSD | Here a Samsung 990 PRO in a Thunderbolt 5 enclosure: two drives give +15% writing. |
| **Tools** | Xcode command line tools and CMake | `xcode-select --install`, `brew install cmake` |

The fast paths are Metal only. Nothing in this fork has been tuned on CUDA, Vulkan or the CPU.

## 1. Build

```sh
git clone https://github.com/skeggsguy/Flash-next-ssd
cd Flash-next-ssd
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The server is `build/bin/llama-server`. To check the Metal operations, including the ones this
fork adds, run `./build/bin/test-backend-ops -b MTL0`.

## 2. Download

The model is Unsloth's **UD-Q4_K_XL** quant of Qwen3.8-Flash-Next: 4 files, 111.3 GB. The draft
head for speculative decoding is npanj's shared MTP head (1.9 GB).

```sh
D=~/models/flashnext && mkdir -p $D
for i in 1 2 3 4; do
  curl -fL --retry 5 -C - -o $D/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000$i-of-00004.gguf \
    https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000$i-of-00004.gguf
done
curl -fL --retry 5 -C - -o $D/mtp-shared-Q4_K_M.gguf \
  https://huggingface.co/nitinpanj/qwen38-flash-next-v3/resolve/main/MTP/mtp-shared-Q4_K_M.gguf
```

`-C -` resumes a broken download. Keep the four model files together: `-m` points at the first,
and the others are found beside it. Download one file at a time; parallel downloads of files this
size gain nothing and make a failure harder to spot.

The sha256 of each file the study used (the same as Hugging Face's LFS hash):

| File | Size (bytes) | sha256 |
|---|---|---|
| `...-00001-of-00004.gguf` | 10,946,624 | `4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082` |
| `...-00002-of-00004.gguf` | 49,859,583,136 | `3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9` |
| `...-00003-of-00004.gguf` | 49,376,141,504 | `56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3` |
| `...-00004-of-00004.gguf` | 12,087,983,520 | `753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a` |
| `mtp-shared-Q4_K_M.gguf` | 1,907,151,936 | `2b468e0a490c7a8e46c66e6e9feb8c02a9c5d8f2aed276a951ef10f8c227f12c` |

Check them with `shasum -a 256 ~/models/flashnext/*.gguf`.

Other quants of the same model work too, for example npanj's `Q4_0-Q8out-v3` (95.5 GiB, see
[his guide](../qwen38-flash-next-v3.md)), but every number on these pages is for UD-Q4_K_XL.

## 3. The second drive (optional)

Copy the four model files to the second SSD, byte for byte:

```sh
mkdir -p /Volumes/<ssd>/flashnext
cp ~/models/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-*.gguf /Volumes/<ssd>/flashnext/
```

Then add to the server command:

```sh
--moe-stream-alt-path /Volumes/<ssd>/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
--moe-stream-alt-split 53
```

The engine refuses to start if a copy's size differs from its twin. The split is a percentage of
expert ids served from the `-m` copy; 53 matches a 6.5 GB/s internal drive beside a 5.7 GB/s
external one. For other drives, set it to the faster drive's share of the two drives' combined read
speed ([Settings](settings.md#two-drives)).

## 4. The GPU's wired-memory limit

macOS caps how much memory the GPU may lock in RAM ("wire"). The default cap is 75% of RAM, 48 GiB
on a 64 GB Mac, and this setup wires about 42 GiB, so the margin is thin. The study ran with it raised:

```sh
sudo sysctl iogpu.wired_limit_mb=59392
```

59,392 MiB is 58 GiB. It is a ceiling, not a reservation: past it, Metal reports an allocation
error that the server survives, instead of macOS locking up. It resets on every reboot. To set it
at every boot, install a LaunchDaemon (as root) at
`/Library/LaunchDaemons/local.iogpu.wiredlimit.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>local.iogpu.wiredlimit</string>
	<key>ProgramArguments</key>
	<array>
		<string>/usr/sbin/sysctl</string>
		<string>-w</string>
		<string>iogpu.wired_limit_mb=59392</string>
	</array>
	<key>RunAtLoad</key>
	<true/>
</dict>
</plist>
```

and load it with `sudo launchctl bootstrap system /Library/LaunchDaemons/local.iogpu.wiredlimit.plist`.

## 5. Run

```sh
LLAMA_SPEC_ADAPTIVE_RATE=1 LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB=4096 \
GGML_METAL_RESIDENCY_KEEP_ALIVE_S=10000000 \
./build/bin/llama-server \
  -m ~/models/flashnext/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -ngl 99 \
  --moe-stream --moe-stream-cache 28 --moe-stream-io-threads 8 --moe-stream-direct \
  -md ~/models/flashnext/mtp-shared-Q4_K_M.gguf --spec-draft-ngl 99 \
  --spec-type draft-mtp-adaptive --spec-draft-n-max 5 --spec-draft-p-min 0.3 --spec-max-prompt 0 \
  -c 200000 -b 4096 -ub 4096 -cms 8192 -ctxcp 3 -np 1 -fa on \
  --cache-reuse 0 --cache-ram 0 --jinja --reasoning-format deepseek \
  --host 127.0.0.1 --port 8080
```

What each part is for, in one line each (details in [Settings](settings.md)):

| Part | Why |
|---|---|
| `LLAMA_SPEC_ADAPTIVE_RATE=1` | the draft head measures its own guessing depth |
| `LLAMA_MOE_STREAM_ALLOC_CHUNK_MIB=4096` | claim the expert cache in 4 GiB steps, so macOS is not rushed at load |
| `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=10000000` | keep the model wired while idle, instead of re-wiring ~41 GiB on the next request |
| `--moe-stream --moe-stream-cache 28` | stream experts from the SSD, with a 28 GiB expert cache |
| `--moe-stream-io-threads 8 --moe-stream-direct` | 8 parallel readers, past macOS's file cache |
| `-md ... --spec-type draft-mtp-adaptive ...` | the MTP draft head, up to 5 tokens ahead |
| `-c 200000 -np 1` | a 200K-token context for one conversation at a time |
| `-b 4096 -ub 4096` | big prompt batches, which the reading room needs |
| `-cms 8192 -ctxcp 3` | prompt-cache checkpoints every 8,192 tokens, at most 3 kept |
| `-fa on` | flash attention, required by this model's sparse attention when reading a prompt |
| `--cache-reuse 0 --cache-ram 0` | no extra prompt cache in RAM: on 64 GB that memory is the expert cache's |
| `--jinja --reasoning-format deepseek` | the model's chat template, with its thinking returned separately |

Then open <http://127.0.0.1:8080>, or point an OpenAI-compatible client at
`http://127.0.0.1:8080/v1`. `/v1/messages` serves Anthropic-style clients. Keep `--host 127.0.0.1`
unless you mean to share the server on your network.

To start this same command with a double-click, or by itself at every login, see
[Start it with one click](../../README.md#start-it-with-one-click) and
[Start at login](../../README.md#start-at-login-suggested) in the README.

## What a healthy start looks like

The server loads in a few seconds. Look for these lines in its log:

- `expert cache size = ... MiB (N slots per layer)`: the expert cache, as experts held per layer.
- `reading room: 1.44 GiB = 1.25 floors of look-ahead, each floor in 4 parts (...); desk ... GiB (... slots per floor)`:
  the reading room is on. ("Floor" and "desk" are the project's words for a layer and the expert cache.)
- `MoE expert streaming uses two runners: ...`: only with a second drive.
- `bookmark pin: on, 1 of the 3 copies stays at the start ...`: the pinned checkpoint.
- `qsa union: ...`: union attention.
- `chat window: on, ...`: only with `--chat-window`.

The first request is slow: the expert cache starts empty, so a 4-token prompt takes about 2 s and
a 17K-token one about 50 s. After that, expect roughly 15-20 tokens/s writing (faster on code and
tool calls, slower on prose) and 400-500 tokens/s reading a prompt.

If something is wrong:

- **It fails at load with a GPU allocation error.** The wired limit is too low for the cache. Check
  `sysctl iogpu.wired_limit_mb`, or use a smaller `--moe-stream-cache`.
- **It is much slower than the numbers above.** Check swap with `sysctl vm.swapusage`. If swap is in
  use, the cache or context is too big for what else is running: see [Memory sizing](memory-sizing.md).
  Also check that the model is on the internal SSD and that no sync client (Dropbox, iCloud) is
  watching its folder.
- **Writing got slow mid-conversation and stayed slow.** `--spec-max-prompt` was set to a number. Set
  it to 0 ([Settings](settings.md#the-mtp-draft-head)).

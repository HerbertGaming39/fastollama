# fastollama

<div align="center">
  <img src="assets/icon-1024.png" width="160" alt="fastollama">
</div>

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/HerbertGaming39/fastollama?style=flat-square&color=E11D2E)](https://github.com/HerbertGaming39/fastollama/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/HerbertGaming39/fastollama?style=flat-square)](https://github.com/HerbertGaming39/fastollama/network/members)
[![Issues](https://img.shields.io/github/issues/HerbertGaming39/fastollama?style=flat-square)](https://github.com/HerbertGaming39/fastollama/issues)
[![License: MIT](https://img.shields.io/badge/license-MIT-E11D2E?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Vulkan-464646?style=flat-square)]()
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?style=flat-square&logo=cplusplus&logoColor=white)

</div>

---

**fastollama is a tuned llama.cpp runner that makes large models run fast on GPUs that shouldn't be able to hold them.**

It is built on the same engine as Ollama — but wires up the things Ollama doesn't:

- a **VRAM governor** that computes exactly what a model + context + KV cache needs *before* starting, and never lets the engine eat your desktop's memory
- **full-GPU mode** — the single biggest speed lever on any card: a model that fits entirely in VRAM is a different speed class than one that spills
- **MTP speculative decoding** (+30–80% on models that ship an MTP head, at zero quality cost)
- **quantized KV cache** so huge contexts (256K+) actually fit
- a **smart tensor offloader** that knows attention tensors must never stream from RAM, but FFN/expert tensors can
- **`fastollama pull`** for one-command model downloads with resume, disk guards and multi-shard support

No daemon, no account, no daemon bloat: one binary, one `settings.txt`, OpenAI-compatible API out of the box.

```bash
bin/fastollama serve          # OpenAI-compatible API on http://127.0.0.1:8080
bin/fastollama chat           # terminal chat
bin/fastollama plan           # show the VRAM plan for the current settings
bin/fastollama pull qwen3.8-27b-iq2s --set
```

Works with **any model llama.cpp supports** — Qwen, Llama, Gemma, Mistral, Phi, DeepSeek and more ship as pre-set pull aliases.

---

## Results (measured on the reference rig: RX 9070 XT 16 GB, Ryzen 7 9700X, 32 GB RAM)

Full **262,144-token context** (native max), VRAM capped at 15 GB so the desktop survives, MTP on:

| model | context | gen t/s | prompt t/s | VRAM (engine) | mode |
|---|---|---|---|---|---|
| **Qwen3.8-27B-UD-IQ2_S** | 262144 | **71–86** | ~1000 | 14.4 GB | full_gpu |
| Qwen3.8-27B-UD-IQ1_M (max speed) | 262144 | **90–92** | 184 | 14.5 GB | full_gpu |
| **Qwen3-Next-80B-A3B-UD-IQ2_XXS** | 262144 | **33** (8T) | ~800 | 13.8 GB | expert split (18/48 expert layers on GPU) |
| Qwen3.8-Flash-Next-UD-IQ1_S (177B-class Qwen4-preview arch) | 262144 | **4.0** | 119 | **14.1 GB** | expert split (44/48 experts in RAM) |
| Qwen3-30B-A3B-UD-Q4_K_XL (MoE) | 262144 | 30.5 | 105 | 13.5 GB | expert split |
| Qwen3.8-27B-UD-Q4_K_XL (max quality) | 262144 | 6.9 | 27 | 13.4 GB | smart split |

An 80-billion-parameter model at full 256K context on a 16 GB card — while the desktop stays usable — that is the whole point of fastollama.

### The one rule that explains every number

> **Any CPU-resident tensor — even FFN weights — collapses a dense model to single-digit t/s. A fully-GPU quant is a different speed class, not an increment.**

A dense model reads all of its weights for every generated token. The 27B IQ2_S (8.4 GB fully on GPU) runs 86 t/s ≈ 715 GB/s ≈ the card's memory bus — bandwidth-bound, i.e. maximally optimized. The same model 2.5 GB bigger (IQ3_XXS) spills to RAM and drops to 4.5 t/s. MoE flips the math: only ~3B params activate per token, so streaming experts from system RAM is cheap — the 80B at 262K does 33 t/s with 30 of 48 expert layers in RAM (30 layers × ~14 MB/token ≈ 420 MB/token over ~65 GB/s DDR5 = the floor; measured 8T=33, 12T=32, 16T=29 — SMT hurts MoE CPU math).

The governor exists to get you into the right class automatically: `model = auto` walks the quant ladder and picks the largest quant that fits 100% on GPU, and `full_gpu = 1` falls back to a smart split (never an OOM) when nothing fits.

### Engine knobs, measured (not guessed)

| experiment | result |
|---|---|
| MTP draft_max = 1 / **2** / 4 / 6 | 6.9 → off, **71–86**, 35, 32 t/s → **2 is optimal** |
| ubatch 512 / **1024** / 2048 | 848 / **1003** / 979 t/s prompt → **1024 shipped** |
| parallel slots 1 / 2 | 66–80 t/s vs 31+33=64 combined → **1 for solo use** |
| HIP/ROCm backend vs Vulkan | 0.6–4.75 vs 71–86 t/s on RDNA4 → **Vulkan default** |
| CPU repack kernels, disk streaming (AirLLM-style) | measured slower → rejected |

---

## What fastollama adds over plain llama.cpp / Ollama

1. **VRAM safety governor** — parses the GGUF (layers, KV heads, experts, quant types), computes exact KV-cache + compute-buffer sizes for your context, and enforces a hard cap (`vram_limit_gb`). The desktop always keeps its memory.
2. **VRAM watchdog** — a live guard while the server runs: if total VRAM use ever crosses the emergency line (driver peak, leak, another app grabbing memory), the server is stopped cleanly instead of freezing the session.
3. **`full_gpu = 1`** — whole model + KV + draft model on the GPU, zero RAM streaming, with automatic fallback to a smart split if it can't fit.
4. **`model = auto`** — picks the largest downloaded quant that fits fully on GPU.
5. **Hybrid-arch-aware planner** — counts real attention/KV layers from tensor names instead of assuming every layer carries KV (avoids 10× KV overestimates on linear-attention models).
6. **Smart offload (FFN-only)** — when a split is unavoidable, only `ffn_`/expert tensors go to CPU; per-token sequential tensors (attention, SSM/gating) stay on GPU.
7. **MTP speculative decoding, wired** — `spec_type = draft-mtp`, arch-mismatch auto-skip, multi-shard drafters supported.
8. **Quantized KV at any context** — q4_0 KV turns a ~40 GB f16 requirement at 256K into ~1.6 GB. This is what makes max context fit at all.
9. **`fastollama pull`** — verified aliases (every URL checked live), resume, multi-shard models, disk-space guard before starting, `--set` writes the model into settings.txt automatically.
10. **Sane defaults, tuned on hardware** — `ubatch = 1024`, `draft_max = 2`, `cache_reuse = 256`, `parallel = 1`.

---

## Pull a model

```bash
bin/fastollama pull qwen3.8-27b-iq2s --set   # 27B dense, best balance (8.4 GB)
bin/fastollama pull qwen3.8-27b-iq1m --set   # 27B max speed tier
bin/fastollama pull qwen3-next-80b   --set   # 80B MoE (26.2 GB)
bin/fastollama pull flash-next-iq1s   --set  # 177B-class Qwen4-preview arch (74 GB, 3 shards)
bin/fastollama pull llama3.1-8b      --set   # non-Qwen works too
bin/fastollama pull gemma3-27b       --set
bin/fastollama pull mistral-nemo     --set
# also: qwen3-8b, qwen3-30b, phi4, deepseek-r1-8b, ... (run pull with a bad name to list all)
# or pass any direct GGUF URL: bin/fastollama pull https://.../Model-Q4_K_M.gguf
# --set wires the download into settings.txt; downloads resume if interrupted;
# refuses to start when disk space is insufficient (checked per shard)
```

Multi-shard GGUFs (like Flash-Next) download all shards automatically; llama.cpp loads the model from shard 1 and finds the rest.

---

## Huge models: the honest playbook

| size class | what fits | expected speed (16 GB card) |
|---|---|---|
| ≤ 8.5 GB weights | everything on GPU | 70–95 t/s |
| 9–15 GB | smart FFN split | 4–10 t/s (dense) — avoid |
| 15–30 GB (MoE, ~3B active) | expert split into system RAM | **20–40 t/s** |
| ≥ 40 GB | heavy streaming | slow — needs a bigger plan |

Rules of thumb any rig can apply: get the *non-expert core + KV* of an MoE into VRAM and let experts stream (that's what MoE is for); for dense models pick the biggest quant that fits **entirely** in VRAM; use q4_0/q8_0 KV to buy context window with VRAM.

---

## Build from source

Requirements: a C++17 compiler, CMake + Ninja, Vulkan SDK (`glslc`), ~15 min.

```bash
# 1. Get llama.cpp at the tested commit
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout 1af554f
git apply ../patches/0001-vulkan-spirv-headers-include-fix.patch   # RDNA4 build fix

# 2. Build the Vulkan backend
cmake -B build-vk -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk --target llama-server llama-cli llama-quantize -j$(nproc)

# 3. Build the fastollama governor
cd .. && g++ -O2 -std=c++17 -o bin/fastollama src/fastollama.cpp

# 4. Pull a model and serve
./bin/fastollama pull qwen3.8-27b-iq2s --set
./bin/fastollama serve
```

The governor is one portable C++ file, no dependencies beyond libstdc++. Verified on Mesa RADV with AMD RDNA3/RDNA4; any Vulkan GPU llama.cpp supports should work — check `bin/fastollama plan` output on yours.

---

## settings.txt reference (the keys that matter)

| key | default | meaning |
|---|---|---|
| `model` | `auto` | name in `models/`, full path, or `auto` (largest quant that fits) |
| `full_gpu` | `1` | whole model on GPU, auto-fallback if it can't fit |
| `context` | `262144` | tokens of context (pair with `kv_cache`) |
| `kv_cache` | `q4_0` | KV quant — f16 is ~25× bigger at 256K |
| `vram_limit_gb` | `15` | hard engine cap — the desktop-safety promise |
| `vram_fixed_gpu_gb` | `0` (auto) | measured non-expert VRAM floor (attn core + KV + buffers). Set it after one calibration run with all experts offloaded — the planner then spends only what is really left |
| `vram_free_min_gb` | `1.3` | always-kept-free margin |
| `vram_guard` | `1` | live watchdog; `vram_emergency_pct` = kill line (default **0.95** — fires *before* the desktop dies) |
| `spec_type` | `draft-mtp` | MTP speculative decoding (`off` to disable) |
| `draft_max` | `2` | tuned on hardware (4/6 measured slower) |
| `ubatch` | `1024` | prompt-processing micro-batch |
| `cache_reuse` | `256` | reuse cached prompt tokens after context shift |
| `parallel` | `1` | concurrent slots (2 halves per-stream speed) |
| `backend` | `vulkan` | `hip`/`cpu` selectable; Vulkan measured fastest on AMD |

---

## Known limits (physics, not defeatism)

- Fully-GPU dense models run at the card's memory-bandwidth limit; that is the ceiling and fastollama gets you to it. Going faster means a smaller quant or an MoE model.
- Filling the whole context window slows decode (attention cost grows with fill) — short chats sit at the top of the speed range, a packed 256K window at the bottom.
- These are per-stream numbers; multiple parallel slots divide the same bandwidth.

*Benchmarks: 2026-09-20/22, Mesa RADV GFX1201, Vulkan, kernel 6.x. Every claim on this page was measured, and the losing experiments are listed too — see `logs/`.*

---

<div align="center">

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HerbertGaming39/fastollama&type=Date)](https://star-history.com/#HerbertGaming39/fastollama&Date)

</div>

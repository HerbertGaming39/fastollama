# fastollama

A tuned llama.cpp runner for the **AMD RX 9070 XT (gfx1201 / RDNA4) + Ryzen 7 9700X + 32 GB RAM** box, built to run the newest Qwen models at their **maximum native context** with every optimization that actually measures faster on this hardware.

Everything is compiled from source (llama.cpp fork, pinned at commit `1af554f`):
- `llama.cpp/build-vk/` — **Vulkan backend (default, fastest on RDNA4)**
- `llama.cpp/build-hip/` — ROCm/HIP backend (built, measured slower — see benchmarks)
- `bin/fastollama` — the CLI wrapper with the VRAM safety governor

```bash
cd ~/fastollama
bin/fastollama serve          # OpenAI-compatible API on http://127.0.0.1:8080
bin/fastollama chat           # terminal chat
bin/fastollama plan           # show the VRAM plan for the current settings
bin/fastollama pull qwen3.8-27b
```

All tuning lives in **`settings.txt`** — no CLI flags to remember.

---

## Headline results (all measured on this PC)

Full **262,144-token context** (the model's native max — no YaRN tricks), VRAM hard-capped so GNOME survives, MTP speculative decoding on:

| model | context | gen t/s | prompt t/s | VRAM (engine) | mode |
|---|---|---|---|---|---|
| **Qwen3.8-27B-UD-IQ2_S** ← default | 262144 | **71–86** | ~1000 | 14.4 GB | full_gpu |
| Qwen3.8-27B-UD-IQ1_M (max speed) | 262144 | **90–92** | 184 | 14.5 GB | full_gpu |
| Qwen3.8-27B-UD-IQ2_XXS | 262144 | 86 | 179 | 14.4 GB | full_gpu |
| Qwen3.8-27B-UD-Q4_K_XL (max quality) | 262144 | 6.9 | 27 | 13.4 GB | smart split |
| **Qwen3-Next-80B-A3B-UD-IQ2_XXS** | 262144 | **36–38** | 808 | 14.5 GB | full_gpu |
| Qwen3-30B-A3B-UD-Q4_K_XL (MoE) | 262144 | 30.5 | 105 | 13.5 GB | expert split |
| Qwen3-30B-A3B-UD-Q4_K_XL (MoE) | 40960 | 40.3 | 154 | 13.9 GB | expert split |

Quality verification on the speed configs (not just "it compiled"): IQ1_M and IQ2_S both answer 17×23=391 ✓, Berlin Wall 1989 ✓, accurate Apollo 11 summaries ✓. These are unsloth *dynamic* quants built with the official imatrix — nothing like the garbage 1-bit quants of years past.

### The one rule that explains every number

> **Any CPU-resident tensor — even FFN weights — collapses a dense model to single-digit t/s.** A fully-GPU quant is a different speed class, not an increment.

The 27B IQ3_XXS (10.9 GB) is a *better* quant than IQ2_S (8.4 GB), but its extra 2.5 GB spills to RAM: 4.5 t/s vs 86. A dense model reads **all** its weights for **every** token:

- full-GPU 8.4 GB × 86 t/s ≈ **715 GB/s** ≈ the 9070 XT's rated bus (644–730 GB/s) → generation is memory-bandwidth-bound, i.e. maximally optimized
- with ~6 GB on CPU: ~26 GB/s effective CPU GEMV → ~5 t/s, no software fixes that
- MoE changes the math: only ~3B params activate per token, so streaming expert weights to RAM is cheap — hence 30–40 t/s on the 30B-A3B

### Engine knobs, measured (not guessed)

| experiment | result |
|---|---|
| MTP draft_max = 1 / **2** / 4 / 6 | 6.9→ off, **71–86**, 35, 32 t/s → **2 is optimal** (acceptance ~0.68 full-GPU) |
| ubatch 512 / **1024** / 2048 | 848 / **1003** / 979 t/s pp → **1024 shipped** (+18%) |
| parallel slots 1 / 2 | 66–80 t/s vs 31+33=64 combined → **1 for solo use** |
| HIP/ROCm backend | 0.6–4.75 t/s (RDNA4 rocBLAS untuned) → **Vulkan stays** |
| CPU repack kernels (q4_K_8x8) | 0% change on Zen 5 (native kernels already optimal) |
| AirLLM-style disk streaming | NVMe 3–7 GB/s < our RAM pipeline → rejected |
| Colibri-style MoE disk offload | validated for MoE only; our expert split already does it |

---

## What fastollama adds over plain llama.cpp / Ollama

1. **VRAM safety governor** — reads the GGUF, computes exact KV-cache + compute-buffer sizes, and enforces a hard cap (`vram_limit_gb = 15`). GNOME never dies. The desktop keeps its ~1.5 GB.
2. **`full_gpu = 1`** — whole model + KV + MTP draft on the GPU, zero RAM streaming. Auto-falls-back to smart split if the quant doesn't fit. This is the 86 t/s switch.
3. **`model = auto`** — walks the quant ladder and picks the *largest quant that fits 100% on GPU* (quality of a fully-GPU small quant beats a bigger spilling one — see the rule above).
4. **Hybrid-arch-aware planner** — Qwen3.8 is 48 linear-attention + 17 full-attention layers; the governor counts real attention/KV layers from tensor names instead of assuming every layer carries KV (a 10× KV overestimate otherwise).
5. **Smart offload (FFN-only)** — when a split is unavoidable, offloads *only* `ffn_` tensors of whole layers and keeps every attention/SSM/gating tensor on GPU; sequential per-token tensors on CPU serialize the entire graph (that bug cost 20× once).
6. **MTP speculative decoding, wired** — `spec_type = draft-mtp` + the Qwen3.8-27B MTP sidecar, with arch-mismatch auto-skip (attaching the wrong draft otherwise aborts the server).
7. **q4_0 KV cache at 262K** — ~1.6 GB instead of ~40 GB f16. This is what makes max context fit at all; stock Ollama would crash or silently shrink the window.
8. **Sane defaults, tuned on this machine** — `ubatch = 1024`, `draft_max = 2`, `threads = 16`, `parallel = 1`.

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

# 4. Drop a GGUF into models/ (see below), then:
./bin/fastollama serve
```

Tested on: Mesa RADV, 9070 XT (GFX1201), kernel 6.x, Arch-style Linux. The governor is ~1 file of portable C++ — no dependencies beyond libstdc++.

---

## Models on disk & how to get more

| file | size | note |
|---|---|---|
| `Qwen3.8-27B-UD-IQ2_S.gguf` | 8.4 GB | **default** — best speed/quality balance |
| `Qwen3.8-27B-UD-IQ1_M.gguf` | 6.7 GB | max speed tier (~92 t/s) |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | 7.3 GB | between the two |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17.6 GB | max quality tier (streams, ~7 t/s) |
| `Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf` | 26.2 GB | **the 80B** — 36–38 t/s @ 262K |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | 1.4 GB | MTP draft head (required for spec decoding) |

```bash
bin/fastollama pull qwen3.8-27b          # Q4_K_XL quality tier
bin/fastollama pull qwen3.8-27b-mtp      # MTP draft
# or any direct URL:
bin/fastollama pull https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ2_S.gguf
```

**Qwen3.8 family facts:** there is no "Qwen3.8-72B" — the dense family is 27B (256K native, vision, reasoning, MTP head) and Flash-Next (176B MoE, cannot fit this PC). The 80B-class 256K model is **Qwen3-Next-80B-A3B**; see below.

---

## Huge models: the honest playbook

| size class | what fits | expected speed here |
|---|---|---|
| ≤ 8.5 GB weights | everything on GPU | 70–95 t/s |
| 9–15 GB | smart FFN split | 4–10 t/s (dense) — avoid |
| 15–30 GB (MoE, ~3B active) | expert split into 32 GB RAM | **20–40 t/s** |
| ≥ 40 GB | disk streaming | 1–3 t/s — don't |

**Proven at the top of that table:** Qwen3-Next-80B-A3B (80B total, 3B active, 512 experts, hybrid linear attention) in UD-IQ2_XXS runs **36–38 t/s generating and 808 t/s prompt processing at the full 262144 context**, VRAM 14.5 GB, answers verified correct (math + literature). The governor keeps the non-expert core + KV on GPU and streams experts from RAM — exactly the workload MoE exists for. Model file: `Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf` (26.2 GB); switch via `model =` in settings.txt.

The 80B sweet spot: **UD-IQ2_XXS (26.2 GB)** — small enough that the non-expert core + KV fit in VRAM, experts stream from RAM at MoE rates.

---

## settings.txt reference (the keys that matter)

| key | default | meaning |
|---|---|---|
| `model` | `Qwen3.8-27B-UD-IQ2_S` | name in `models/` or full path; `auto` = ladder |
| `full_gpu` | `1` | whole model on GPU, auto-fallback if it can't fit |
| `context` | `262144` | max native context of Qwen3.8-27B |
| `kv_cache` | `q4_0` | KV quant (f16 = 40 GB at 262K — impossible; q8_0 = 2× q4) |
| `spec_type` | `draft-mtp` | MTP speculative decoding |
| `draft_max` | `2` | tuned on hardware (4/6 measured slower) |
| `ubatch` | `1024` | prompt-processing micro-batch (+18% pp vs 512) |
| `parallel` | `1` | concurrent slots (2 halves per-stream speed) |
| `threads` | `16` | CPU threads (offload + prompt processing) |
| `vram_limit_gb` | `15.0` | hard engine cap — the GNOME promise |
| `vram_free_min_gb` | `1.0` | always-kept-free margin |
| `backend` | `vulkan` | `hip` exists but measured 5–100× slower on RDNA4 |

---

## Known limits (physics, not defeatism)

- **86 t/s is the dense-27B ceiling on this card.** We are at ~715 GB/s effective — the bus. 100 t/s would need ~6.7 GB weights (IQ1_M hits 92 with that) or sub-6 GB (quality collapses).
- **262144 ctx costs ~1.6 GB of q4_0 KV + ~1.5 GB compute buffers.** Filling the whole 262K window slows decode somewhat (attention is quadratic in fill) — short chats run at the top of the range, a fully-loaded 262K window at the bottom.
- **MoE models are the throughput path**, not the density path: 36–38 t/s at full 262K on the 80B-A3B, 30.5 t/s on the 30B-A3B.

*Benchmarks: 2026-09-20/21, Mesa RADV GFX1201, Vulkan, kernel 6.x, temperatures ambient, 3-run medians where variance existed. Scripts and logs in `logs/`.*

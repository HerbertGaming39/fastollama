# Contributing to fastollama

The project is deliberately small: **one C++ file** (`src/fastollama.cpp`), one
settings file, one patch for the pinned llama.cpp. Keep changes that shape.

## Ground rules

1. **Every speed claim needs a number.** If a change claims to be faster,
   include measured t/s (same model, same prompt, same context) before AND
   after. Guesses don't get merged.
2. **Never break the VRAM cap.** The governor's whole job is keeping the
   desktop alive. If your change can exceed `vram_limit_gb` on any path,
   fix the estimate, not the cap.
3. **No model weights in git.** Ever. They live on HuggingFace.
4. **Offload changes must preserve sequential tensors.** Anything touched on
   every token (attention projections, SSM/gating) stays on the GPU unless
   the plan proves the split still wins — see the 3.4 t/s regression story
   in the README.

## Testing a change

```bash
g++ -O2 -std=c++17 -o bin/fastollama src/fastollama.cpp
./bin/fastollama plan          # dry run: shows the VRAM plan
./bin/fastollama serve         # then bench with your own client
```

`plan` is safe to run anytime and touches no VRAM — use it liberally.

## Commit style

Short subject line describing the *why*, benchmarks in the body when relevant.

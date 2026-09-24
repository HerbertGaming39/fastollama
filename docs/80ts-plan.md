# 80 t/s on Qwen3-Next-80B — the plan (after a reboot)

## Why a reboot
Today's ~10 heavy GPU load cycles left the RADV driver in a degraded state:
every GPU load >= ~10.5 GB now dies with
`radv/amdgpu: Not enough memory for command submission` -> `ErrorDeviceLost`,
while small models and CPU-only loads work fine. GTT and system RAM are NOT
exhausted — the driver's internal state is stuck. A reboot (or a dGPU reset)
clears it. Reset was declined; reboot whenever convenient.

## What is already done and verified (no redo needed)
- `llama.cpp/common/speculative.cpp` PATCHED: `FASTOLLAMA_DRAFT_CTX` env var
  shrinks the DRAFT model context (upstream forces draft ctx = target ctx,
  which costs the drafter's full KV at 262K = ~7.5 GB for a 0.6B model).
  Saved as `patches/0001-draft-ctx-override.patch` too.
- `build-vk` rebuilt WITH the patch (Vulkan SDK headers issue solved via
  SPIRV-Headers installed to ~/.local/spirv; configure with
  `-DSPIRV-Headers_DIR=$HOME/.local/spirv/share/cmake/SPIRV-Headers`).
- Drafter downloaded: `models/Qwen3-0.6B-Q8_0.gguf` (shares the 80B's exact
  tokenizer — the compatibility check passed in a live run).
- Full pipeline VALIDATED on CPU (port 8093 test): draft loads, small draft
  ctx works, synth acceptance works, generation coherent.
- fastollama wired end-to-end: `spec_type = draft` in settings.txt emits
  `--model-draft --spec-type draft-simple` + `FASTOLLAMA_DRAFT_CTX` from
  `draft_ctx` setting; `vram_limit_gb = 10.5` reserves drafter headroom.

## Exact commands after reboot
```
cd ~/fastollama
# real thing (the goal):
setsid nohup ./bin/fastollama serve > logs/serve-80b-draft.log 2>&1 < /dev/null &
# then: warmup + 3x bench, same bench prompt as always

# map acceptance -> speed (no real drafter needed):
for L in 2 3 4 5; do
  pkill -x llama-server; sleep 10
  cd llama.cpp
  FASTOLLAMA_DRAFT_CTX=8192 setsid nohup build-vk/bin/llama-server \
    -m ../models/Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf \
    --model-draft ../models/Qwen3-0.6B-Q8_0.gguf \
    --spec-type draft-simple --spec-draft-n-max 6 --spec-draft-n-min 1 \
    --spec-synth-len $L \
    -ot 'blk\.(11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47)\.ffn_.*_exps=CPU' \
    -ngl 49 -ngld 49 -t 6 --poll 100 -tb 6 -b 2048 -ub 1024 \
    -c 262144 -fa on -ctk q4_0 -ctv q4_0 --load-mode none --cache-reuse 256 \
    --host 127.0.0.1 --port 8080 -np 1 --jinja \
    > ../logs/synth-L$L.log 2>&1 < /dev/null &
  cd ..
  sleep 5; until curl -s --noproxy '*' -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:8080/health | grep -q 200; do sleep 3; done
  curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count from one to thirty."}],"max_tokens":150,"temperature":0.6}' -o /dev/null
  curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count from one to thirty."}],"max_tokens":150,"temperature":0.6}' -o /dev/null
  grep "eval time" logs/synth-L$L.log | tail -1
done
```

## The measured math (from this session)
Per-verify cost (k tokens verified together, K experts on GPU, 48-K on CPU):
- CPU expert streaming: 0.66 ms x k x (48-K)/48 x 48 ~= 0.66k ms
- GPU fixed per pass: ~15 ms -> F(k) ~= 15 + 0.66k
- At K=17 (drafter-reserved budget): F(1)=25.7, F(4)=27.7, F(6)=29.1
- Effective t/s = k / F(k): 1->39, 4->144, 6->206 t/s  (100% acceptance)
- 80 t/s needs effective 12.5 ms/token:
  * len 4: need ~68% acceptance
  * len 6: need ~56% acceptance
  * len 3: need ~76% acceptance

## Honest caveat (cost of each drafted token is NOT free)
Drafting runs the 0.6B on the GPU (~2-3 ms) and adds CPU work (~1-2 ms),
so real speedup ~= F(1) / (F(k)/a + c) with c ~= 3-4 ms:
  * len 4 @ 70%: 27.7/0.70 + 3.5 ~= 43 ms/tok -> ~23 t/s  (WORSE than baseline)
  * len 4 @ 90%: 27.7/0.90 + 3.5 ~= 34.3 -> ~29 t/s       (still worse!)
CONCLUSION: with 37 CPU expert layers the verify cost scales ~linearly with k,
so speculation CANNOT reach 80 here — the expert-streaming term dominates.
Speculation only pays when most experts are ON GPU (k-verify nearly free):
  * K=48 (all experts GPU, 23.5 GB — needs 24 GB card): F(k) ~= 15, len 4 @ 70%
    -> ~4/0.75 ~= 5.3 ms/tok -> ~188 t/s. That is where 80+ lives.
On THIS 16 GB card the honest ceiling remains ~35-40 t/s (config) and the
80 t/s goal requires either: 24 GB GPU, OR a smaller MoE that fits fully
(Qwen3-30B-A3B at low quant ~= 12-14 GB, could plausibly hit 60-100 t/s),
OR an engine that streams CPU experts faster than 0.66 ms/layer-token.

## Other projects (evaluated, not adopted)
- airllm (HF): inference-time layer streaming for huge models — same
  bandwidth physics as our -ot split, but Python/transformers (slower engine
  than llama.cpp); nothing to gain.
- Colibri/parallax-like layer-parallel runners: parallelize LAYERS across
  devices — helps multi-GPU rigs, not single-GPU + RAM streaming.
- PowerInfer / KTransformers: hot/cold expert placement (active experts on
  GPU, rest CPU) — WE ALREADY DO THIS via -ot + measured floor; KTransformers'
  extra trick is better CPU kernels (AMX) — we have AVX-512 already and
  llama.cpp's repack kernels compiled in.
- The fork's ngram spec: free +0.6 t/s on stories, more on code; kept wired.

## Settings state at save time (this session)
vram_limit_gb=10.5 (drafter headroom; restore to 14 after reboot if running
WITHOUT draft), spec_type=draft, draft_model=Qwen3-0.6B-Q8_0, draft_ctx=8192,
draft_max=6, threads=6, poll=100, est_safety_pct=1.02, vram_fixed_gpu_gb=4.25.

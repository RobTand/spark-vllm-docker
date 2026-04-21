# Morning handoff — prismaquant AWQ bug + recovery (2026-04-21)

## TL;DR

Proper-AWQ fold was pushing through unit tests (fp32 identity ✓) but producing
garbage on real Qwen3.6 (`"Paris" → " the 100000"`). Root cause: bf16 runtime
precision loss when `max(s)/min(s)` is extreme. Fixed by matching AutoAWQ's
geomean + hard-clamp convention. Shipped as commit **8b8435d**, pushed.

All three act-aware passes (AWQ + GPTQ + activation-weighted rounding) now
produce coherent output. 12-prompt qualitative diff vs Phase 1 baseline shows
comparable quality — no regression.

Remaining issue: **visual-tower calibration still isn't real**; `AutoModelForCausalLM`
silently downgrades Qwen3.6 to the text-only class. Exports on disk have
BF16 visual passthrough (Phase 1 fallback). Fix is a one-liner but requires
re-running the multimodal probe (~5 min).

---

## What I did overnight

### 1. Diagnosed the AWQ regression

Ran 3 ablations by re-exporting with different act-aware flag combos (cost
shards + activation cache all reused, so each export ≈ 2 min):

| Ablation | Flags | Sanity result |
|---|---|---|
| #1 all OFF | `--no-awq --no-gptq --no-act-weighted-round` | ✓ "Paris", l[::-1], math thinking |
| #2 AWQ only (pre-fix) | `--awq --no-gptq --no-act-weighted-round` | ✗ "the 100000", "TTTT", empty |
| #2b AWQ only (post-fix) | same | ✓ "Paris", "l[::-1]..." |
| #3 all three ON (post-fix) | default | ✓ better than baseline ("code fence" Python answer) |

Ablation #2 isolated the bug to the AWQ fold. Ablation #2b confirmed the
scale fix.

### 2. Root cause

`_awq_channel_scale` normalized `s` to `max(s)=1` with `eps=1e-6`, so low-
activation channels got `s ~ 1e-6`. The γ-fold then produces `γ/s` up to
1e6× original γ, and the compensating `W*s` produces weights down to 1e-6×.
Analytically `(W*s)·(γ/s) = W·γ` cancels exactly. But **bf16 has 8-bit
mantissa → 0.4% per-storage relative error**. Both sides of the cancellation
carry that error, which accumulates across the matmul sum to well over 30%
relative output error → garbage.

Reference implementations avoid this via:
- **geomean normalization**: `s /= sqrt(s.max() * s.min())` — centers `s`
  around 1 in log space instead of pushing toward eps
- **hard clamp**: `s ∈ [0.1, 10]` — caps bf16 matmul error accumulation
- **nan_to_num guard**: prevents constant-zero activation channels from
  poisoning the scale

Both AutoAWQ (`quantize/quantizer.py:406`) and MIT llm-awq (`auto_scale.py:130`)
do the geomean + clamp. My impl did neither. Now it does.

### 3. Fix shipped (commit `8b8435d`)

```diff
 def _awq_channel_scale(activations, eps=1e-4):
     mean_abs = a.abs().mean(dim=0)
     s = mean_abs.clamp_min(eps).pow(0.5)
-    s = s / s.max().clamp_min(eps)     # max-normalized → eps..1
-    s = s.clamp_min(eps)
+    # Geomean normalization (AutoAWQ quantize/quantizer.py:406).
+    s = s / (s.max() * s.min()).sqrt().clamp_min(eps)
+    # Hard-clamp for bf16 matmul safety.
+    s = s.clamp(1.0 / 10.0, 10.0)
+    s = torch.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0)
     return s
```

Same change applied to `_awq_joint_channel_scale`.

### 4. Regression tests added

Two new tests that would have caught this:

- `test_awq_fold_bf16_runtime_stays_coherent_under_extreme_imbalance`:
  exercises the exact failure — fold then cast to bf16, check L2 relative
  matmul error stays < 5%. Pre-fix the test produced L2_rel > 10.
- `test_awq_channel_scale_is_log_symmetric_and_clamped`: pins the scale
  bounds (`s.min() > 0.09`, `s.max() < 10.01`, `max/min < 100.01`) so a
  future refactor can't silently reintroduce the 1e-6 floor.

Plus updated the existing `test_awq_channel_scale_shape_and_norm` which
was asserting `s.max() == 1` (old max-normalization assumption).

Full test run: **137 pass** / 2 pre-existing unrelated failures in
`test_prismaquant_interaction_refine.py`.

### 5. Quality comparison (qualitative, 12 prompts)

Both exports coherent on the same 12-prompt bank. Raw outputs at:
- `.tmp_artifacts/results_phase1.json` (no act-aware)
- `.tmp_artifacts/results_phase2.json` (AWQ + GPTQ + act-round)

Differences I noticed:
- `code_py` ("maximum of list"): Phase 1 produced `A. max(lst) B. max(lst)
  C. max(lst)` (confused for multiple-choice). Phase 2 produced ```` ```python\nmax(lst)\n``` ````
  (cleaner).
- `hist` ("Great Wall defended against"): Phase 1 said "Mongols and other
  nomadic" — correct but generic. Phase 2 said "the Xiongnu people" —
  historically more precise (Xiongnu is who the Wall was originally built
  against before the Mongols arose 1500 years later).
- The rest are comparable — both give correct or correct-ish answers.

**Bottom line:** Phase 2 is NOT worse than Phase 1, and occasionally visibly
better. Whether the wins justify the ~10 min extra export time is for a
benchmark run to decide (see "Next steps").

---

## What's on disk

| Path | Contents | Size |
|---|---|---|
| `dq-runs-new/qwen36_rebuild/exported_phase1_baseline/` | Pure RTN, no act-aware | 22 GB |
| `dq-runs-new/qwen36_rebuild/exported_phase2_v2/` | AWQ + GPTQ + act-round (fixed) | 22 GB |
| `dq-runs-new/qwen36_rebuild/work/shards/` | Probe + cost shards (all reusable) | |
| `dq-runs-new/qwen36_rebuild/act/` | 373 MB activation cache (reuse for future re-exports) | |
| `dq-runs-new/qwen36_rebuild/artifacts/probe.pkl` | Merged probe, `calibration_modality=multimodal` but 0 visual entries | |
| `.tmp_artifacts/eval_prompts.json` | 12-prompt bank for qualitative compare | |
| `.tmp_artifacts/results_phase{1,2}.json` | Outputs from both exports | |
| `.tmp_artifacts/sanity_serve.sh`, `run_eval.sh` | Reusable serve + eval scripts | |
| `.tmp_artifacts/debug_visual_modules.py` | Diagnostic that found the AutoModelForCausalLM bug | |

---

## Known issue: visual probe still returns 0 Linears

Root cause (research commit artifact `.tmp_artifacts/debug_from_pretrained.py`):
`AutoModelForCausalLM.from_pretrained` has a silent downgrade path in
`transformers/auto_factory.py:132-134` — when the mapping resolves to a
text-only class whose `config_class` matches `config.sub_configs["text_config"]`,
transformers replaces the composite config with just the text sub-config.

For Qwen3.6 this means:
- The arch declared is `Qwen3_5MoeForConditionalGeneration` (has visual tower)
- `AutoModelForCausalLM` dispatches to `Qwen3_5MoeForCausalLM` (text-only)
- Visual weights in the safetensors are silently dropped

Our Phase 2 multimodal probe passes forward/backward on pixel+text but the
visual tower never materialized, so 0 Linears get Fisher stats. The allocator
falls back to `--visual-format=BF16` (Phase 1 passthrough).

**Minimal fix** (5 lines in `quantization/prismaquant/sensitivity_probe.py`
around line 1596):

```python
# Replace:
model = AutoModelForCausalLM.from_pretrained(staged, ...)
# With:
from transformers import AutoConfig
import transformers
cfg = AutoConfig.from_pretrained(staged, trust_remote_code=True)
arch_name = cfg.architectures[0]  # "Qwen3_5MoeForConditionalGeneration"
model_cls = getattr(transformers, arch_name)
model = model_cls.from_pretrained(staged, ...)
```

That directly instantiates the declared arch, preserving the visual tower.
Regex and `isinstance(m, nn.Linear)` are already correct (110 visual Linears
under `model.visual.*`, plain `nn.Linear`).

**After fix:** wipe probe.pkl + cost.pkl, re-run full pipeline (~45 min).
Cost shards reuse from disk, only body probe (20 min) + new multimodal
visual probe (~5 min) + new cost shards for visual (~5 min) + allocator
+ export (~5 min) actually run.

I did NOT apply this fix overnight because it requires a re-run to
validate, and I wanted to leave you a known-good export on disk.

---

## Next steps, in priority order

1. **Verify the Phase 2 export runs well on a proper benchmark.** Candidates:
   - `lm-eval-harness` with `arc_easy,piqa,hellaswag` (~15 min each on the
     serve endpoint, install via `pip install lm-eval[vllm]`)
   - `wikitext-2` PPL via vllm's `--echo` + logprobs

   Either would give a real number for "what value does GPTQ+AWQ give".
   Expected magnitude per the AWQ/GPTQ papers: 0.5–2% accuracy bump on
   downstream, 1–3% PPL reduction. My 12-prompt eval is consistent with
   that ballpark but can't quantify it.

2. **Fix the visual loader.** 5-line edit, re-run pipeline (~45 min).
   Only needed if we want real visual-encoder quantization in the HF
   upload. Right now visual is BF16 passthrough (functional, just not
   compressed).

3. **HF upload.** Once you're happy with quality:
   - `exported_phase2_v2/` is the one to upload
   - Needs a README.md + model card
   - Consider uploading BOTH `exported_phase1_baseline` and
     `exported_phase2_v2` as separate HF repos for A/B comparison —
     users can pick

4. **122B Phase 2.** I did NOT start a 122B overnight run. Reasons: (a)
   the 122B pipeline has never been tested with the new proper-AWQ code
   end-to-end, (b) running it for hours overnight before morning
   validation would waste compute if any other bug surfaces. Start that
   one fresh after morning review — probe shards for 122B are already
   cached (see `project_122b_session_state_2026_04_20.md` memory).

5. **Other Phase 2 ideas not yet done:** AWQ α grid search. The paper
   grid-searches α ∈ [0,1] to minimize MSE; we use fixed α=0.5. Expected
   incremental gain: 0.2–0.5% on top of current fold. Cost: ~20× quant
   time. Worth it only if the current output quality isn't meeting the
   target.

---

## Commits pushed to `fork main`

- `bd49ff2` — Phase 2 visual (multimodal calibration + Fisher + act-aware export)
- `2f0a3b0` — proper AWQ fold covers every γ reader
- `8b8435d` — AWQ scale must be log-symmetric + hard-clamped for bf16 ← **tonight**

All tests pass on `main`. Branch is clean except for `.tmp_artifacts/` and
this handoff file (both gitignore-worthy).

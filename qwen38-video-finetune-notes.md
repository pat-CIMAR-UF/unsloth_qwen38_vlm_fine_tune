# Qwen3.8-27B Video QLoRA — Setup and Findings

Session notes, 2026-08-14. Covers getting `video_qwen35.py` running, switching to Qwen3.8-27B, and three training runs that isolated two separate bugs in how the fine-tune was set up.

---

## Objective

Fine-tune a video VLM on 22 robot clips in `sample/` to answer "which side of the frame is the robot on?" with one word. This is a smoke test for the real target: automated review of strawberry-picking robot video.

---

## Environment

### What was wrong

The venv was effectively bare — `torch 2.13.0+cu130` and a `flash-attn 2.8.3` wheel with an ABI mismatch:

```
ImportError: flash_attn_2_cuda...so: undefined symbol:
_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

No unsloth, transformers, trl, peft, bitsandbytes, or numpy. `pyproject.toml` declared only `flash-attn==2.8.3`.

### Binding constraints

`unsloth 2026.8.18` caps `torch<2.12` and `transformers<=5.5.0`. Both bounds are also present on git `main`, so there was no newer-unsloth escape hatch. transformers 5.5.0 was verified to carry the architecture Qwen3.8 needs before committing to the pin.

### Pinned stack

| Package | Version | Note |
|---|---|---|
| torch | 2.11.0 | newest allowed by `unsloth`'s `torch<2.12` |
| torchvision | 0.26.0 | matching pair |
| transformers | 5.5.0 | ceiling of unsloth's range |
| unsloth | 2026.8.18 | |
| unsloth-zoo | 2026.8.12 | |
| trl | 0.24.0 | |
| bitsandbytes | 0.50.1 | |
| decord | 0.6.0 | video backend; ctypes-based, so the `py3-none` wheel works on 3.12 |

**flash-attn dropped.** unsloth bundles Triton flash-linear-attention kernels for the gated-deltanet layers this architecture uses, and falls back to SDPA elsewhere. **xformers overridden out** — its wheels are linked against one exact torch build, and a mismatch makes unsloth's `cpp_lib` probe throw at import.

Hardware: H100 NVL, 93.087 GB usable, driver 595.71.05, CUDA 13.2, Python 3.12.3.

---

## Model

`unsloth/Qwen3.8-27B` — dense 27B native VLM, 55.6 GB bf16 across 18 shards.

- `model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration` — built on the Qwen3.5 architecture, so transformers 5.5.0 loads it and unsloth has explicit handling (including VLM-specific save logic).
- Hybrid attention: 3 linear-attention layers per full-attention layer (`full_attention_interval: 4`) across 64 layers. MTP head present in the checkpoint, ignored on load via `_keys_to_ignore_on_load_unexpected`.
- **No pre-quantized bnb-4bit repo exists** for this checkpoint, so `load_in_4bit=True` quantizes the full bf16 download on load.
- Processor resolves to `Qwen3VLProcessor`. Vision tower: `patch_size 16`, `spatial_merge_size 2`, `temporal_patch_size 2`.

---

## Script changes

1. **`MODEL_NAME`** → `unsloth/Qwen3.8-27B`.
2. **`SFTTrainer(tokenizer=…)` → `processing_class=…`** — trl 0.23 dropped the alias. The unsloth-patched signature ends in `**kwargs`, so the old form would have been silently swallowed rather than erroring.
3. **Resize comment corrected.** `unsloth_zoo.fetch_video` rounds the requested size to a multiple of its own hardcoded `IMAGE_FACTOR = 28`, *then* Qwen3.8's video processor re-resizes to the model's grid factor of 32. The requested 512×832 decodes to 504×840 before that second step — the knobs are a request, not a guarantee.

### The `fps` kwarg is load-bearing

Passing the decoded clip's fps through to the processor changes the input by an order of magnitude:

| | input_ids | pixel_values_videos |
|---|---|---|
| with `fps=20.12` | 11,507 | 44,928 × 1536 |
| without | 1,331 | 4,992 × 1536 |

Without it the processor re-samples down to a handful of frames and warns about missing `video_metadata` rather than failing.

---

## Three runs

All: 22 examples × 3 epochs ÷ 4 grad-accum = 18 steps. LoRA r=16, α=16 on vision + language, attention + MLP — 124,427,776 of 27,481,156,336 params (0.45 %).

| | run 1 baseline | run 2 masked | run 3 masked + no-thinking |
|---|---|---|---|
| supervised tokens | 10,686 / 10,686 | 7 / 10,686 | 7 / 10,644 |
| prompt is training prefix | no | no | **yes** |
| step-1 loss (LR = 0) | 7.484 | 2.443 | **0.00053** |
| final loss | 0.076 | 1.35e-5 | 1.87e-7 |
| **pre**-train output | reasoning + `right` | reasoning + `right` | **`right`** |
| **post**-train output | reasoning + `right` | reasoning + `right` | **`right`** |
| training time | 622.8 s | 551.1 s | 550.4 s |
| peak reserved | 25.348 GB | 25.348 GB | 25.67 GB |

Scripts: `video_qwen35.py`, `video_qwen38_responses_only.py`, `video_qwen38_no_thinking.py`. Adapters in `qwen_video_lora*/`.

Run 1's step 1 took 112.4 s against run 2's 41.6 s — that gap is unsloth compiling its patched module sources into `unsloth_compiled_cache/` on first import, not a difference between the configurations.

---

## Diagnosis 1 — the loss was measuring a constant

With `train_on_responses_only` at its default of `False`, `UnslothVisionDataCollator` sets `labels = input_ids` and masks only padding. Every position is supervised, including ~10,600 video tokens, the system preamble, and the instruction — all byte-identical across the 22 examples.

So the answer was **1 part in ~1500** of the loss. Run 1's clean-looking 7.48 → 0.076 curve was mostly the model fitting a constant prefix.

### Fix

```python
UnslothVisionDataCollator(
    model, tokenizer,
    train_on_responses_only = True,
    instruction_part        = "<|im_start|>user\n",
    response_part           = "<|im_start|>assistant\n",
)
```

The two strings are **landmarks, not content**. Everything from `response_part` to the next `instruction_part` is supervised; everything else is set to `-100`, PyTorch's `ignore_index`, contributing zero loss and zero gradient. Those tokens are still fed to the model as input — it still *sees* the video, it just isn't graded on predicting it.

Internally the markers are tokenized and reduced to a stable core of token IDs (`_find_common_token_ids`), because the same text tokenizes differently depending on surrounding context. This is why the trailing `\n` belongs in each marker. Passing *neither* argument triggers auto-detection from the chat template; passing exactly one raises.

Masking does not make training cheaper — the same tokens still go forward and backward.

---

## Diagnosis 2 — the generation prompt was off by one token

Run 2 fixed the measurement (loss 1.35e-5 on the answer span) and changed nothing about the model's behavior. Cause:

```
train:     <think>(248068)  \n\n(271)  </think> …
generate:  <think>(248068)  \n(198)              ← state never supervised
```

Qwen3.8 enables thinking by default at `reasoning_effort='xhigh'`. With thinking on, a generation prompt ends *inside* an open think block, one token away from anything training conditioned on. The LoRA never applies and the base model's reasoning habit survives — no amount of masking reaches this.

### Fix

`enable_thinking=False` on **every** `apply_chat_template` call. In the text-only render, the generation prompt becomes an exact token prefix of the training sequence:

```
train:     …<|im_start|>assistant\n<think>\n\n</think>\n\nright<|im_end|>\n   (20 tokens)
generate:  …<|im_start|>assistant\n<think>\n\n</think>\n\n                    (17 tokens)
continuation: [(1246,'right'), (248046,'<|im_end|>'), (198,'\n')]
```

It must be applied to training too, not just inference — the flag also drops the `xhigh` system preamble, so a one-sided change trades one mismatch for another. `UnslothVisionDataCollator` renders training text internally and its `formatting_func` hook transforms the *example* rather than the text, so the flag is injected by patching `apply_chat_template` on the processor.

### Two guards now in the script

Both run before training and cost seconds:

```
Label mask check:  7 of 10644 tokens carry loss (0.066%)
Supervised span -> '<think>\n\n</think>\n\nright<|im_end|>\n'
Prompt alignment: generation prompt is a token prefix of training sequence -> True
Model must generate -> [(1246,'right'), (248046,'<|im_end|>'), (198,'\n')]
```

A wrong marker or a template change produces silence, not an error — either everything masked or nothing masked. These catch that.

---

## What the runs actually proved

**The format goal was never a training problem.** Run 3's *pre*-train row is the tell: with the prompt correctly constructed, the untuned model already emits a bare `right`. Step-1 loss of 5.3e-4 at learning rate 0 means there was no gradient signal left to extract — run 3 was numerically close to a no-op.

**The dataset can't demonstrate learning.** The label is perfectly predicted by the filename suffix:

```
MP1 → right  (12/12)
MP2 → left   (10/10)
```

MP1/MP2 appear to be camera IDs, so "which side is the robot on" is fully confounded with "which camera". Nothing here distinguishes a model that watched the robot from one that recognized the viewpoint.

What *was* validated is the pipeline: environment, model load, video decode, masking, prompt alignment, training loop, adapter save.

Minor note: `temperature=0.7, min_p=0.1` in these scripts are inert — `do_sample` is never set, so generation is greedy. That's correct for classification, just not sampling.

---

## Applying this to strawberry-pick review

Target question: "did the robot do the work successfully, and if not, which failure category?" Planned as a two-stage cascade.

**Stage 1 — fast binary classifier.** "Did the robot pick the strawberry?" `enable_thinking=False` plus response-only masking. Single-token answer, cheap to evaluate over many clips, and `P(yes)` vs `P(no)` on that token doubles as a routing-confidence signal for borderline cases.

**Stage 2 — failure-category diagnosis, thinking left on.** Try prompting the base model with the taxonomy first. Fine-tuning *with* thinking on requires reasoning traces in the training targets — bare category labels would rebuild Diagnosis 2 exactly. Rationales mean hand-writing, distilling from a stronger model, or rejection-sampling the base model's own correct traces. Given that run 3 showed the base model solving the toy task with zero gradient steps, assume prompting may suffice until measured.

The cascade matches labelling cost to need (binary labels on everything, taxonomy labels only on failures) and reasoning compute to need (only the minority that failed).

### What decides whether it works

- **Stage 1's metric is failure recall, not accuracy.** A false "success" is unrecoverable — that clip never reaches stage 2. A false "failure" costs one extra call. Tune stage 1 to over-flag. With successes dominating, a model that always answers "success" will look excellent and be worthless.
- **Split by session / robot / day, never randomly.** See the MP1/MP2 confound above. The same trap awaits if failure clips differ systematically in lighting, operator, or duration.
- **Check the evidence is in the sampled frames.** A failure is often a brief event. 80 frames over a 3 s clip captures everything; over a 60 s clip that's 1.3 fps and may sample straight past the slip.
- **Report per-class recall and a confusion matrix.** Aggregate accuracy and loss will both look fine while a rare category is never predicted once.
- **Volume.** 22 clips is a smoke test. A multi-class taxonomy on a 27B LoRA realistically wants hundreds to low thousands, each category properly represented.

Both stages can be separate LoRA adapters over the same 4-bit base — swap adapters, no second 55 GB download.

---

## Open items

1. **No eval harness exists.** Every result above is one generation on one training clip. Needed: held-out split, batch generation, exact-match parse, confusion matrix, failure recall.
2. **Generalize the script** to a configurable label taxonomy so it takes the review dataset directly.
3. **Baseline before tuning.** Measure the zero-shot base model with thinking on. Nothing so far establishes that fine-tuning beats prompting.

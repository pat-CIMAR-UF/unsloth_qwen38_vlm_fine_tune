# -*- coding: utf-8 -*-
"""Qwen3.8 video QLoRA fine-tuning — response-only loss + thinking disabled.

Third variant of the same experiment (classify which side of the frame a robot
is on, from .mp4 clips in ``sample/``). It carries the fix from
``video_qwen38_responses_only.py`` and adds the one that run was missing.

What the two earlier runs showed:

1. ``video_qwen35.py`` leaves ``train_on_responses_only`` at ``False``, so
   ``UnslothVisionDataCollator`` sets ``labels = input_ids`` and supervises the
   whole sequence — ~10.7k tokens of video, system preamble and instruction
   that are identical across all 22 examples, versus 7 tokens of answer. The
   loss fell to 0.076 while measuring almost nothing.
2. ``video_qwen38_responses_only.py`` masks everything but the assistant turn,
   which fixed the measurement (loss 1.35e-5 on the answer span) but *not* the
   behaviour: the tuned model still emitted a full reasoning trace. The cause
   is a token-boundary mismatch, not the masking::

       train:     <think>(248068)  \\n\\n(271)  </think> …
       generate:  <think>(248068)  \\n(198)     <- state never supervised

   With thinking on, a generation prompt ends *inside* an open think block, one
   token off from anything training conditioned on, so the LoRA never applies
   and the base model's reasoning habit survives.

The fix here is ``enable_thinking = False`` on **every** chat-template render.
That makes the generation prompt an exact token-level prefix of the training
sequence::

    train:     …<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nright<|im_end|>\\n
    generate:  …<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
    continuation: ['right', '<|im_end|>', '\\n']

It has to be applied to training as well as inference, because the flag also
drops Qwen3.8's ``reasoning_effort='xhigh'`` system preamble — setting it on
one side only would trade one mismatch for another.
``UnslothVisionDataCollator`` renders training text internally and its
``formatting_func`` hook transforms the *example* rather than the text, so the
flag goes in via a small patch on ``apply_chat_template``.

Both invariants are asserted before training starts, so a template change
fails loudly instead of quietly wasting a run.

Outputs go to their own directories so the earlier runs' artifacts survive.

Run: ``uv run video_qwen38_no_thinking.py``
"""

# --- Install notes ----------------------------------------------------------
# Dependencies are pinned in pyproject.toml; `uv sync` installs them.
# The binding constraints come from unsloth 2026.8.18: torch<2.12 and
# transformers<=5.5.0, hence torch 2.11.0 / torchvision 0.26.0 /
# transformers 5.5.0 / trl 0.24.0.
# decord is the preferred video backend for unsloth. torchcodec / PyAV also work.

import os
import csv
from pathlib import Path

from unsloth import FastVisionModel  # noqa: E402  (must import before torch)
import torch

# Qwen3.5 VL uses 3-D M-RoPE position_ids (shape [3, batch, seq]).
# transformers ≥5.2 _is_packed_sequence() assumes 2-D and misidentifies them
# as a packed sequence, triggering varlen flash-attention with wrong cu_seqlens
# → CUDA illegal memory access. Patch the check to bail on non-2-D inputs.
import transformers.modeling_flash_attention_utils as _fa_utils
_orig_is_packed = _fa_utils._is_packed_sequence
def _is_packed_sequence_safe(position_ids, batch_size):
    if position_ids is not None and position_ids.dim() != 2:
        return False
    return _orig_is_packed(position_ids, batch_size)
_fa_utils._is_packed_sequence = _is_packed_sequence_safe


# ---------------------------------------------------------------------------
# Video sampling knobs
# ---------------------------------------------------------------------------
MAX_FRAMES     = 80       # max frames fed to the VLM per clip (multiple of 2)
FPS            = None     # target sampling fps; set to None to force nframes=MAX_FRAMES
# These are a request, not a guarantee: unsloth_zoo.fetch_video first rounds
# them to a multiple of its own IMAGE_FACTOR (28), then Qwen3.8's video
# processor re-resizes to the model's grid factor of 32 (patch_size=16 *
# spatial_merge_size=2). 512x832 decodes to 504x840 before that second step.
RESIZED_HEIGHT = 512
RESIZED_WIDTH  = 832

# ---------------------------------------------------------------------------
# Response-only loss knobs
# ---------------------------------------------------------------------------
# Qwen3.8 uses the ChatML markers; these are the exact strings the template
# emits at the start of each turn.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART    = "<|im_start|>assistant\n"

# Applied to every apply_chat_template call — training and inference alike.
# See the module docstring for why this cannot be an inference-only setting.
ENABLE_THINKING = False

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
VIDEO_DIR    = HERE / "sample"
LABEL_CSV    = HERE / "sample_dataset_label.csv"
OUTPUT_DIR   = HERE / "outputs_video_no_thinking"
ADAPTER_DIR  = HERE / "qwen_video_lora_no_thinking"

# ---------------------------------------------------------------------------
# Load model (4-bit => QLoRA)
# ---------------------------------------------------------------------------
# No pre-quantized bnb-4bit repo exists for this checkpoint, so the 55.6 GB
# bf16 weights are quantized to 4-bit on load.
MODEL_NAME = "unsloth/Qwen3.8-27B"

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit = True,                    # 4-bit base => QLoRA
    use_gradient_checkpointing = "unsloth", # save memory for long video contexts
)

# Force thinking off for every chat-template render. UnslothVisionDataCollator
# calls processor.apply_chat_template() itself with no way to pass template
# kwargs, so the flag is injected at the processor instead of at each call
# site — that way the training text and the generation prompt cannot drift
# apart. setdefault, so an explicit enable_thinking= still wins.
_orig_apply_chat_template = tokenizer.apply_chat_template
def _apply_chat_template_no_thinking(*args, **kwargs):
    kwargs.setdefault("enable_thinking", ENABLE_THINKING)
    return _orig_apply_chat_template(*args, **kwargs)
tokenizer.apply_chat_template = _apply_chat_template_no_thinking

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# ---------------------------------------------------------------------------
# Build the video dataset from CSV + local .mp4 files
# ---------------------------------------------------------------------------
INSTRUCTION = (
    "Watch the short robot clip. Which side of the frame is the robot on? "
    "Answer with exactly one word: 'left' or 'right'."
)


def load_labels(csv_path: Path):
    with csv_path.open(newline = "") as f:
        reader = csv.DictReader(f)
        return [(row["video_name"], row["robot_on_which_side"].strip().lower())
                for row in reader]


def _probe_total_frames(video_path: Path) -> int:
    """Cheap decord probe for total frame count — avoids fetching pixels."""
    import decord
    return len(decord.VideoReader(str(video_path), num_threads=1))


def make_video_element(video_path: Path) -> dict:
    """Build the video dict consumed by unsloth_zoo.process_vision_info.

    Keys understood by unsloth:
      - fps / nframes (mutually exclusive)
      - min_frames / max_frames  (only used with fps; cap the count)
      - resized_height / resized_width  (fixed output size)
      - min_pixels / max_pixels         (alternative pixel-budget control)
    """
    ele = {
        "type": "video",
        "video": str(video_path),
        "resized_height": RESIZED_HEIGHT,
        "resized_width":  RESIZED_WIDTH,
    }
    if FPS is not None:
        ele["fps"] = FPS
        # FPS_MIN_FRAMES default is 4, FPS_MAX_FRAMES default is 768.
        # Cap with our own MAX_FRAMES so we never exceed the budget.
        ele["max_frames"] = MAX_FRAMES
        # min_frames must be <= max_frames and a multiple of FRAME_FACTOR (=2).
        ele["min_frames"] = min(2, MAX_FRAMES)
    else:
        # smart_nframes requires nframes <= total_frames and a multiple of
        # FRAME_FACTOR (=2). Cap at MAX_FRAMES but clamp to the clip's own
        # length (rounded down to even) for short videos.
        total = _probe_total_frames(video_path)
        ele["nframes"] = min(MAX_FRAMES, (total // 2) * 2)
    return ele


def convert_to_conversation(video_path: Path, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [
                make_video_element(video_path),
                {"type": "text", "text": INSTRUCTION},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer},
            ]},
        ]
    }


rows = load_labels(LABEL_CSV)
missing = [name for name, _ in rows if not (VIDEO_DIR / name).exists()]
if missing:
    raise FileNotFoundError(f"Missing video files in {VIDEO_DIR}: {missing}")

converted_dataset = [convert_to_conversation(VIDEO_DIR / name, label)
                     for name, label in rows]
print(f"Built {len(converted_dataset)} video training examples "
      f"(MAX_FRAMES={MAX_FRAMES}, FPS={FPS}, "
      f"size={RESIZED_HEIGHT}x{RESIZED_WIDTH}).")

# Sanity-check: decode one clip through unsloth's pipeline before training.
from unsloth_zoo.vision_utils import process_vision_info  # noqa: E402
_probe_imgs, _probe_vids, _probe_kw = process_vision_info(
    converted_dataset[0]["messages"], return_video_kwargs = True,
)
print(f"Probe clip -> {_probe_vids[0].shape if _probe_vids else 'no video'} "
      f"kwargs={_probe_kw}")

# ---------------------------------------------------------------------------
# Pre-train inference sample
#
# With ENABLE_THINKING = False the generation prompt already contains a closed,
# empty think block, so the base model has no room to reason and should answer
# straight away — unlike the two earlier runs, where the prompt ended inside an
# open <think>.
# ---------------------------------------------------------------------------
from transformers import TextStreamer  # noqa: E402

FastVisionModel.for_inference(model)

_sample = converted_dataset[0]["messages"]
_inf_messages = [{"role": "user", "content": [
    make_video_element(VIDEO_DIR / rows[0][0]),
    {"type": "text", "text": INSTRUCTION},
]}]
_inf_imgs, _inf_vids, _inf_vid_kwargs = process_vision_info(
    _inf_messages, return_video_kwargs = True,
)
_inf_text = tokenizer.apply_chat_template(
    _inf_messages, add_generation_prompt = True, tokenize = False,
)
_inf_fps = _inf_vid_kwargs.get("fps") or []
_inf_inputs = tokenizer(
    text            = [_inf_text],
    images          = _inf_imgs,
    videos          = _inf_vids,
    return_tensors  = "pt",
    **({"fps": float(_inf_fps[0])} if _inf_fps else {}),
).to("cuda")

print("\n--- Pre-train generation (expected:", rows[0][1], ") ---")
_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**_inf_inputs, streamer = _streamer, max_new_tokens = 512,
                   use_cache = True, temperature = 0.7, min_p = 0.1)
print()

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402
from trl import SFTTrainer, SFTConfig  # noqa: E402

FastVisionModel.for_training(model)

collator = UnslothVisionDataCollator(
    model, tokenizer,
    max_seq_length = 40960,
    # The whole point of this variant: mask everything but the assistant turn.
    train_on_responses_only = True,
    instruction_part        = INSTRUCTION_PART,
    response_part           = RESPONSE_PART,
)

# Verify the mask before burning 10 minutes of H100 time: decode the tokens
# that survive masking on one example. This should print only the assistant
# turn (empty think block + the label), NOT the system prompt or instruction.
_check_batch = collator([converted_dataset[0]])
_labels = _check_batch["labels"][0]
_kept   = _labels[_labels != -100]
print(f"\nLabel mask check: {_kept.numel()} of {_labels.numel()} tokens carry loss "
      f"({100.0 * _kept.numel() / _labels.numel():.3f}%)")
print("Supervised span ->", repr(tokenizer.decode(_kept)))
if _kept.numel() == 0:
    raise RuntimeError(
        "Response-only masking removed every label — check INSTRUCTION_PART / "
        "RESPONSE_PART against the chat template's turn markers."
    )
del _check_batch, _labels, _kept
torch.cuda.empty_cache()

# Verify the fix this script exists for: the generation prompt must be an exact
# token-level PREFIX of the training sequence. If it is not, the model is asked
# at inference to continue from a state training never supervised, the LoRA does
# not apply, and the base model's habits survive — which is exactly what
# happened in the two earlier runs. Compared on text only (no video decode):
# the video placeholder expands identically on both sides.
_align_msgs  = converted_dataset[0]["messages"]
_train_text  = tokenizer.apply_chat_template(_align_msgs, tokenize = False,
                                             add_generation_prompt = False)
_prompt_text = tokenizer.apply_chat_template(_align_msgs[:1], tokenize = False,
                                             add_generation_prompt = True)
_inner_tok   = getattr(tokenizer, "tokenizer", tokenizer)
_train_ids   = _inner_tok(_train_text,  add_special_tokens = False)["input_ids"]
_prompt_ids  = _inner_tok(_prompt_text, add_special_tokens = False)["input_ids"]
_aligned     = _train_ids[:len(_prompt_ids)] == _prompt_ids
print(f"\nPrompt alignment check: generation prompt is a token prefix of the "
      f"training sequence -> {_aligned}")
print("Model must generate ->",
      [(i, _inner_tok.decode([i])) for i in _train_ids[len(_prompt_ids):]])
if not _aligned:
    _first_diff = next(
        (i for i, (a, b) in enumerate(zip(_train_ids, _prompt_ids)) if a != b),
        min(len(_train_ids), len(_prompt_ids)),
    )
    raise RuntimeError(
        "Generation prompt is not a prefix of the training sequence; they "
        f"diverge at token {_first_diff} "
        f"(train {_train_ids[_first_diff:_first_diff+3]} vs "
        f"prompt {_prompt_ids[_first_diff:_first_diff+3]}). "
        "Check that ENABLE_THINKING reaches both renders."
    )

trainer = SFTTrainer(
    model            = model,
    processing_class = tokenizer,   # trl >= 0.23 dropped the `tokenizer` alias
    data_collator    = collator,
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,       # video batches are heavy
        gradient_accumulation_steps = 4,
        warmup_steps                = 1,
        max_steps                   = -1,       # smoke run; set num_train_epochs=1 for a full epoch
        num_train_epochs            = 3,
        learning_rate               = 2e-4,
        logging_steps               = 1,
        optim                       = "adamw_8bit",
        weight_decay                = 0.001,
        lr_scheduler_type           = "linear",
        seed                        = 3407,
        output_dir                  = str(OUTPUT_DIR),
        report_to                   = "none",

        # Required for vision fine-tuning
        remove_unused_columns = False,
        dataset_text_field    = "",
        dataset_kwargs        = {"skip_prepare_dataset": True},
        max_length            = 40960,          # videos produce many vision tokens
    ),
)

gpu_stats          = torch.cuda.get_device_properties(0)
start_gpu_memory   = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_memory         = round(gpu_stats.total_memory          / 1024**3, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory          = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory,           3)
print(f"{trainer_stats.metrics['train_runtime']:.1f} s training runtime.")
print(f"Peak reserved memory = {used_memory} GB "
      f"(training delta = {used_memory_for_lora} GB).")

# ---------------------------------------------------------------------------
# Post-train inference on the same clip
# ---------------------------------------------------------------------------
FastVisionModel.for_inference(model)

_post_inputs = tokenizer(
    text            = [_inf_text],
    images          = _inf_imgs,
    videos          = _inf_vids,
    return_tensors  = "pt",
    **({"fps": float(_inf_fps[0])} if _inf_fps else {}),
).to("cuda")
print(f"\n--- Post-train generation (expected: {rows[0][1]}) ---")
_ = model.generate(**_post_inputs, streamer = _streamer, max_new_tokens = 512,
                   use_cache = True, temperature = 0.7, min_p = 0.1)
print()

# ---------------------------------------------------------------------------
# Save LoRA adapters
# ---------------------------------------------------------------------------
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print(f"Saved LoRA adapters to {ADAPTER_DIR}")

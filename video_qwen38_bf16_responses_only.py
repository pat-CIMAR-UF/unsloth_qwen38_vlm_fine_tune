# -*- coding: utf-8 -*-
"""Qwen3.8 video LoRA fine-tuning (bf16) — response-only loss variant.

Same experiment as ``video_qwen35.py`` (classify which side of the frame a
robot is on, from .mp4 clips in ``sample/``), with one deliberate change to
the training objective.

``video_qwen35.py`` leaves ``train_on_responses_only`` at its default of
``False``. In that mode ``UnslothVisionDataCollator`` sets
``labels = input_ids`` and masks only the padding, so the loss covers the
whole sequence: the ~11.5k identical video tokens, the shared system prompt,
and the instruction — all of which are byte-identical across the 22 examples.
The one-word answer is a vanishing fraction of that, so a low reported loss
mostly reflects fitting a constant prefix, and the tuned model's answering
behaviour barely moves.

This script masks everything except the assistant turn::

    train_on_responses_only = True
    instruction_part        = "<|im_start|>user\\n"
    response_part           = "<|im_start|>assistant\\n"

so every gradient lands on the span the task actually cares about. Qwen3.8
renders a training turn as::

    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nright<|im_end|>

i.e. an *empty* think block followed by the answer, while a generation prompt
ends at ``<|im_start|>assistant\\n<think>\\n``. Masking from
``<|im_start|>assistant\\n`` onward therefore also teaches the model to close
the think block immediately instead of reasoning — which is what we want for a
one-word classifier, and what the unmasked run failed to do.

Outputs go to their own directories so the baseline run's artifacts survive.

Run: ``uv run video_qwen38_bf16_responses_only.py``

NOTE: bf16 variant. Identical to video_qwen38_responses_only.py except that the base model
is loaded unquantized (``load_in_4bit = False``, ``load_in_16bit = True``),
making this plain LoRA rather than QLoRA. Outputs go to their own directories
so the 4-bit run's artifacts survive.
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

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
VIDEO_DIR    = HERE / "sample"
LABEL_CSV    = HERE / "sample_dataset_label.csv"
OUTPUT_DIR   = HERE / "outputs_video_bf16_responses_only"
ADAPTER_DIR  = HERE / "qwen_video_lora_bf16_responses_only"

# ---------------------------------------------------------------------------
# Load model (bf16 => LoRA, no quantization)
# ---------------------------------------------------------------------------
# The 55.6 GB bf16 weights are used as-is. Quantization exists to fit a large
# model on a small card; a 27B model on a 93 GB H100 does not need it. Keeping
# the vision tower unquantized also avoids putting quantization error in front
# of the fine visual discrimination the real task depends on.
#
# unsloth requires load_in_4bit to be turned off explicitly - it defaults to
# True, and setting more than one of 4bit/8bit/16bit raises.
MODEL_NAME = "unsloth/Qwen3.8-27B"

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit  = False,                  # no quantization at all
    load_in_16bit = True,                   # bf16 base => plain LoRA, not QLoRA
    use_gradient_checkpointing = "unsloth", # save memory for long video contexts
)

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
# Qwen3.8's chat template turns thinking on by default (reasoning_effort
# 'xhigh'), so the base model emits a <think> block before answering. The
# response-only masking below trains the model out of that: its targets start
# at "<|im_start|>assistant\n" and contain an empty think block.
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

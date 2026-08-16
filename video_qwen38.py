# -*- coding: utf-8 -*-
"""Qwen3.8 video QLoRA fine-tuning example.

Uses the unified multimodal ``unsloth/Qwen3.8-27B`` checkpoint — a dense 27B
native VLM (image+video+text from a single model) built on the Qwen3.5
architecture, so transformers loads it as ``Qwen3_5ForConditionalGeneration``.
Adapted from the Unsloth vision notebook — instead of single images, this
script loads .mp4 clips from ``sample/`` (listed in
``sample_dataset_label.csv``) and fine-tunes the model to answer "which side
is the robot on?".

Video sampling is controlled by three knobs at the top of the script:

* ``MAX_FRAMES``        - hard cap on the number of frames sampled per clip
* ``FPS``               - target sampling rate (fps). Set to ``None`` to use
                          ``MAX_FRAMES`` as an exact frame count instead.
* ``RESIZED_HEIGHT`` /
  ``RESIZED_WIDTH``     - output resolution fed to the vision tower

The heavy lifting (decord/torchvision decode, uniform sampling, smart resize
to patch-factor multiples) is done by ``unsloth_zoo.vision_utils`` via the
``{"type": "video", ...}`` message field.

Run: ``uv run video_qwen35.py``
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
# Dataset paths
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
VIDEO_DIR    = HERE / "sample"
LABEL_CSV    = HERE / "sample_dataset_label.csv"
OUTPUT_DIR   = HERE / "outputs_video"
ADAPTER_DIR  = HERE / "qwen_video_lora"

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
# 'xhigh'), so the base model will emit a <think> block before answering.
# We leave it on here because UnslothVisionDataCollator calls
# apply_chat_template with template defaults during training — turning it off
# only at inference would train on a different prompt than we generate with.
# The training targets are bare "left"/"right", so the tuned model learns to
# answer immediately.
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

trainer = SFTTrainer(
    model            = model,
    processing_class = tokenizer,   # trl >= 0.23 dropped the `tokenizer` alias
    data_collator = UnslothVisionDataCollator(model, tokenizer,
                                              max_seq_length = 40960),
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

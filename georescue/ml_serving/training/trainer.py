"""LoRA fine-tuning for Qwen2-VL-7B on GIS flood analysis data (AMD ROCm compatible)."""
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from .dataset import FloodAnalysisDataset

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)


def load_model_for_training(model_id: str = MODEL_ID):
    has_gpu = torch.cuda.is_available()
    dtype = torch.float16 if has_gpu else torch.float32

    if not has_gpu:
        print("[warn] No GPU detected — loading in float32 on CPU. Training will be very slow.")
        print("[warn] For real training speed, ensure ROCm/CUDA drivers are installed.")
    else:
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"[gpu] {name}  {vram} GB VRAM")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if has_gpu else None,
        trust_remote_code=True,
    )
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return model, processor


def _make_collate_fn(processor):
    def collate(batch: list) -> dict:
        texts, images = [], []
        for item in batch:
            has_image = "image" in item
            content = []
            if has_image:
                content.append({"type": "image", "image": item["image"]})
                images.append(item["image"])
            content.append({"type": "text", "text": item["prompt"]})

            conversation = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": item["expected_output"]}]},
            ]
            texts.append(
                processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
            )

        inputs = processor(
            text=texts,
            images=images or None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    return collate


def train(
    dataset_index: "str | Path",
    output_dir: "str | Path" = "checkpoints",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    eval_split: float = 0.1,
    resume_from: Optional[str] = None,
) -> None:
    model, processor = load_model_for_training()

    full_dataset = FloodAnalysisDataset(dataset_index, require_image=False)
    n_eval = max(1, int(len(full_dataset) * eval_split))
    n_train = len(full_dataset) - n_eval
    train_set, eval_set = torch.utils.data.random_split(full_dataset, [n_train, n_eval])

    output_dir = Path(output_dir)
    # Epoch checkpoints go to a sibling dir so output_dir stays clean for the adapter.
    checkpoint_dir = output_dir.with_name(output_dir.name + "_checkpoints")

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=_make_collate_fn(processor),
    )

    trainer.train(resume_from_checkpoint=resume_from)

    # Save final adapter directly to output_dir — this is where model_loader.py loads from.
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Fine-tuned adapter saved to {output_dir}")

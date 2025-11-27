#!/usr/bin/env python3
"""
SFT Training script for TASK format traces.
Supports multi-GPU training with DeepSpeed, checkpointing, and resumption.

Usage:
    # Single run
    accelerate launch --config_file accelerate_config.yaml train.py
    
    # Resume from checkpoint
    accelerate launch --config_file accelerate_config.yaml train.py --resume
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen3-4B-Base"
    
    # Data
    data_path: str = "traces.processed.jsonl"
    max_seq_length: int = 32768  # Qwen3 supports 32k
    
    # Training (optimized for 8xH200 140GB)
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # H200 can handle much more
    gradient_accumulation_steps: int = 2  # Effective batch = 4 * 2 * 8 GPUs = 64
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    
    # LoRA (set use_lora=False for full fine-tune)
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Optimization
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    
    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    report_to: str = "tensorboard"
    
    # Resume
    resume_from_checkpoint: Optional[str] = None


def load_traces(data_path: str) -> list[dict]:
    """Load processed traces from JSONL."""
    traces = []
    with open(data_path) as f:
        for line in f:
            try:
                item = json.loads(line)
                traces.append({"text": item["trace"]})
            except:
                pass
    print(f"Loaded {len(traces)} traces from {data_path}")
    return traces


def create_dataset(traces: list[dict], tokenizer, max_seq_length: int) -> Dataset:
    """Create tokenized dataset from traces."""
    
    # First pass: get true lengths without truncation
    true_lengths = []
    for trace in traces:
        tokens = tokenizer(trace["text"], truncation=False, add_special_tokens=True)
        true_lengths.append(len(tokens["input_ids"]))
    
    # Report truncation stats
    truncated = sum(1 for l in true_lengths if l > max_seq_length)
    if truncated > 0:
        over_lengths = [l for l in true_lengths if l > max_seq_length]
        print(f"\n⚠️  Truncation warning:")
        print(f"   {truncated}/{len(traces)} traces exceed max_seq_length ({max_seq_length})")
        print(f"   Max length: {max(true_lengths)}, Median: {sorted(true_lengths)[len(true_lengths)//2]}")
        print(f"   Truncated lengths: min={min(over_lengths)}, max={max(over_lengths)}, avg={sum(over_lengths)//len(over_lengths)}")
    else:
        print(f"✓ No truncation needed (max trace: {max(true_lengths)} tokens)")
    
    def tokenize(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=True,
        )
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    dataset = Dataset.from_list(traces)
    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
        desc="Tokenizing"
    )
    
    # Filter out sequences that are too short
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 100)
    print(f"Filtered {original_len - len(dataset)} short sequences")
    
    return dataset


def setup_model_and_tokenizer(config: TrainConfig):
    """Load model and tokenizer."""
    
    print(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
    )
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Apply LoRA if enabled
    if config.use_lora:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--data", type=str, default="traces.processed.jsonl")
    parser.add_argument("--output", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, help="Resume from specific checkpoint")
    parser.add_argument("--save-steps", type=int, default=500)
    args = parser.parse_args()
    
    # Build config
    config = TrainConfig(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        save_steps=args.save_steps,
    )
    
    # Handle resume
    if args.resume_from:
        config.resume_from_checkpoint = args.resume_from
    elif args.resume:
        # Find latest checkpoint
        checkpoint_dir = Path(config.output_dir)
        if checkpoint_dir.exists():
            checkpoints = sorted(
                [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda x: int(x.name.split("-")[1])
            )
            if checkpoints:
                config.resume_from_checkpoint = str(checkpoints[-1])
                print(f"Resuming from: {config.resume_from_checkpoint}")
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load and prepare data
    traces = load_traces(config.data_path)
    dataset = create_dataset(traces, tokenizer, config.max_seq_length)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample token lengths: {[len(dataset[i]['input_ids']) for i in range(min(5, len(dataset)))]}")
    
    # Data collator - handles padding for variable length sequences
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,  # Efficient for tensor cores
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        tf32=config.tf32,
        gradient_checkpointing=config.gradient_checkpointing,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        # DeepSpeed will be configured via accelerate
        deepspeed=None,
        # For multi-GPU
        ddp_find_unused_parameters=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print(f"  Model: {config.model_name}")
    print(f"  Data: {config.data_path} ({len(dataset)} samples)")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size per device: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max seq length: {config.max_seq_length}")
    print(f"  LoRA: {config.use_lora}")
    print(f"  Output: {config.output_dir}")
    if config.resume_from_checkpoint:
        print(f"  Resuming from: {config.resume_from_checkpoint}")
    print("="*60 + "\n")
    
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(config.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final"))
    
    print("Training complete!")


if __name__ == "__main__":
    main()


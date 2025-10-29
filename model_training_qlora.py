"""
Fine-tune Llama 3.1 Instruct model using QLoRA for AI Ethics Education
QLoRA (Quantized LoRA) enables efficient fine-tuning of large models on consumer hardware
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prompt_templates import format_llama_training_example
import os

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Can also use 3.1-70B if you have resources
OUTPUT_DIR = "./ai_ethics_llm_qlora"
FINAL_MODEL_DIR = "./ai_ethics_llm_final"

# QLoRA Configuration - 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading base model: {MODEL_NAME}")
print("Using 4-bit quantization for memory efficiency...")

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA Configuration
lora_config = LoraConfig(
    r=16,  # Rank of the update matrices (higher = more parameters, but better quality)
    lora_alpha=32,  # Scaling factor
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Target all linear layers in attention and FFN
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("Applying LoRA adapters...")
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")
print(f"Total parameters: {total_params:,}")

# Load dataset
print("\nLoading training dataset...")
dataset = load_dataset('json', data_files='ai_ethics_dataset.jsonl')

print(f"Dataset size: {len(dataset['train'])}")
if len(dataset['train']) > 0:
    print(f"Sample entry: {dataset['train'][0]}")

# Tokenize the dataset using Llama 3.1 format
def tokenize_function(examples):
    """
    Format examples using Llama 3.1 Instruct chat template
    """
    texts = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        # Use the proper Llama 3.1 Instruct format
        formatted_text = format_llama_training_example(instruction, response)
        texts.append(formatted_text)

    # Tokenize with truncation and padding
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=2048,  # Llama 3.1 supports up to 128k, but 2048 is reasonable for training
        padding="max_length",
        return_tensors="pt"
    )

    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing"
)

# Training arguments optimized for QLoRA
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
    num_train_epochs=3,
    learning_rate=2e-4,  # Higher LR for LoRA
    fp16=False,
    bf16=True,  # Use bfloat16 for better stability
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
    gradient_checkpointing=True,  # Further reduce memory usage
    max_grad_norm=0.3,
    group_by_length=True,  # Efficiency improvement
    report_to="none",  # Disable wandb/tensorboard if not needed
)

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# Train the model
print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")

trainer.train()

# Save the LoRA adapters
print("\n" + "="*50)
print("Training complete! Saving model...")
print("="*50 + "\n")

model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print(f"Model and adapters saved to {FINAL_MODEL_DIR}")
print("\nTo use the model:")
print("1. Load base model with quantization")
print("2. Load LoRA adapters from saved directory")
print("3. Merge if deploying to production (optional)")

# Optional: Merge LoRA weights with base model for easier deployment
print("\nWould you like to merge LoRA adapters with base model?")
print("This creates a standalone model but requires more disk space.")
print("Run the merge_lora.py script to do this.")

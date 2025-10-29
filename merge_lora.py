"""
Merge LoRA adapters with base model for standalone deployment
This creates a full model that can be used without loading adapters separately
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_MODEL = "./ai_ethics_llm_final"
OUTPUT_MODEL = "./ai_ethics_llm_merged"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL)

print("Merging LoRA weights with base model...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(OUTPUT_MODEL)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_MODEL)

print(f"\nMerged model saved to {OUTPUT_MODEL}")
print("This model can now be loaded like any standard Hugging Face model.")

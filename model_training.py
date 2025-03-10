import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Load base model (choose a smaller model for resource efficiency)
model_name = "EleutherAI/pythia-1.4b"  # A relatively small model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    # Use the EOS token as the padding token if it exists
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Otherwise, add a new padding token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Resize the token embeddings since we added a new token
        model.resize_token_embeddings(len(tokenizer))

# Load your dataset
dataset = load_dataset('json', data_files='ai_ethics_dataset.jsonl')

# Add this after loading the dataset
print(f"Dataset size: {len(dataset['train'])}")
print(f"Sample entry: {dataset['train'][0]}")

# Tokenize the dataset
def tokenize_function(examples):
    # Format as instruction-response pairs
    texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples["instruction"], examples["response"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./ai_ethics_llm",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=5e-5,
    fp16=False,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./ai_ethics_llm_final")
tokenizer.save_pretrained("./ai_ethics_llm_final") 
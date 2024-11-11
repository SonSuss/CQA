from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
import torch
import torch.nn as nn
import sys
import os
from datasets import load_dataset

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'processing')))

from VisionEncoder import VisionTower, SiglipVisionEmbeddings, SiglipEncoderLayer


def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

model_name = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# print(model)
in_channels=3
hidden_size = 1152
patch_size = 14
num_patches = 2916  # Updated number of patches
num_layers = 27
mlp_dim = 4304
r = 20
# Replace the vision tower with the modified version
model.vision_tower._vision_tower.vision_model.embeddings =  SiglipVisionEmbeddings(in_channels, hidden_size, patch_size, num_patches)
model.vision_tower._vision_tower.vision_model.encoder.layers = nn.ModuleList([SiglipEncoderLayer(hidden_size, mlp_dim, r) for _ in range(num_layers)])

total_params = calculate_model_parameters(model)
print(f"Total number of parameters in the model: {total_params}")

#load dataset
# Load the ChartQA dataset
dataset = load_dataset("ahmed-masry/ChartQA")

def tokenize_function(examples):
    inputs = tokenizer(examples["question"], examples["chart"], truncation=True, padding="max_length")
    outputs = tokenizer(examples["answer"], truncation=True, padding="max_length")
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": outputs["input_ids"], "decoder_attention_mask": outputs["attention_mask"]}

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Create the data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Configure training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=100,
    save_strategy="epoch",  # Save the model at the end of each epoch
    learning_rate=1e-4,
    warmup_steps=int(0.03 * len(tokenized_datasets['train'])),
    weight_decay=0.01,
    evaluation_strategy="epoch",)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],

)

# Train the model
trainer.train()

# Save the model after each epoch
for epoch in range(1, 4):
    trainer.save_model(f'./model_{epoch}')
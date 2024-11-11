from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import torch.nn as nn
import sys
import os
from datasets import load_dataset

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'processing')))

from VisionEncoder import VisionTower, SiglipVisionEmbeddings, SiglipEncoderLayer

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


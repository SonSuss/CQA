from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import torch.nn as nn

model_name = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# class VisualTokenMerging(nn.Module):
#     def __init__(self, hidden_size):
#         super(VisualTokenMerging, self).__init__()
#         self.linear = nn.Linear(hidden_size, hidden_size)

#     def forward(self, x):
#         return self.linear(x)

# # Modify the vision tower to include the Visual Token Merging layer
# class ModifiedSiglipVisionModel(nn.Module):
#     def __init__(self, original_model):
#         super(ModifiedSiglipVisionModel, self).__init__()
#         self.original_model = original_model
#         self.add_visual_token_merging()

#     def add_visual_token_merging(self):
#         for layer in self.original_model.vision_model.encoder.layers:
#             layer.self_attn.visual_token_merging = VisualTokenMerging(layer.self_attn.q_proj.in_features)

#     def forward(self, *args, **kwargs):
#         return self.original_model(*args, **kwargs)

# # Replace the vision tower with the modified version
# model.vision_tower._vision_tower = ModifiedSiglipVisionModel(model.vision_tower._vision_tower)
# # Print the modified configuration for verification
# # print(model.config)
print(model.vision_tower)
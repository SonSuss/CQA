from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import sys
import os
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
import transformers
import requests
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'processing')))
from VisionEncoder import CustomSiglipEncoderLayer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..' )))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

model_name = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
config = model.config
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
model.tokenizer = tokenizer
model.config.use_cache = False

processor = AutoImageProcessor.from_pretrained(config.vision_model_name_or_path)
processor.size = {"height": 768, "width": 768}
raw_image = Image.open("E:\hocbaidcm\DACN\\test\CQA\data\\test\png\\166.png")
img_input = processor(images=raw_image, return_tensors='pt').to(device, torch.float16)
print(img_input)
text_img = tokenizer(text="Are the lines diverging?",image = img_input,text_target = "Yes",
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=128,
                                                    return_tensors="pt")
print(text_img)
print(tokenizer.decode(text_img['labels'][0], skip_special_tokens=True))

# inputs = processor(images=raw_image, return_tensors='pt').to(0, torch.float16)
# print(inputs)
# print(inputs['pixel_values'].shape)

config.vision_config = SiglipVisionConfig(hidden_act = "gelu_pytorch_tanh",
                                          hidden_size = 1152,
                                          image_size=768,
                                          intermediate_size=4304,
                                          layer_norm_eps= 1e-06,
                                          model_name_or_path = "siglip_vision_with_tome",
                                          num_attention_heads= 16,
                                          num_hidden_layers= 27,
                                          patch_size= 14)

r = 20
# print(config)
# Replace the vision tower with the modified version
model.vision_tower._vision_tower.vision_model.embeddings = SiglipVisionEmbeddings(config.vision_config)
model.vision_tower._vision_tower.vision_model.encoder.layers = nn.ModuleList([CustomSiglipEncoderLayer(config.vision_config, r) for _ in range(config.vision_config.num_hidden_layers)])

# print(model)
# Number of parameters in the model: 3217417280 384 14
# Number of parameters in the model: 3218070464 512 14
# Number of parameters in the model: 3219439040 768 16
# Number of parameters in the model: 3219936704 768 14
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters in the model: {total_params}")

# Root directory for images in train and validation sets
trainset = load_dataset('json', data_files=['E:/son/ChartQA Dataset/train/train_human.json', 'E:/son/ChartQA Dataset/train/train_augmented.json'], cache_dir="E:/son/", split='train')
valset = load_dataset('json', data_files=['E:/son/ChartQA Dataset/val/val_augmented.json','E:/son/ChartQA Dataset/val/val_human.json'], cache_dir="E:/son/", split='train')
train_image_root = 'E:/son/ChartQA Dataset/train/png/'
val_image_root = 'E:/son/ChartQA Dataset/val/png/'

def preprocess_function(examples,root):
    # Load and preprocess images
    images = [
        image_transform(Image.open(os.path.join(root, imgname)).convert("RGB"))
        for imgname in examples["imgname"]
    ]
    images = torch.stack(images).to(device)

    # Tokenize questions and labels directly using the tokenizer
    text_inputs = tokenizer(
        examples["query"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    label_inputs = tokenizer(
        examples["label"],  # Ground truth label
        padding="max_length",
        truncation=True,
        max_length=50,
        return_tensors="pt"
    )

    # Ensure the output includes all necessary keys
    model_inputs = {
        "pixel_values": images,  # Add images as pixel values
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "labels": label_inputs["input_ids"]
    }

    return model_inputs


# tokenized_trainsets = trainset.map(lambda x: preprocess_function(x, train_image_root), batched=True,batch_size=128)
# tokenized_valsets = valset.map(lambda x: preprocess_function(x, val_image_root), batched=True,batch_size=128)



# # Move the model to the GPU
# model.to(device)

# # Calculate the number of warmup steps
# num_train_steps = len(tokenized_trainsets) // 128 * 3  # Assuming 128 is the batch size and 3 is the number of epochs
# warmup_steps = int(0.03 * num_train_steps)

# # Configure training arguments
# training_args = TrainingArguments(
#     output_dir="./chartqa_results",
#     num_train_epochs=3,
#     per_device_train_batch_size=32,  # Adjust batch size as needed for GPU memory
#     warmup_steps=warmup_steps,
#     weight_decay=0.01,
#     learning_rate=1e-4,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     fp16=True,
# )
# # class CustomDataCollator(DataCollatorWithPadding):
# #     def __call__(self, features):
        
# #         # Separate pixel values from other features
# #         # pixel_values = torch.stack([feature["pixel_values"] for feature in features])
# #         # labels = torch.stack([feature["labels"] for feature in features])
        
# #         # Use the parent class to handle the rest
# #         batch = super().__call__([{
# #             "input_ids": feature["input_ids"],
# #             "attention_mask": feature["attention_mask"],
# #             "labels": feature["labels"],
# #             "pixel_values": feature["pixel_values"]  # Ensure pixel_values is included
# #         } for feature in features])
        
# #         # batch["pixel_values"] = pixel_values
        
# #         return batch

# # Use the custom data collator
# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")

# # Trainer setup
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_trainsets,
#     eval_dataset=tokenized_valsets,
#     data_collator=data_collator
# )


# # Train the model
# trainer.train()

# # Save the model after each epoch
# for epoch in range(1, 4):
#     trainer.save_model(f'./model_{epoch}')
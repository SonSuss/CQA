import os
import json
import argparse

from typing import List
from tqdm import tqdm

import torch

from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_deplot_text(text):
    return (
        text.replace("<0x0A>", "\n")
            .replace("\x0a", "\n")
            .replace("nan", "null")
            .strip()
    )

def load_deplot_model(model_name):
    processor = Pix2StructProcessor.from_pretrained(model_name)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    return model, processor

def load_image_batch(image_paths):
    """Load a batch of images from file paths"""
    images = []
    for path in image_paths:
        try:
            if os.path.exists(path):
                image = Image.open(path).convert('RGB')
                images.append(image)
            else:
                print(f"[WARNING] Image not found: {path}")
                images.append(None)
        except Exception as e:
            print(f"[ERROR] Failed to load image {path}: {e}")
            images.append(None)
    return images

def deplot_image(data_path, model, processor, batch_size, output_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    for i in tqdm(range(0,len(data),batch_size), desc="Running DePlot"):
        batch = data[i:i + batch_size]
        image_paths = []
        for entry in batch:
            image_filename = entry["image"]
            image_paths.append(image_filename)
        images=load_image_batch(image_paths)
        
        valid_data = []
        valid_images = []
        for img, entry in zip(images, batch):
            if img is not None:
                valid_images.append(img)
                valid_data.append(entry)
        
        if not valid_images:
            # If no valid images, add original entries without deplot text
            results.extend(batch)
            continue
        try:
            inputs = processor(images=valid_images, text="Generate underlying data table of the figure below:", return_tensors="pt", padding=True).to(DEVICE)
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            output_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
            for entry, deplot_text in zip(valid_data, output_texts):
                norm_text = normalize_deplot_text(deplot_text)
                original_query = entry["conversations"][0]["value"].replace("<image>\n", "")
                new_entry = {
                    "id": entry["id"],
                    "image": entry["image"],
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\nChart content:{norm_text}\n{original_query}"
                        },
                        {
                            "from": "gpt",
                            "value": entry["conversations"][1]["value"]
                        }
                    ]
                }
                results.append(new_entry)
            failed_entries = [entry for img, entry in zip(images, batch) if img is None]
            results.extend(failed_entries)

        except Exception as e:
            print(f"[ERROR] DePlot failed on batch {i}-{i+batch_size}: {e}")
            # Add original entries without modification if processing fails
            results.extend(batch)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Results saved to {output_path}")
    
    
# === MAIN ===
if __name__ == "__main__":
    model, processor = load_deplot_model('google/deplot')
    batch_size = 6


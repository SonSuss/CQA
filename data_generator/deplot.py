import os
import json
import argparse

from typing import List
from tqdm import tqdm

import torch

from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def images_collect(json_path)->list:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    collected_images = []
    seen = set()
    for entry in data:
        image_path = entry.get("image")
        if image_path and image_path not in seen:
            collected_images.append(image_path)
            seen.add(image_path)
    return collected_images

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

def deplot_image(images_list, model, processor, batch_size, output_path):
    results = []
    for i in tqdm(range(0, len(images_list), batch_size), desc="Running DePlot"):
        batch = images_list[i:i + batch_size]
        images = load_image_batch(batch)
        valid_data = []
        valid_images = []
        for img, image_path in zip(images, batch):
            if img is not None:
                valid_images.append(img)
                valid_data.append(image_path)
        if not valid_images:
            continue
        try:
            inputs = processor(images=valid_images, text="Generate underlying data table of the figure below:", return_tensors="pt", padding=True).to(DEVICE)
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            output_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for image_path, deplot_text in zip(valid_data, output_texts):
                norm_text = normalize_deplot_text(deplot_text)
                new_entry = {
                    "image": image_path,
                    "deplot": f"<image>\nChart content:{norm_text}\n"
                }
                results.append(new_entry)
            failed_entries = [entry for img, entry in zip(images, batch) if img is None]


        except Exception as e:
            print(f"[ERROR] DePlot failed on batch {i}-{i+batch_size}: {e}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open("data/failed_deplot.json", "w", encoding="utf-8") as f:
        json.dump(failed_entries, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Results saved to {output_path}, total {len(results)} entries.")
    print(f"Failed entries saved to data/failed_deplot.json, total {len(failed_entries)} entries.")
    
    
# === MAIN ===
if __name__ == "__main__":
    images_list = images_collect("CQA/data/train.json")
    model, processor = load_deplot_model('google/deplot')
    batch_size = 6
    output_path = "CQA/data/deplot.json"
    deplot_image(images_list, model, processor, batch_size, output_path)

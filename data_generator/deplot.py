import os
import json
import argparse
from typing import List
from tqdm import tqdm
import torch
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def images_collect(json_path) -> list:
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
    model.eval()
    return model, processor

def load_image(path):
    try:
        if os.path.exists(path):
            return Image.open(path).convert('RGB')
    except Exception as e:
        print(f"[ERROR] Failed to load image {path}: {e}")
    return None

def load_image_batch_parallel(image_paths: List[str]) -> List:
    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor(max_workers=16) as executor:
        images = list(executor.map(load_image, image_paths))
    return images

def deplot_image(images_list, model, processor, batch_size, output_path):
    results = []
    failed_entries = []

    for i in tqdm(range(0, len(images_list), batch_size), desc="Running DePlot"):
        batch = images_list[i:i + batch_size]
        images = load_image_batch_parallel(batch)

        valid_data = []
        valid_images = []
        for img, image_path in zip(images, batch):
            if img is not None:
                valid_images.append(img)
                valid_data.append(image_path)
            else:
                failed_entries.append(image_path)

        if not valid_images:
            continue

        try:
            with torch.no_grad():
                inputs = processor(
                    images=valid_images,
                    text=["Generate underlying data table of the figure below:"] * len(valid_images),
                    return_tensors="pt",
                    padding=True
                ).to(DEVICE)

                # Enable fp16 if supported
                with torch.amp.autocast(enabled=DEVICE == "cuda"):
                    generated_ids = model.generate(**inputs, max_new_tokens=512)

                output_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                for image_path, deplot_text in zip(valid_data, output_texts):
                    norm_text = normalize_deplot_text(deplot_text)
                    results.append({
                        "image": image_path,
                        "deplot": f"<image>\nChart content:{norm_text}\n"
                    })

        except Exception as e:
            print(f"[ERROR] DePlot failed on batch {i}-{i+batch_size}: {e}")
            failed_entries.extend(valid_data)  # Treat whole batch as failed if model crashed

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open("data/failed_deplot.json", "w", encoding="utf-8") as f:
        json.dump(failed_entries, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {output_path}")
    print(f"{len(failed_entries)} failed entries saved to data/failed_deplot.json")


# === MAIN ===
if __name__ == "__main__":
    images_list = images_collect("data/train.json")
    print(f"Images size: {len(images_list)}")
    model, processor = load_deplot_model('google/deplot')
    batch_size = 40  # Increase if GPU memory allows
    output_path = "data/train_deplot.json"
    deplot_image(images_list, model, processor, batch_size, output_path)

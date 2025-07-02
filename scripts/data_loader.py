import json
import os

from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

def save_dataset(dataset, json_filename, images_folder):
    os.makedirs(images_folder, exist_ok=True)
    json_data = []

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Saving dataset"):
        # Save image
        image = item['image']
        image_filename = f"{idx}.png"
        image_path = os.path.join(images_folder, image_filename)
        image.save(image_path)

        entry = {
            "id": "chartQA_" + str(idx),
            "image": str(image_path),
            "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + item['query']
            },
            {
                "from": "gpt",
                "value": item['label'][0]
            }
        ]
        }
        json_data.append(entry)

    # Save JSON file
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print("Data length for " + json_filename + ": ", len(json_data))

if __name__ == "__main__":
    dataset = load_dataset("HuggingFaceM4/ChartQA")
    images_folder = "data/images"
    data_folder = "data"
    for split in ["train", "val", "test"]:
        split_set = dataset[split]
        json_filename = f"{data_folder}/{split}.json"
        save_dataset(split_set, json_filename, images_folder)


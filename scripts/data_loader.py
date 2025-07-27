import json
import os
import requests
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# def save_dataset(dataset, json_filename, images_folder):
#     os.makedirs(images_folder, exist_ok=True)
#     json_data = []

#     for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Saving dataset"):
#         # Save image
#         image = item['image']
#         image_filename = f"{idx}.png"
#         image_path = os.path.join(images_folder, image_filename)
#         image.save(image_path)

#         entry = {
#             "id": "chartQA_" + str(idx),
#             "image": str(image_path),
#             "conversations": [
#             {
#                 "from": "human",
#                 "value": "<image>\n" + item['query']
#             },
#             {
#                 "from": "gpt",
#                 "value": item['label'][0]
#             }
#         ]
#         }
#         json_data.append(entry)

#     # Save JSON file
#     with open(json_filename, 'w', encoding='utf-8') as f:
#         json.dump(json_data, f, ensure_ascii=False, indent=2)
#     print("Data length for " + json_filename + ": ", len(json_data))

# if __name__ == "__main__":
#     dataset = load_dataset("HuggingFaceM4/ChartQA")
#     images_folder = "data/images"
#     data_folder = "data"
#     for split in ["train", "val", "test"]:
#         split_set = dataset[split]
#         json_filename = f"{data_folder}/{split}.json"
#         save_dataset(split_set, json_filename, images_folder)




def data_preprocess_for_chart_QA(data_path,output_path):
    input_path = os.path.join(data_path, "ChartQA Dataset")
    output_folder = os.path.join(data_path, output_path)
    os.makedirs(output_folder, exist_ok=True)
    src_data = ["train", "test", "val"]
    val_set, train_set = [], []
    image_track = {}
    
    print("Copying images (avoiding duplicates)...")
    total_copied = 0
    total_skipped = 0
    
    for name in src_data:
        src_folder = os.path.join(input_path, name)
        src_img_folder = os.path.join(src_folder, "png")
        if not os.path.exists(src_img_folder):
            continue
            
        for img in os.listdir(src_img_folder):
            img_path = os.path.join(src_img_folder, img)
            img_id = os.path.splitext(img)[0]
            
            if os.path.isfile(img_path):
                image_track[img_id] = img_path
    
    print(f"Images processed: {total_copied} copied, {total_skipped} skipped")

    print("Processing JSON data...")
    for name in src_data:
        name_path = os.path.join(input_path, name)
        data = []
        
        # Load augmented and human data
        aug_file = f"{name_path}/{name}_augmented.json"
        human_file = f"{name_path}/{name}_human.json"
        
        if os.path.exists(aug_file):
            with open(aug_file, "r", encoding="utf-8") as f:
                data += json.load(f)
        
        if os.path.exists(human_file):
            with open(human_file, "r", encoding="utf-8") as f:
                data += json.load(f)
        
        print(f"Processing {len(data)} items from {name} split...")
        
        if name == 'val':
            for item in data:
                img_id = os.path.splitext(item['imgname'])[0]
                if img_id in image_track:
                    entry = {
                        "id": img_id,
                        "image": image_track[img_id],
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<|image|>\n" + "Question:\n" + item['query']
                            },
                            {
                                "from": "gpt",
                                "value": item['label']
                            }
                        ]
                    }
                    val_set.append(entry)
        else:
            for item in data:
                img_id = os.path.splitext(item['imgname'])[0]
                if img_id in image_track:
                    entry = {
                        "id": img_id,
                        "image": image_track[img_id],
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<|image|>\n" + "Question:\n" + item['query']
                            },
                            {
                                "from": "gpt",
                                "value": item['label']
                            }
                        ]
                    }
                    train_set.append(entry)

    train_path = os.path.join(output_folder, "train.json")
    val_path = os.path.join(output_folder, "val.json")

    print("Saving JSON files...")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=2)
    
    # Final results
    print(f"Total images: {len(image_track)}")
    if train_set:
        print(f"Sample entry: {train_set[0]}")
    print(f"Train_set len: {len(train_set)}")
    print(f"Val_set len: {len(val_set)}")

def chartqa_chart_to_table_addition(data_path, max_workers=8):
    chartqa_path= os.path.join(data_path, "ChartQA Dataset")
    preprocessed_folder = os.path.join(data_path, "processed_data")
    non_table_train_path = os.path.join(preprocessed_folder, "train.json")
    non_table_val_path = os.path.join(preprocessed_folder, "val.json")
    with open(non_table_train_path, "r", encoding="utf-8") as f:
        train_set = json.load(f)
    with open(non_table_val_path, "r", encoding="utf-8") as f:
        val_set = json.load(f)
    src_data = ["train", "test", "val"]
    for src in src_data:
        src_folder = os.path.join(chartqa_path, src)
        img_folder = os.path.join(src_folder, "png")
        table_folder = os.path.join(src_folder, "tables")
        img_files = [f for f in os.listdir(img_folder) if f.endswith('.png')]

        def process_img(img_file):
            img_id = os.path.splitext(img_file)[0]
            csv_path = os.path.join(table_folder, f"{img_id}.csv")
            img_path = os.path.join(img_folder, img_file)
            if not os.path.exists(csv_path):
                return None
            try:
                with open(csv_path, 'r', encoding='utf-8') as csv_file:
                    csv_data = csv_file.read()
                entry = {
                    "id": img_id,
                    "image": img_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<|image|>\nQuestion:\nGenerate the table data in CSV format based on the chart in the image."
                        },
                        {
                            "from": "gpt",
                            "value": csv_data
                        }
                    ]
                }
                return entry
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            entries = list(tqdm(executor.map(process_img, img_files), total=len(img_files), desc=f"Processing {src} images"))
            entries = [e for e in entries if e is not None]

        if src == 'val':
            val_set.extend(entries)
        else:
            train_set.extend(entries)
    output_folder = os.path.join(data_path, "preprocessed_data_with_tables")
    os.makedirs(output_folder, exist_ok=True)
    train_path = os.path.join(output_folder, "train.json")
    val_path = os.path.join(output_folder, "val.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=2)
    print(f"Train_set len: {len(train_set)}")
    print(f"Val_set len: {len(val_set)}")
    
def download_and_extract(url, extract_path="data/"):
    try:
        # Download
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save temporarily
        temp_filename = "temp_download.zip"
        with open(temp_filename, 'wb') as f:
            f.write(response.content)
        
        # Extract
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(temp_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Clean up temp file
        os.remove(temp_filename)
        
        print(f"Successfully downloaded and extracted to {extract_path}")
        
                
    except Exception as e:
        print(f"Error: {e}")

def check_corresponding_files(annotations_path, tables_path, output_path):

    json_files = os.listdir(annotations_path)
    json_names = [f.replace('.json', '') for f in json_files if f.endswith('.json')]
    matching_names = []
    missing_csv = []
    
    for name in json_names:
        csv_path = f'{tables_path}/{name}.csv'
        if os.path.exists(csv_path):
            matching_names.append(name)
        else:
            missing_csv.append(name)
    
    print(f"Total JSON files: {len(json_names)}")
    print(f"Matching pairs: {len(matching_names)}")
    print(f"Missing CSV files: {len(missing_csv)}")
    
    if missing_csv:
        print(f"Files without CSV: {missing_csv}")
    
    # Save matching names to names.json
    names_dict = [name for name in matching_names]
    print(matching_names[0])
    with open(output_path, 'w') as f:
        json.dump(names_dict, f, indent=2)


def __main__():
    url = "https://huggingface.com/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
    download_and_extract(url)
    json_folder = "data/extracted_files/ChartQA Dataset/train/annotations"
    csv_folder = "data/extracted_files/ChartQA Dataset/train/tables"
    check_corresponding_files(json_folder, csv_folder, 'train_names.json')


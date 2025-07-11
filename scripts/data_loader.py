import json
import os
import requests
import zipfile
import shutil

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
    image_path = os.path.join(output_folder,"images")
    os.makedirs(image_path, exist_ok=True)
    src_data_img = ["train", "test", "val"]
    for name in src_data_img:
        src_folder = os.path.join(input_path, name)
        src_folder = os.path.join(src_folder, "png")
        for img in os.listdir(src_folder):
            img_path = os.path.join(src_folder, img)
            out_path = os.path.join(image_path, img)
            if os.path.isfile(img_path):
                try:
                    shutil.copy2(img_path, out_path)
                except FileExistsError:
                    print(f"Skipped (exists): {img_path}")
    print(len(os.listdir(image_path)))

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

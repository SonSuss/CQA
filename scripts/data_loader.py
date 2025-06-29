import requests
import zipfile
import os

from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import hf_hub_download

repo_id = "mPLUG/TinyChartData"
filenames = [
    "/tinychart_images.tar",
    "/train.json",
    "/test.json"
]

for filename in filenames:
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        hf_hub_download(repo_id=repo_id, filename=filename, cache_dir='./data')
        
if __name__ == "__main__":
    pass
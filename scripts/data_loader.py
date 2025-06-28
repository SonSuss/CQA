import requests
import zipfile
import os

from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import hf_hub_download

json_path = hf_hub_download(repo_id="mPLUG/TinyChartData",cache_dir='./data')

if __name__ == "__main__":
    pass
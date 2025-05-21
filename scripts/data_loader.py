import requests
import zipfile
import os
from tqdm import tqdm  # For progress bar

def download_and_extract(url, output_dir):
    """
    Downloads a file from the given URL and extracts it if it's a zip file.

    Args:
        url (str): The URL to download the file from.
        output_dir (str): The directory to save and extract the file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # File name from URL
    file_name = url.split("/")[-1]
    file_path = os.path.join(output_dir, file_name)

    # Download the file with a progress bar
    print(f"Downloading {file_name}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, "wb") as file, tqdm(
            desc=file_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                bar.update(len(chunk))
        print(f"Downloaded {file_name} to {file_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return

    # Extract if it's a zip file
    if file_name.endswith(".zip"):
        print(f"Extracting {file_name}...")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted to {output_dir}")

        # Optionally, delete the zip file after extraction
        os.remove(file_path)
        print(f"Deleted the zip file: {file_path}")

# Example usage
if __name__ == "__main__":
    urls = [
        "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
        "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
    ]
    output_dir = "./data"
    for url in urls:
        download_and_extract(url, output_dir)
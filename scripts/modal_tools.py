import modal

app = modal.App("LoadChartQA")

# Create or attach a persistent volume
volume = modal.Volume.from_name("chartqa-A100-llava-siglip-phi4_6", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("requests", "datasets", "tqdm", "pillow", "gitpython")
    .run_commands([
        "apt-get update && apt-get install -y git",
        "git clone https://github.com/SonSuss/CQA.git /root/CQA",
        "cd /root/CQA && git config --global --add safe.directory /root/CQA",
    ])
    .workdir("/root/CQA")
)

@app.function(
    image=image,
    volumes={"/root/data": volume},
    timeout=1800,  # 30 minutes
    cpu=1,         # Default CPU - no waste
    memory=1024    # Default memory - minimal cost
)
def remove_folder():
    import os
    import shutil
    root = "/root/data"
    remove_lst = ["eval_results",
                  "siglip_-1-resampler2_768_128_3-phi4_1_plus"]
    
    for folder in remove_lst:
        folder_path = os.path.join(root, folder)
        if os.path.exists(folder_path):
            print(f"Removing folder: {folder_path}")
            shutil.rmtree(folder_path)
        else:
            print(f"Folder does not exist: {folder_path}")
            
@app.function(
    image=image,
    volumes={"/root/data": volume},
    timeout=1800,  # 30 minutes
    cpu=1,         # Default CPU - no waste
    memory=1024    # Default memory - minimal cost
)
def coppy_file_to_folder():
    import os
    import shutil
    src_file="checkpoint-siglip_-1-resampler2_768_128_3-phi4_1_plus/answers_0.2_0.5.json"
    dest_folder="eval_results/siglip_-1-resampler2_768_128_3-phi4_1_plus"
    root = "/root/data"
    src_path = os.path.join(root, src_file)
    dest_path = os.path.join(root, dest_folder)
    if not os.path.isfile(src_path):
        print(f"Source file does not exist: {src_path}")
        return

    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)

    dest_file_path = os.path.join(dest_path, os.path.basename(src_file))
    shutil.copy2(src_path, dest_file_path)
    print(f"Copied {src_path} to {dest_file_path}")
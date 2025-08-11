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
    remove_lst = ["checkpoint-siglip_-1-resampler_768_256_3-phi4_init_testsave"]
    
    for folder in remove_lst:
        folder_path = os.path.join(root, folder)
        if os.path.exists(folder_path):
            print(f"Removing folder: {folder_path}")
            shutil.rmtree(folder_path)
        else:
            print(f"Folder does not exist: {folder_path}")
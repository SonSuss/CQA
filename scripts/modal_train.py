import modal

app = modal.App("TrainChartQA")

# Create or attach a persistent volume
volume = modal.Volume.from_name("chartqa", create_if_missing=True)

# Define image with dependencies
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

def pull_latest_code():
    """Pull the latest code from Git repository"""
    import subprocess
    import os
    
    try:
        # Change to repo directory
        os.chdir("/root/CQA")
        
        # Pull latest changes
        result = subprocess.run(["git", "pull", "origin", "main"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Git pull successful: {result.stdout}")
        
        # Show current commit
        commit_result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], 
                                     capture_output=True, text=True, check=True)
        print(f"📌 Current commit: {commit_result.stdout.strip()}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git pull failed: {e.stderr}")
        print("Continuing with existing code...")

@app.function(
    image=image,
    volumes={"/root/data": volume},
    timeout=600  # 10 minutes
)
def download_datasets():
    # Pull latest code first
    pull_latest_code()
    
    from scripts.data_loader import download_and_extract, check_corresponding_files

    url = "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
    extract_path = "/root/data/extracted_files"

    download_and_extract(url, extract_path=extract_path)

    json_folder = f"{extract_path}/ChartQA Dataset/train/annotations"
    csv_folder = f"{extract_path}/ChartQA Dataset/train/tables"
    output_json_path = "/root/data/train_names.json"

    check_corresponding_files(json_folder, csv_folder, output_json_path)
    volume.commit()

@app.function(
    image=image,
    timeout=300
)
def update_code():
    """Dedicated function to update code and show current status"""
    pull_latest_code()
    
    # Show file tree to verify update
    import os
    def show_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        try:
            items = sorted(os.listdir(path))
            for i, item in enumerate(items):
                if item.startswith('.'):
                    continue
                item_path = os.path.join(path, item)
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item}")
                if os.path.isdir(item_path) and current_depth < max_depth - 1:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    show_tree(item_path, next_prefix, max_depth, current_depth + 1)
        except PermissionError:
            pass
    
    print("\n📁 Repository structure:")
    show_tree("/root/CQA")
    
    # Show recent commits
    import subprocess
    try:
        result = subprocess.run(["git", "log", "--oneline", "-5"], 
                              capture_output=True, text=True, check=True)
        print(f"\n📝 Recent commits:\n{result.stdout}")
    except subprocess.CalledProcessError:
        print("Could not fetch recent commits")

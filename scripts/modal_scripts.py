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
    timeout=1800,  # 30 minutes
    cpu=1,         # Default CPU - no waste
    memory=1024    # Default memory - minimal cost
)
def download_datasets():
    # Pull latest code first
    pull_latest_code()
    
    from scripts.data_loader import download_and_extract, data_preprocess_for_chart_QA
    import time

    url = "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
    extract_path = "/root/data/Chart_QA"

    print("⏱️ Starting download...")
    start_time = time.time()
    download_and_extract(url, extract_path)
    download_time = time.time() - start_time
    print(f"✅ Download completed in {download_time:.2f} seconds")

    print("⏱️ Starting data preprocessing...")
    preprocess_start = time.time()
    data_preprocess_for_chart_QA(extract_path, "processed_data")
    preprocess_time = time.time() - preprocess_start
    print(f"✅ Preprocessing completed in {preprocess_time:.2f} seconds")

    volume.commit()
    print(f"📊 Total processing time: {(time.time() - start_time):.2f} seconds")

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
                
                # Special handling for images folders
                if os.path.isdir(item_path) and item.lower() in ['images', 'image', 'png', 'jpg', 'jpeg']:
                    try:
                        image_files = [f for f in os.listdir(item_path) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
                        print(f"{prefix}{current_prefix}{item}/ ({len(image_files)} images)")
                    except PermissionError:
                        print(f"{prefix}{current_prefix}{item}/ (images - permission denied)")
                elif os.path.isdir(item_path):
                    print(f"{prefix}{current_prefix}{item}/")
                    if current_depth < max_depth - 1:
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        show_tree(item_path, next_prefix, max_depth, current_depth + 1)
                else:
                    print(f"{prefix}{current_prefix}{item}")
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

@app.function(
    image=image,
    volumes={"/root/data": volume},
    timeout=600,   # 10 minutes
    cpu=1,         # Default CPU - no waste
    memory=1024    # Default memory - minimal cost
)
def download_only():
    """Fast download without preprocessing"""
    pull_latest_code()
    
    from scripts.data_loader import download_and_extract
    import time

    url = "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
    extract_path = "/root/data/Chart_QA"

    print("⏱️ Starting download...")
    start_time = time.time()
    download_and_extract(url, extract_path)
    download_time = time.time() - start_time
    print(f"✅ Download completed in {download_time:.2f} seconds")

    volume.commit()
    print("📁 Data downloaded and ready for preprocessing")

@app.function(
    image=image,
    volumes={"/root/data": volume},
    timeout=1200,  # 20 minutes
    cpu=1,         # Default CPU - no waste for I/O bound tasks
    memory=2048    # Increase to 2GB for safety
)
def preprocess_only():
    """Separate preprocessing step with progress tracking"""
    pull_latest_code()
    
    import time
    import os
    
    extract_path = "/root/data/Chart_QA"
    
    # Check if data exists
    input_path = os.path.join(extract_path, "ChartQA Dataset")
    if not os.path.exists(input_path):
        print("❌ Data not found. Run download_only() first.")
        return
    
    print("⏱️ Starting optimized preprocessing...")
    start_time = time.time()
    
    # Use the original function but with timing
    from scripts.data_loader import data_preprocess_for_chart_QA
    data_preprocess_for_chart_QA(extract_path, "processed_data")
    
    preprocess_time = time.time() - start_time
    print(f"✅ Preprocessing completed in {preprocess_time:.2f} seconds")

    volume.commit()




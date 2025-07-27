import modal

app = modal.App("LoadChartQA")

# Create or attach a persistent volume
volume = modal.Volume.from_name("chartqa-A100-llava-siglip-phi4_5", create_if_missing=True)

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
        print(f"Git pull successful: {result.stdout}")
        
        # Show current commit
        commit_result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], 
                                     capture_output=True, text=True, check=True)
        print(f"Current commit: {commit_result.stdout.strip()}")

    except subprocess.CalledProcessError as e:
        print(f"Git pull failed: {e.stderr}")
        print("Continuing with existing code...")

@app.function(
    image=image,
    volumes={"/root/data": volume},
    timeout=1800,  # 30 minutes
    cpu=1,         # Default CPU - no waste
    memory=1024    # Default memory - minimal cost
)
def download_preprocess_datasets():
    pull_latest_code()
    from scripts.data_loader import download_and_extract, data_preprocess_for_chart_QA, chartqa_chart_to_table_addition
    import time
    url = "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
    extract_path = "/root/data/Chart_QA"

    download_and_extract(url, extract_path)

    print("Starting data preprocessing...")
    data_preprocess_for_chart_QA(extract_path, "processed_data")
    print("Data preprocessing completed.")
    chartqa_chart_to_table_addition(extract_path)

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
    timeout=600,  
    cpu=1, 
    memory=1024    
)
def download_only():
    """Fast download without preprocessing"""
    pull_latest_code()
    
    from scripts.data_loader import download_and_extract

    url = "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
    extract_path = "/root/data/Chart_QA"

    download_and_extract(url, extract_path)

    volume.commit()
    print("📁 Data downloaded and ready for preprocessing")

@app.function(
    image=image,
    volumes={"/root/data": volume},
    timeout=1200, 
    cpu=1,  
    memory=1024 
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
        print("Data not found. Run download_only() first.")
        return
    
    print("Starting optimized preprocessing...")
    start_time = time.time()
    
    # Use the original function but with timing
    from scripts.data_loader import data_preprocess_for_chart_QA
    data_preprocess_for_chart_QA(extract_path, "processed_data")
    
    preprocess_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocess_time:.2f} seconds")

    volume.commit()


@app.function(
    image=image,
    volumes={"/root/data": volume},
    timeout=300,
    cpu=1,
    memory=1024
)
def get_training_status():
    """Get current training status and live logs"""
    import os
    import glob
    
    checkpoint_dir = "/root/data/checkpoints"
    
    status = {
        "training_active": False,
        "latest_checkpoint": None,
        "total_checkpoints": 0,
        "log_tail": ""
    }
    
    if os.path.exists(checkpoint_dir):
        # Find checkpoints
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
        status["total_checkpoints"] = len(checkpoints)
        
        if checkpoints:
            # Get latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
            status["latest_checkpoint"] = os.path.basename(latest_checkpoint)
        
        # Read latest log entries
        log_file = os.path.join(checkpoint_dir, "train.log")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    status["log_tail"] = ''.join(lines[-10:])  # Last 10 lines
            except Exception as e:
                status["log_tail"] = f"Error reading log: {e}"
    
    print(f"🎯 Training Status:")
    print(f"Total checkpoints: {status['total_checkpoints']}")
    print(f"Latest checkpoint: {status['latest_checkpoint']}")
    
    if status["log_tail"]:
        print(f"\n📋 Recent log entries:")
        print(status["log_tail"])
    
    return status

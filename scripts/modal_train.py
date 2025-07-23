import modal

app = modal.App("TrainChartQA")

# Create or attach a persistent volume
volume = modal.Volume.from_name("chartqa-A100-llava-siglip-phi4_2", create_if_missing=True)


cuda_version = "12.6.0"
flavor = "devel" 
operating_sys = "ubuntu22.04"
tag = f"nvidia/cuda:{cuda_version}-{flavor}-{operating_sys}"
gpu = "A100-40GB"

training_image = (
    modal.Image.from_registry(tag, add_python="3.11")
    .pip_install("bitsandbytes", gpu=gpu)
    .pip_install(
        [
            "torch", "torchvision", "torchaudio",
            "transformers==4.48.3",
            "accelerate==1.7.0", 
            "bitsandbytes==0.46.0",
            "safetensors==0.4.5",
            "huggingface-hub==0.32.4",
            "einops==0.8.0",
            "peft==0.14.0",
            "deepspeed==0.17.0",
            "hf-xet==1.1.3",
            "shortuuid==1.0.13",
            "datasets", "tokenizers", "sentencepiece", "protobuf",
            "requests", "tqdm", "pillow", "gitpython", "tensorboard",
            "psutil", 
        ],
    )
    .pip_install( 
        "flash-attn==2.8.1", extra_options="--no-build-isolation"
    )
    .run_commands([
        "apt-get update && apt-get install -y git build-essential",
        "git clone https://github.com/SonSuss/CQA.git /root/CQA",
    ])
    .workdir("/root/CQA")
    .env({"PYTHONPATH": "/root/CQA"})
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

# Training configuration constants
MINUTES = 60
TRAIN_GPU = gpu
TRAIN_CPU_COUNT = (1.0,8.0)
TRAIN_MEMORY_GB = (8 * 1024,32 * 1024)  # 8GB to 32GB
TRAIN_TIME = 10 # hours
CHECKPOINT = "/root/data/checkpoints-siglip-mlp2x_gelu-phi4"

@app.function(
    image=training_image,
    gpu=TRAIN_GPU,
    timeout= 60 * MINUTES,
    cpu=TRAIN_CPU_COUNT
)
def check_gpu_info():
    pull_latest_code()
    """Check GPU, system, and library information for training environment"""
    import torch
    import sys
    import platform
    import pkg_resources
    import os
    
    print("🔍 COMPLETE TRAINING ENVIRONMENT INFO")
    print("=" * 60)
    
    # =============================================================================
    # SYSTEM INFORMATION
    # =============================================================================
    print("\n🖥️  SYSTEM INFORMATION:")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    # CPU Information
    try:
        cpu_count = os.cpu_count()
        print(f"CPU Cores: {cpu_count}")
    except:
        print("CPU Cores: Unable to detect")
    
    # Memory Information
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / 1024**3:.2f} GB")
        print(f"Available RAM: {memory.available / 1024**3:.2f} GB")
        print(f"RAM Usage: {memory.percent}%")
    except ImportError:
        print("RAM Info: psutil not available")
    
    # =============================================================================
    # GPU INFORMATION
    # =============================================================================
    print("\n🎯 GPU INFORMATION:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPU count: {device_count}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\n📊 GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total VRAM: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multi-processor count: {props.multi_processor_count}")
            print(f"  CUDA capability: {props.major}.{props.minor}")
            print(f"  Memory bandwidth: ~{props.total_memory * 1000 // (1024**3)} GB/s")
            
            # Check current memory usage
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3
            
            print(f"  Allocated VRAM: {allocated:.2f} GB")
            print(f"  Cached VRAM: {cached:.2f} GB") 
            print(f"  Free VRAM: {free:.2f} GB")
            print(f"  Memory efficiency: {(free/props.total_memory*1024**3)*100:.1f}% available")
    else:
        print("❌ No CUDA devices found")
    
    # =============================================================================
    # CRITICAL TRAINING LIBRARIES
    # =============================================================================
    print("\n📚 CRITICAL TRAINING LIBRARIES:")
    
    critical_libs = [
        'torch', 'torchvision', 'torchaudio',
        'transformers', 'accelerate', 'peft', 
        'bitsandbytes', 'datasets', 'tokenizers',
        'safetensors', 'huggingface-hub', 'einops',
        'deepspeed', 'flash-attn', 'tensorboard'
    ]
    
    lib_info = {}
    for lib in critical_libs:
        try:
            version = pkg_resources.get_distribution(lib).version
            print(f"  ✅ {lib}: {version}")
            lib_info[lib] = version
        except pkg_resources.DistributionNotFound:
            print(f"  ❌ {lib}: NOT INSTALLED")
            lib_info[lib] = "NOT INSTALLED"
        except Exception as e:
            print(f"  ⚠️  {lib}: Error - {e}")
            lib_info[lib] = f"Error: {e}"
    
    # =============================================================================
    # PYTORCH CAPABILITIES
    # =============================================================================
    print("\n🔥 PYTORCH CAPABILITIES:")
    print(f"PyTorch version: {torch.__version__}")
    try:
        import torchvision
        print(f"TorchVision version: {torchvision.__version__}")
    except:
        print(f"TorchVision version: Unknown")
    print(f"Mixed precision support: {torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7}")
    print(f"Flash Attention available: {'flash_attn' in lib_info and lib_info['flash_attn'] != 'NOT INSTALLED'}")
    print(f"Distributed training support: {torch.distributed.is_available()}")
    print(f"Automatic Mixed Precision: {hasattr(torch.cuda.amp, 'autocast')}")
    
    # Test tensor operations
    try:
        if torch.cuda.is_available():
            test_tensor = torch.randn(1000, 1000, device='cuda')
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            result = torch.matmul(test_tensor, test_tensor)
            end_time.record()
            torch.cuda.synchronize()
            
            elapsed = start_time.elapsed_time(end_time)
            print(f"GPU compute test (1000x1000 matmul): {elapsed:.2f}ms")
            
            del test_tensor, result
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"GPU compute test failed: {e}")
    
    # =============================================================================
    # ENVIRONMENT VARIABLES
    # =============================================================================
    print("\n🌍 RELEVANT ENVIRONMENT VARIABLES:")
    env_vars = [
        'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF', 
        'NCCL_DEBUG', 'OMP_NUM_THREADS', 'TOKENIZERS_PARALLELISM',
        'TRANSFORMERS_CACHE', 'HF_HOME', 'WANDB_DISABLED'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # =============================================================================
    # TRAINING READINESS CHECK
    # =============================================================================
    print("\n✅ TRAINING READINESS CHECK:")
    
    checks = {
        "CUDA Available": torch.cuda.is_available(),
        "GPU Memory > 40GB": torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 40 * 1024**3,
        "Mixed Precision Support": torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7,
        "Transformers Available": 'transformers' in lib_info and lib_info['transformers'] != 'NOT INSTALLED',
        "Accelerate Available": 'accelerate' in lib_info and lib_info['accelerate'] != 'NOT INSTALLED',
        "PEFT Available": 'peft' in lib_info and lib_info['peft'] != 'NOT INSTALLED',
        "Flash Attention Available": 'flash-attn' in lib_info and lib_info['flash-attn'] != 'NOT INSTALLED',
        "BitsAndBytes Available": 'bitsandbytes' in lib_info and lib_info['bitsandbytes'] != 'NOT INSTALLED',
    }
    
    all_ready = True
    for check, status in checks.items():
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {check}")
        if not status:
            all_ready = False
    
    print(f"\n🎯 OVERALL TRAINING READINESS: {'✅ READY' if all_ready else '❌ ISSUES DETECTED'}")
    
    # =============================================================================
    # RETURN STRUCTURED DATA
    # =============================================================================
    return {
        "system_info": {
            "os": f"{platform.system()} {platform.release()}",
            "architecture": platform.machine(),
            "python_version": sys.version,
            "cpu_cores": os.cpu_count(),
        },
        "gpu_info": {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_vram_gb": torch.cuda.get_device_properties(i).total_memory / 1024**3,
                    "cuda_capability": f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}",
                    "multiprocessor_count": torch.cuda.get_device_properties(i).multi_processor_count,
                    "allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                    "free_gb": (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1024**3
                }
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else []
        },
        "library_versions": lib_info,
        "training_readiness": checks,
        "overall_ready": all_ready
    }
    
    
@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    gpu=TRAIN_GPU,
    cpu=TRAIN_CPU_COUNT,
    memory=TRAIN_MEMORY_GB,
    timeout=60 * MINUTES,
)
def preload_models():
    import os
    import torch
    from models.chart_qa_model.model.modeling_phi3 import Phi3ForCausalLM
    from models.chart_qa_model.model.configuration_phi3 import Phi3Config
    from models.components.vision_towers.siglip_tome.siglip_tome import SigLipVisionTower, SigLipVisionConfig
    from transformers import AutoTokenizer
    cache_dir= "/root/data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    model_name="microsoft/Phi-4-mini-instruct"
    
    print("  📝 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        model_max_length=1024,  # Same as training
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    print("  ⚙️ Downloading model config...")
    config = Phi3Config.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    print("  🧠 Downloading Phi-4 model weights...")
    phi_model = Phi3ForCausalLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,  # Same as training
        device_map="cpu",  # Keep on CPU during download
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Same as training
    )
    del phi_model
    torch.cuda.empty_cache()
    
    volume.commit()

@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    gpu=TRAIN_GPU,
    cpu=4.0,  # Fixed CPU value 
    memory=16 * 1024,  # Fixed memory value (16GB)
    timeout= TRAIN_TIME * 60 * MINUTES,
)
def train_chartqa():
    pull_latest_code()
    import os
    from models.chart_qa_model.train.train import train
    from models.components.config import ModelArguments, DataArguments, TrainingArguments
    
    print("🔍 Checking paths and configuration...")
    
    # Check if we're in the right working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Define paths for Modal environment
    data_path = "/root/data/Chart_QA/processed_data/train.json"
    eval_data_path = "/root/data/Chart_QA/processed_data/val.json"
    output_dir = CHECKPOINT
    cache_dir = "/root/data/cache"
    
    # Check if paths exist
    paths_to_check = {
        "Data file": data_path,
        "Eval data file": eval_data_path,
    }
    
    print("\n📁 Path validation:")
    missing_paths = []
    
    for desc, path in paths_to_check.items():
        if os.path.exists(path):
            print(f"  ✅ {desc}: {path}")
        else:
            print(f"  ❌ {desc}: {path}")
            missing_paths.append(f"{desc}: {path}")
    
    # Raise error if any paths are missing
    if missing_paths:
        error_msg = f"❌ Missing required paths:\n" + "\n".join(f"  - {path}" for path in missing_paths)
        print(f"\n{error_msg}")
        raise FileNotFoundError(f"Required paths do not exist: {missing_paths}")
    
    #model config
    model_args = ModelArguments(
        model_name_or_path="microsoft/Phi-4-mini-instruct",
        version="phi4_instruct",
        freeze_backbone=True,
        tune_mm_mlp_adapter=False,
        vision_tower="mPLUG/TinyChart-3B-768-siglip",
        mm_vision_select_layer=-2,
        pretrain_mm_mlp_adapter=None,
        mm_projector_type="mlp2x_gelu",
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        mm_patch_merge_type="flat",
        mm_vision_select_feature="patch",
        resampler_hidden_size=768,
        num_queries=128,
        num_resampler_layers=3,
        tune_vision_tower=True,
        tune_entire_model=False,
        tune_vit_from_layer=-1,
        tune_embed_tokens=False,
    )

    data_args = DataArguments(
        data_path=data_path,
        eval_data_path=eval_data_path,
        lazy_preprocess=True,
        is_multimodal=True,
        image_folder="",  # Adjusted for volume
        image_aspect_ratio="square",
    )

    # Test configuration with smaller values for quick validation
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3, 
        per_device_train_batch_size=7,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=1000, 
        save_total_limit=1,
        mm_projector_lr=5e-5,
        vision_tower_lr=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=25,
        fp16=False,
        bf16=True,
        model_max_length=2048,
        gradient_checkpointing=True,
        dataloader_num_workers=12, 
        dataloader_persistent_workers=True,
        report_to="tensorboard",
        cache_dir=cache_dir,
        optim="adamw_torch_fused",
        bits=16,
        group_by_modality_length=True,
        warmup_steps=150,
        max_grad_norm=0.5,
        local_rank=-1,  # For single GPU
    )
    train(model_args, data_args, training_args, log_rewrite=True)
    
    volume.commit()
    
@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    timeout=15 * MINUTES,
)
def remove_old_checkpoints():
    import os
    import shutil
    import glob
    
    checkpoint_base = CHECKPOINT
    if not os.path.exists(checkpoint_base):
        print(f"📁 No checkpoint directory found at: {checkpoint_base}")
        return {"status": "no_checkpoints", "removed": 0, "space_saved_gb": 0}
    checkpoint_pattern = os.path.join(checkpoint_base, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    if not checkpoint_dirs:
        print("Checkpoint directory exists but no checkpoint-* folders found")
        return {"status": "empty_dir", "removed": 0, "space_saved_gb": 0}
    # Sort by step number (checkpoint-1000, checkpoint-2000, etc.)
    def get_step_number(path):
        try:
            basename = os.path.basename(path)
            step_str = basename.split("-")[1]
            return int(step_str)
        except (IndexError, ValueError):
            return 0
    
    checkpoint_dirs.sort(key=get_step_number)
    
    print(f"Found {len(checkpoint_dirs)} checkpoint directories:")
    for cp_dir in checkpoint_dirs:
        basename = os.path.basename(cp_dir)
        step = get_step_number(cp_dir)
        print(f"{basename} (step {step})")
        
    if len(checkpoint_dirs) <= 1:
        print("Only one checkpoint found, keeping it")
        return {"status": "single_checkpoint", "removed": 0, "space_saved_gb": 0}
    
    latest_checkpoint = checkpoint_dirs[-1]  # Last one after sorting
    to_remove = checkpoint_dirs[:-1]  # All except the last one
    
    print(f"\nKeeping latest: {os.path.basename(latest_checkpoint)}")
    print(f"Removing {len(to_remove)} old checkpoints:")
    total_space_saved = 0
    removed_count = 0
    for cp_dir in to_remove:
        basename = os.path.basename(cp_dir)
        
        try:
            # Quick size estimation (count files)
            file_count = sum(len(files) for _, _, files in os.walk(cp_dir))
            estimated_size_gb = file_count * 0.1  # Rough estimate: 100MB per file average

            print(f"Removing {basename} (~{estimated_size_gb:.1f} GB estimated)")

            # Remove the directory
            shutil.rmtree(cp_dir)
            
            total_space_saved += estimated_size_gb
            removed_count += 1
            
            print(f"Successfully removed")
            
        except Exception as e:
            print(f"Failed to remove {basename}: {e}")
        print(f"\nCleanup Summary:")
    print(f"Checkpoints removed: {removed_count}")
    print(f"Estimated space saved: ~{total_space_saved:.1f} GB")
    print(f"Remaining checkpoint: {os.path.basename(latest_checkpoint)}")
    volume.commit()
    return {
        "status": "cleanup_complete",
        "removed": removed_count,
        "space_saved_gb": total_space_saved,
        "kept_checkpoint": os.path.basename(latest_checkpoint)
    }
    

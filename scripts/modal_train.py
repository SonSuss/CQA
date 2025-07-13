import modal


app = modal.App("TrainChartQA")

# Create or attach a persistent volume
volume = modal.Volume.from_name("chartqa", create_if_missing=True)

# Training image with GPU dependencies - using valid CUDA base image
training_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
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
            "psutil",  # For system monitoring
        ],
        extra_index_url="https://download.pytorch.org/whl/cu121"  # Match CUDA 12.1
    )
    .pip_install(
        ["flash-attn"],
        extra_options="--no-build-isolation"
    )
    .run_commands([
        "apt-get update && apt-get install -y git build-essential",
        "git clone https://github.com/SonSuss/CQA.git /root/CQA",
        "cd /root/CQA && git config --global --add safe.directory /root/CQA",
        "echo 'export PYTHONPATH=/root/CQA:$PYTHONPATH' >> /root/.bashrc",
    ])
    .workdir("/root/CQA")
    .env({"PYTHONPATH": "/root/CQA"})
)

# Training configuration constants
MINUTES = 60
TRAIN_GPU_COUNT = 1
TRAIN_GPU = f"L40S:{TRAIN_GPU_COUNT}"
TRAIN_CPU_COUNT = 8
TRAIN_MEMORY_GB = 32
TRAIN_TIME = 2 # hours


@app.function(
    image=training_image,
    gpu=TRAIN_GPU,
    timeout=5 * MINUTES,
)
def check_gpu_info():
    """Check GPU, system, and library information for training environment"""
    import torch
    import sys
    import platform
    import subprocess
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
    memory=TRAIN_MEMORY_GB * 1024,
    timeout= TRAIN_TIME * 60 * MINUTES,
)
def train_chartqa_accelerate(
    num_epochs: int = 2,
    batch_size: int = 4,
    quick_check: bool = False,
):
    
    import os
    import time
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    
    print("🚀 Starting ChartQA training with Accelerate optimization...")
    print(f"📊 Training config: epochs={num_epochs}, batch_size={batch_size}")
    
    # Check what models are available in cache
    print("\n🔍 Checking for cached models...")
    cache_check = check_cache_and_models.remote()
    print(f"Found {len(cache_check.get('found_models', {}))} cached models")
    
    # Check if preprocessed data exists
    data_path = "/root/data/Chart_QA/processed_data"
    if not os.path.exists(data_path):
        print("❌ Preprocessed data not found. Run preprocess_only() first.")
        result = {"status": "failed", "error": "No preprocessed data found"}
        print("🎯 Early Exit Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        return result
    
    # Initialize accelerator with optimizations
    accelerator = Accelerator(
        gradient_accumulation_steps=2,
        mixed_precision="bf16",  # Use bfloat16 for better performance
        log_with="tensorboard",
        project_dir="/root/data/checkpoints",
        cpu=False,  # Force GPU usage
    )
    
    print(f"🔧 Accelerator: device={accelerator.device}, processes={accelerator.num_processes}")
    
    # Check VRAM
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / 1024**3
        print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Total VRAM: {total_vram:.2f} GB")
        
        # Memory optimization based on VRAM
        if total_vram >= 40:  # L40S has 48GB
            effective_batch_size = batch_size * 2  # Can handle larger batches
            gradient_accumulation = 1
            print("🚀 High VRAM detected - using larger batch size")
        else:
            effective_batch_size = batch_size
            gradient_accumulation = 2
            print("⚡ Standard VRAM - using gradient accumulation")
    
    set_seed(42)  # For reproducibility
    start_time = time.time()
    
    # Import training components
    from models.chart_qa_model.train.train import train
    from models.components.config import ModelArguments, DataArguments, TrainingArguments
    
    # Model configuration
    model_args = ModelArguments(
        model_name_or_path="microsoft/Phi-4-mini-instruct",
        version="phi_instruct",
        freeze_backbone=True,
        tune_mm_mlp_adapter=False,
        vision_tower="mPLUG/TinyChart-3B-768-siglip",
        mm_vision_select_layer=-1,
        mm_projector_type="linear",
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        tune_vision_tower=True,
        tune_entire_model=False,
    )

    data_args = DataArguments(
        data_path=f"{data_path}/train.json",
        eval_data_path=f"{data_path}/val.json",
        lazy_preprocess=True,
        is_multimodal=True,
        image_folder="",
        image_aspect_ratio="square",
    )


    # Accelerate-optimized training arguments
    training_args = TrainingArguments(
        output_dir="/root/data/checkpoints",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=100 if quick_check else 500,
        save_total_limit=3,
        mm_projector_lr=1e-4,
        vision_tower_lr=5e-5,
        weight_decay=0.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5 if quick_check else 10,
        fp16=False,
        bf16=True,  # Better than fp16 for training stability
        model_max_length=1024,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        report_to="tensorboard",
        cache_dir="/root/data/cache",
        optim="adamw_torch_fused",  # Faster optimizer
        bits=16,
        group_by_modality_length=True,
        warmup_steps=50 if quick_check else 100,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,  # Accelerate optimization
        dataloader_pin_memory=True,  # Faster data loading
        remove_unused_columns=False,  # Keep for multimodal
        local_rank=-1,
    )
    try:
        # Monitor VRAM before training
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 Initial VRAM usage: {initial_memory:.2f} GB")
        
        # Run training with accelerate
        train(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args
        )
        
        # Final VRAM stats
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            final_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 Peak VRAM usage: {peak_memory:.2f} GB")
            print(f"💾 Final VRAM usage: {final_memory:.2f} GB")
        
        training_time = time.time() - start_time
        print(f"✅ Accelerate training completed in {training_time:.2f} seconds")
        
        volume.commit()
        
        result = {
            "status": "success",
            "training_time": training_time,
            "epochs": num_epochs,
            "batch_size": effective_batch_size,
            "peak_vram_gb": peak_memory if torch.cuda.is_available() else 0,
            "accelerator_device": str(accelerator.device),
            "output_dir": "/root/data/checkpoints"
        }
        
        print("🎯 Training Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        return result
        
    except Exception as e:
        print(f"❌ Accelerate training failed: {e}")
        import traceback
        traceback.print_exc()
        
        error_result = {
            "status": "failed", 
            "error": str(e),
            "training_time": time.time() - start_time
        }
        
        print("🎯 Training Error Result:")
        for key, value in error_result.items():
            print(f"  {key}: {value}")
        
        return error_result

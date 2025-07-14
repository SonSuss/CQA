import modal
import torch


app = modal.App("TrainChartQA")

# Create or attach a persistent volume
volume = modal.Volume.from_name("chartqa", create_if_missing=True)


cuda_version = "12.6.0"
flavor = "devel" 
operating_sys = "ubuntu22.04"
tag = f"nvidia/cuda:{cuda_version}-{flavor}-{operating_sys}"
gpu = "L40S"

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
TRAIN_GPU = f"L40S"
TRAIN_CPU_COUNT = (1.0,8.0)
TRAIN_MEMORY_GB = (8 * 1024,32 * 1024)  # 8GB to 32GB
TRAIN_TIME = 2 # hours

@app.function(
    image=training_image,
    gpu=TRAIN_GPU,
    timeout= 30 * MINUTES,
    cpu=1
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
    timeout= TRAIN_TIME * 60 * MINUTES,
)
def test_phi_llava_model():
    pull_latest_code()
    import torch
    # from models.chart_qa_model.model.phi_4_llava import PhiLlava_config, PhiLlavaForCausalLM
    from models.chart_qa_model.model.modeling_phi3 import Phi3ForCausalLM
    from models.chart_qa_model.model.configuration_phi3 import Phi3Config
    from transformers import AutoTokenizer
    
    print("🧪 TESTING PHI-LLAVA MODEL WITH FIXED CONFIG")
    print("=" * 60)
    
    model_path = "microsoft/Phi-4-mini-instruct"
    cache_dir = "/root/data/cache"
    
    try:
        # Import necessary libraries
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

        # Set a seed for reproducibility
        set_seed(2024)

        # Define the prompt for the model. In this case, the prompt is a request for C# code.
        prompt = "Write a C# code that reads the content of multiple text files and save the result as CSV"

        # Define the model checkpoint and Phi-3 Model Required
        model_checkpoint = "microsoft/Phi-3-mini-4k-instruct"

        # Load the tokenizer from the model checkpoint
        # trust_remote_code=True allows the execution of code from the model files
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,trust_remote_code=True)

        # Load the model from the model checkpoint
        # trust_remote_code=True allows the execution of code from the model files
        # torch_dtype="auto" automatically determines the appropriate torch.dtype
        # device_map="cuda" specifies that the model should be loaded to the GPU
        # model = Phi3ForCausalLM.from_pretrained(model_checkpoint,
        #                                         trust_remote_code=True,
        #                                         torch_dtype="auto",
        #                                         device_map="cuda")
        
        # Test 1: Official model
        model_official = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cuda"
        )

        # Test 2: Your custom model  
        model_custom = Phi3ForCausalLM.from_pretrained(
            model_checkpoint,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cuda"
        )
        inputs = tokenizer(prompt,
                return_tensors="pt").to("cuda")

        # Test generation with both models
        print("\n🔍 DETAILED GENERATION COMPARISON:")

        try:
            # Test official model generation
            print("Testing official model generation...")
            with torch.no_grad():
                outputs_official = model_official.generate(**inputs, 
                                                        do_sample=True, 
                                                        max_new_tokens=5)  # Small number for testing
                print("✅ Official model generation successful!")
                response = tokenizer.decode(outputs_official[0], skip_special_tokens=True)
                print(f"Response from official model: {response}")

        except Exception as e:
            print(f"❌ Official model generation failed: {e}")

        try:
            # Test custom model generation
            print("Testing custom model generation...")
            with torch.no_grad():
                outputs_custom = model_custom.generate(**inputs, 
                                                    do_sample=True, 
                                                    max_new_tokens=5)  # Small number for testing
                print("✅ Custom model generation successful!")
                response = tokenizer.decode(outputs_custom[0], skip_special_tokens=True)
                print(f"Response from custom model: {response}")
                
        except Exception as e:
            print(f"❌ Custom model generation failed: {e}")
            print("This confirms the issue is in your custom model's generation logic")
            import traceback
            traceback.print_exc()

        # Tokenize the prompt and move the tensors to the GPU


        # Generate a response from the model
        # do_sample=True means the model will generate text by sampling from the distribution of possible outputs
        # max_new_tokens=200 limits the length of the generated text to 200 tokens

        # Decode the generated tokens and remove any special tokens
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(response)
        # # Test our custom config
        # print("\n🔧 LOADING WITH CUSTOM PHI-LLAVA CONFIG:")
        # config = Phi3Config.from_pretrained(model_path, cache_dir=cache_dir)
        # print("✅ Custom config loaded successfully!")
        
        # print(f"Model type: {config.model_type}")
        # print(f"Hidden size: {config.hidden_size}")
        # print(f"Num attention heads: {config.num_attention_heads}")
        # print(f"Expected rope_scaling length: {config.hidden_size // config.num_attention_heads // 2}")
        
        # if hasattr(config, 'rope_scaling') and config.rope_scaling:
        #     print(f"Short factor length: {len(config.rope_scaling['short_factor'])}")
        #     print(f"Long factor length: {len(config.rope_scaling['long_factor'])}")
        
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"Using device: {device}")
        
        # # Test model loading - Microsoft's approach from tuning script
        # model_kwargs = dict(
        #     use_cache=False,
        #     trust_remote_code=True,
        #     attn_implementation="flash_attention_2",
        #     torch_dtype=torch.bfloat16,
        #     device_map=device 
        # )
        # print(f"\n🚀 LOADING PHI-LLAVA MODEL:")
        # model = Phi3ForCausalLM.from_pretrained(
        #     model_path,
        #     **model_kwargs
        # )
        # # model.eval()
        # # model = torch.compile(model)
        # print("✅ Phi-3 model loaded successfully!")


        # # Test tokenizer
        # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
        # # tokenizer.model_max_length = 2048
        # # tokenizer.pad_token = tokenizer.unk_token
        # # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        # # tokenizer.padding_side = 'right'
        # print("✅ Tokenizer loaded successfully!")
        
        # print("tokenizer :",tokenizer)
        # volume.commit()
        
        # # Quick test
        # print(f"\n🧠 QUICK INFERENCE TEST:")
        # # Check if tokenizer has a chat template
        # print(f"Tokenizer chat template: {getattr(tokenizer, 'chat_template', 'None')}")
        # print(f"Special tokens: {tokenizer.special_tokens_map}")
        
        # # Use proper instruction format for Phi-4-mini-instruct
        # if hasattr(tokenizer, 'apply_chat_template'):
        #     # Use the tokenizer's chat template if available
        #     messages = [
        #         {"role": "user", "content": "Hello, who are you?"}
        #     ]
        #     try:
        #         test_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        #         print("Using chat template format:")
        #     except:
        #         test_prompt = "<|user|>\nHello, how are you?<|end|>\n<|assistant|>\n"
        #         print("Using manual format (chat template failed):")
        # else:
        #     test_prompt = "<|user|>\nHello, how are you?<|end|>\n<|assistant|>\n"
        #     print("Using manual format (no chat template):")
        
        # inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
            
        # with torch.no_grad():
        #     outputs = model.generate(**inputs, do_sample=True, max_new_tokens=120)
        #     print(outputs)
        #     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #     print("Response: ", response)

        # Memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"\n💾 GPU MEMORY USAGE:")
            print(f"Allocated: {allocated:.2f} GB")
            print(f"Cached: {cached:.2f} GB")
        
        print("SUCCESS")
        return "Success"
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return "Failed"
    
    

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
    image_path = "/root/data/Chart_QA/processed_data/images"
    output_dir = "/root/data/checkpoints"
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
        version="phi_instruct",
        freeze_backbone=True,
        tune_mm_mlp_adapter=False,
        vision_tower="mPLUG/TinyChart-3B-768-siglip",
        mm_vision_select_layer=-1,
        pretrain_mm_mlp_adapter=None,
        mm_projector_type="resampler",
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
        num_train_epochs=1,  # Reduced for testing
        per_device_train_batch_size=2,  # Reduced for testing
        per_device_eval_batch_size=2,  # Reduced for testing
        gradient_accumulation_steps=2,  # Reduced for testing
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=50,  # More frequent saves for testing
        save_total_limit=3,  # Reduced for testing
        mm_projector_lr=1e-4,
        vision_tower_lr=5e-5,
        weight_decay=0.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        fp16=False,
        bf16=True,
        model_max_length=1024,
        gradient_checkpointing=True,
        dataloader_num_workers=4,  # Reduced for testing
        dataloader_persistent_workers=False,  # Disabled for testing
        report_to="tensorboard",
        cache_dir=cache_dir,
        optim="adamw_torch",
        bits=16,
        group_by_modality_length=True,
        warmup_steps=10,  # Reduced for testing
        max_grad_norm=1.0,
        local_rank=-1,  # For single GPU
    )
    train(model_args, data_args, training_args, log_rewrite=True)
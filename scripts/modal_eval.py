import modal
import torch


app = modal.App("TrainChartQA")

# Create or attach a persistent volume
volume = modal.Volume.from_name("chartqa-A100-llava-siglip-phi4", create_if_missing=True)


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
VAL_GPU = gpu
VAL_CPU_COUNT = (1.0,8.0)
VAL_MEMORY_GB = (8 * 1024,32 * 1024)  # 8GB to 32GB
VAL_TIME = 10 # hours
MODEL_PATH = "/root/data/checkpoints-siglip-mlp2x_gelu-phi4"

@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    gpu=VAL_GPU,
    timeout= 5 * MINUTES,
    cpu=VAL_CPU_COUNT,
    memory=VAL_MEMORY_GB,
)
def model_inference():
    pull_latest_code()
    from eval.inference_model import inference_model
    from models.chart_qa_model.builder import load_pretrained_llava_model
    
    tokenizer, model, image_processor, context_len = load_pretrained_llava_model(MODEL_PATH,
                                                                            device="cuda")
    
    image_path = "/root/data/Chart_QA/ChartQA Dataset/val/png/289.png"
    text = "What's the leftmost value of bar in \"All adults\" category?" #48
    response = inference_model(
            image_path,  # Remove the list brackets - function expects single path
            text, 
            model, 
            tokenizer, 
            image_processor, 
            conv_mode="phi4_instruct",  # This goes in the conv_mode position
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=100,
    )
    print(f"Response: {response}")
    
@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    gpu=VAL_GPU,
    timeout=15 * MINUTES,
    cpu=VAL_CPU_COUNT,
    memory=VAL_MEMORY_GB,
)
def simple_text_test():
    """Test model with text-only input first"""
    pull_latest_code()
    from models.chart_qa_model.builder import load_pretrained_llava_model
    
    tokenizer, model, image_processor, context_len = load_pretrained_llava_model(
        MODEL_PATH, 
        device="cuda"
    )
    
    # Simple text-only test
    text = "What is 2+2?"
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Text-only response: '{response}'")
    return {"response": response}

@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    gpu=VAL_GPU,
    timeout=15 * MINUTES,  # Increase timeout
    cpu=VAL_CPU_COUNT,
    memory=VAL_MEMORY_GB,
)
def model_inference():
    pull_latest_code()
    from eval.inference_model import inference_model
    from models.chart_qa_model.builder import load_pretrained_llava_model
    
    print("🔄 Loading model...")
    tokenizer, model, image_processor, context_len = load_pretrained_llava_model(
        MODEL_PATH, 
        device="cuda"
    )
    
    # 🛠️ FIX: Set pad token properly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("✅ Model loaded successfully")
    
    image_path = "/root/data/Chart_QA/ChartQA Dataset/val/png/289.png"
    text = "What's the leftmost value of bar in \"All adults\" category?"
    
    print("🔄 Running vision inference...")
    response = inference_model(
        image_path,
        text, 
        model, 
        tokenizer, 
        image_processor, 
        conv_mode="phi4_instruct", 
        temperature=0.0,   
        top_p=1.0,                 
        max_new_tokens=50,        
    )
        
    print(f"✅ Vision response: '{response}'")
    return {"response": response, "text_test_works": True}
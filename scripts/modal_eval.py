import modal

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
    .run_commands([
        "apt-get update && apt-get install -y git build-essential clang",
    ])
    .pip_install("bitsandbytes", gpu=gpu)
    .pip_install(
        [
            "torch", "torchvision", "torchaudio",
            "transformers==4.49.0",
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
        # "apt-get update && apt-get install -y git build-essential",
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

MINUTES = 60
VAL_GPU = gpu
VAL_CPU_COUNT = (1.0,8.0)
VAL_MEMORY_GB = (8 * 1024,32 * 1024)  # 8GB to 32GB
VAL_TIME = 10 # hours

    
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
    import torch
    from models.chart_qa_model.builder import load_pretrained_llava_model
    
    tokenizer, model, image_processor, context_len = load_pretrained_llava_model(
        MODEL_PATH, 
        device="cuda"
    )
    
    # Simple text-only test
    text = "What is 2+2?, answer with python code that prints only the result."
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=1000,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Text-only response: '{response}'")
    return {"response": response}

@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    gpu=VAL_GPU,
    timeout=15 * MINUTES, 
    cpu=VAL_CPU_COUNT,
    memory=VAL_MEMORY_GB,
)
def model_inference():
    pull_latest_code()
    import torch
    from eval.inference_model import inference_model
    from models.chart_qa_model.builder import load_pretrained_llava_model
    import time
    print("Loading model...")
    tokenizer, model, image_processor, context_len = load_pretrained_llava_model(
        MODEL_PATH, 
        device="cuda"
    )
    
    # image_path = "/root/data/Chart_QA/ChartQA Dataset/val/png/289.png"
    # text = "What's the leftmost value of bar in \"All adults\" category?"
    # "id": "multi_col_1238", "question": "<|image|>\nQuestion:\nWhat's the highest Distribution of employment by economic sector in 2010",
    # "gt_answer": "Services", "final_model_answer": "24.5 <|end|>" '23.66 <|end|>''23.66 <|end|>'
    image_path = "/root/data/images/test3.png"
    texts= [
        "Question:\nWhat is the red line variable?",
        "Question:\nWhat is the red line variable?",
        "Question:\nWhat is the red line variable?"
        ]

    s_time = time.time()
    print("Running vision inference:")
    for text in texts:
        response = inference_model(
            image_path,
            text, 
            model, 
            tokenizer, 
            image_processor, 
            conv_mode="phi4_instruct", 
            temperature=0.8,
            top_p=0.8,
            max_new_tokens=1024
        )
        print(f"Model response: '{response}'")
    e_time = time.time()
    print(f"Inference time: {e_time - s_time:.2f} seconds")
    
@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    gpu=VAL_GPU,
    timeout=15 * MINUTES, 
    cpu=VAL_CPU_COUNT,
    memory=VAL_MEMORY_GB,
)
def model_inference_system():
    pull_latest_code()
    import torch
    from eval.inference_model import inference_model_system
    from models.chart_qa_model.builder import load_pretrained_llava_model
    import time
    print("Loading model...")
    tokenizer, model, image_processor, context_len = load_pretrained_llava_model(
        MODEL_PATH, 
        device="cuda"
    )
    
    # image_path = "/root/data/Chart_QA/ChartQA Dataset/val/png/289.png"
    # text = "What's the leftmost value of bar in \"All adults\" category?"
    # "id": "multi_col_1238", "question": "<|image|>\nQuestion:\nWhat's the highest Distribution of employment by economic sector in 2010",
    # "gt_answer": "Services", "final_model_answer": "24.5 <|end|>" '23.66 <|end|>''23.66 <|end|>'
    image_path = "/root/data/images/test1.png"
    system_prompt = """You are a helpful assistant for solving chart-based questions using Python.  
Think step by step to understand the chart and the question.  
Then generate Python code that computes the correct answer.  
The code must end with printing only the final result (no explanations, no extra text)."""
    texts= [
        """
Question:
Which year has the lowest GDP in the chart?<|end|>
<|assistant|>
gdp_values = {"2000": 1.2, "2005": 1.0, "2010": 2.3}
lowest_year = min(gdp_values, key=gdp_values.get)
print(lowest_year)<|end|>
<|user|>
Image
<|image|>
Question:
What is the total number of events that occur when "school start" and "after school" """,
"What is the total number of events that occur when \"school start\" and \"after school\""
        ]

    s_time = time.time()
    print("Running vision inference:")
    for text in texts:
        response = inference_model_system(
            image_path,
            system_prompt,
            text, 
            model, 
            tokenizer, 
            image_processor, 
            conv_mode="phi4_instruct_system", 
            temperature=0.8,
            top_p=0.8,
            max_new_tokens=1024
        )
        print(f"model response: '{response}'")
    e_time = time.time()
    print(f"Inference time: {e_time - s_time:.2f} seconds")
        
MODEL_PATH = "/root/data/checkpoint-siglip_-1-resampler2_768_128_3-phi4"
@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    gpu=VAL_GPU,
    timeout=60 * MINUTES, 
    cpu=VAL_CPU_COUNT,
    memory=VAL_MEMORY_GB,
)
def eval_model_chart_qa():
    pull_latest_code()
    import os
    from eval.chart_qa.eval import get_eval
    
    suffix = MODEL_PATH.split("/root/data/checkpoint-")[-1]
    output_path = os.path.join("/root/data/eval_results_chartQAPro/", suffix)
    
    get_eval(model_path=MODEL_PATH,
             valset_path="/root/data/ChartQAPro/testset.json",
             output_path=output_path,
             image_folder="",
            #  temperature=0.0,
            #  top_p=1.0,
            temperature=0.3,
            top_p=0.5,
            max_new_tokens=32,
            min_new_tokens=1,
            num_beams=1)
    
    volume.commit()
    

@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    timeout=15 * MINUTES, 
    cpu=VAL_CPU_COUNT,
    memory=VAL_MEMORY_GB,
)
def run_eval_model_chart_qa():
    pull_latest_code()
    import os
    from eval.chart_qa.eval import eval_model
    eval_path = "/root/data/eval_results_chartQAPro/"
    model_file_list = [
        name for name in os.listdir(eval_path)
        if os.path.isdir(os.path.join(eval_path, name))
    ]
    for answers_path in model_file_list:
        full_answers_path = os.path.join(eval_path, answers_path)
        eval_model(answers_path=full_answers_path, output_path=full_answers_path)
        
    volume.commit()

@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    timeout=15 * MINUTES, 
    cpu=VAL_CPU_COUNT,
    memory=VAL_MEMORY_GB,
) 
def collect_correct_answers():
    import os, json, re
    from difflib import SequenceMatcher
    eval_path = "/root/data/eval_results_chartQAPro/siglip_-1-resampler2_768_128_3-phi4/answers_0.3_0.5.json"
    out_path = "/root/data/eval_results_chartQAPro/siglip_-1-resampler2_768_128_3-phi4/correct_relaxed_0.3_0.5.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    def norm(x):
        if x is None:
            return ""
        return x.replace("<|end|>", "").strip().lower()
    
    def clean_end(s):
        if not isinstance(s, str):
            return ""
        return s.replace("<|end|>", "").strip()
    
    def norm_text(s):
        s = clean_end(s).lower()
        s = re.sub(r'[%,$]', '', s)  # drop symbols but keep digits
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    num_pat = re.compile(r'-?\d+(?:\.\d+)?')

    def extract_numbers(s):
        if not isinstance(s, str):
            return []
        return [float(x) for x in num_pat.findall(s)]

    def numbers_match(gt_nums, pred_nums):
        if not gt_nums or not pred_nums:
            return False
        # Single number case
        if len(gt_nums) == 1 and len(pred_nums) == 1:
            g, p = gt_nums[0], pred_nums[0]
            denom = max(abs(g), abs(p), 1e-8)
            rel_err = abs(g - p) / denom
            return rel_err <= 0.05  # 5%
        # Range (take first two)
        if len(gt_nums) >= 2 and len(pred_nums) >= 2:
            g1, g2 = gt_nums[0], gt_nums[1]
            p1, p2 = pred_nums[0], pred_nums[1]
            def close(a,b):
                return abs(a-b) / max(abs(a), abs(b), 1e-8) <= 0.05
            return close(g1,p1) and close(g2,p2)
        return False

    def strings_match(gt, pred):
        gt_n = norm_text(gt)
        pred_n = norm_text(pred)
        if not gt_n or not pred_n:
            return False
        # Exact after normalization
        if gt_n == pred_n:
            return True
        # Handle year shorthand like 2037-38 vs 2037-2038
        if re.match(r'^\d{4}-\d{2}$', gt_n):
            # expand gt: 2037-38 -> 2037-2038
            base = gt_n[:4]
            tail = gt_n[-2:]
            expanded = f"{base}-{base[:2]}{tail}"
            if expanded == pred_n:
                return True
        if re.match(r'^\d{4}-\d{2}$', pred_n):
            base = pred_n[:4]
            tail = pred_n[-2:]
            expanded = f"{base}-{base[:2]}{tail}"
            if expanded == gt_n:
                return True
        # Fuzzy similarity
        sim = SequenceMatcher(None, gt_n, pred_n).ratio()
        return sim >= 0.90

    correct = []
    for item in data:
        gt_raw = item.get("gt_answer")
        pred_raw = item.get("final_model_answer")
        gt_clean = clean_end(gt_raw)
        pred_clean = clean_end(pred_raw)

        gt_nums = extract_numbers(gt_clean)
        pred_nums = extract_numbers(pred_clean)

        ok = False
        if gt_nums and pred_nums:
            ok = numbers_match(gt_nums, pred_nums)
        if not ok:
            ok = strings_match(gt_clean, pred_clean)

        if ok:
            correct.append({
                "id": item.get("id"),
                "question": item.get("question"),
                "gt_answer": gt_raw,
                "final_model_answer": pred_raw
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(correct, f, ensure_ascii=False, indent=2)

    print(len(correct), "correct answers saved to", out_path)
    volume.commit()
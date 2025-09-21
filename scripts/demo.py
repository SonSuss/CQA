import modal
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

app = modal.App("Demo_TrainChartQA")

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
    .pip_install("fastapi[standard]")
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
        
MODEL_PATH = "/root/data/checkpoint-siglip_-1-resampler2_768_128_3-phi4"
MINUTES = 60
VAL_GPU = gpu
VAL_CPU_COUNT = (1.0,8.0)
VAL_MEMORY_GB = (8 * 1024,32 * 1024)  # 8GB to 32GB
VAL_TIME = 10 # hours

MODEL_PATH_1 = "/root/data/checkpoint-siglip_-1-resampler2_768_128_3-phi4"
MODEL_PATH_2 = "/root/data/checkpoint-siglip_-1-resampler2_768_96_3-phi4_plus"
MODEL_PATH_3 = "model_path_3"
MODEL_PATH_4 = "model_path_4"
MODEL_PATH_5 = "model_path_5"

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
    image_path = "/root/data/Chart_QA/ChartQA Dataset/val/png/multi_col_1238.png"
    texts= [
        "Question:\nWhat's the highest Distribution of employment by economic sector in 2010",
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
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=32
        )
        print(f"Vision response: '{response}'")
    e_time = time.time()
    print(f"Inference time: {e_time - s_time:.2f} seconds")

def model_inference_with_image(tokenizer, model, image_processor, image, texts, temperature, top_p, max_new_tokens):
    conv_mode="phi4_instruct"
    responses = []
    
    import time
    import torch
    from eval.inference_model import inference_model
    
    s_time = time.time()
    print("Running vision inference:")
    for text in texts:
        response = inference_model(
            image,
            text, 
            model, 
            tokenizer, 
            image_processor, 
            conv_mode, 
            temperature,
            top_p,
            max_new_tokens
        )
        print(f"Vision response: '{response}'")
        responses.append(response)
    e_time = time.time()
    run_time = e_time - s_time
    return (responses, run_time) # ([],float)

def model_inference_without_image(tokenizer, model, texts, temperature, top_p, max_new_tokens):
    output_ids_s = []
    responses = []
    
    import time
    import torch
    
    s_time = time.time()
    
    # input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        for text in texts:
            input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=1000,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            output_ids_s.append(output_ids)
    for output_ids in output_ids_s:
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        responses.append(response)
        
    e_time = time.time()
    run_time = e_time - s_time
    return (responses, run_time) # ([],float)

web_app = FastAPI()
# Serve HTML directly from Python
@web_app.get("/", response_class=HTMLResponse)
async def get_form():
    return HTMLResponse(content=HTML_1)

# @web_app.post("/choose")
# async def submit_model(model_path: str = Form(...)):
#     global MODEL_PATH
#     match model_path:
#         case "model1": MODEL_PATH = MODEL_PATH_1
#         case "model2": MODEL_PATH = MODEL_PATH_2
#         case "model3": MODEL_PATH = MODEL_PATH_3
#         case "model4": MODEL_PATH = MODEL_PATH_4
#         case "model5": MODEL_PATH = MODEL_PATH_5 
    
#     # Import Lib
#     from models.chart_qa_model.builder import load_pretrained_llava_model
#     import time
      
#     # Preload Model
#     tokenizer, model, image_processor, context_len = load_pretrained_llava_model(
#         MODEL_PATH, 
#         device="cuda"
#     )
#     return {"model": MODEL_PATH}


@web_app.post("/submit")
async def sumbit_form(
    image: UploadFile = File(None)
    , text: str = Form(...)
    , model_path: str = Form(...)
    , temperature: float = Form(0.0)
    , top_p: float = Form(0.0)
    , max_new_tokens: float = Form(32)
    ):
    pull_latest_code()
    # Import Lib for Image
    from PIL import Image
    import io
    # Import Lib for Preload Model
    from models.chart_qa_model.builder import load_pretrained_llava_model
    
    try:
        match model_path:
            case "model1": model_path = MODEL_PATH_1
            case "model2": model_path = MODEL_PATH_2
            case "model3": model_path = MODEL_PATH_3
            case "model4": model_path = MODEL_PATH_4
            case "model5": model_path = MODEL_PATH_5
            
        # Preload Model
        tokenizer, model, image_processor, context_len = load_pretrained_llava_model(
            model_path, 
            device="cuda"
        )
        
        # Split text by newlines to get multiple questions
        texts = [t.strip() for t in text.split('\n') if t.strip()]
        
        # NO IMAGE CASE
        if not image or image.filename == "":
            responses, runtime = model_inference_without_image(tokenizer, model, texts, temperature, top_p, max_new_tokens)
            # After you get the responses list, add this line:
            responses = [r.replace("<|end|>", "").strip() for r in responses]
            return {
                "response": responses
                , "runtime": runtime,
                "question": texts
                , "model": model_path
                , "temperature": temperature
                , "top_p": top_p
                , "max_new_token": max_new_tokens
                , "filename": "No image uploaded"
            }
            
        # IMAGE CASE
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run ChatBot 
        responses, runtime = model_inference_with_image(tokenizer, model, image_processor, pil_image, texts, temperature, top_p, max_new_tokens)
        # After you get the responses list, add this line:
        responses = [r.replace("<|end|>", "").strip() for r in responses]
        return {
            "response": responses
            , "runtime": runtime,
            "question": texts
            , "model": model_path
            , "temperature": temperature
            , "top_p": top_p
            , "max_new_token": max_new_tokens
            , "filename": image.filename
        }
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.function(
    image=training_image,
    volumes={"/root/data": volume},
    gpu=VAL_GPU,
    timeout=15 * MINUTES, 
    cpu=VAL_CPU_COUNT,
    memory=VAL_MEMORY_GB,
)
@modal.asgi_app()
def fastapi_app():
    return web_app

# AJAX
HTML_1 = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Modern ChartQA Chatbot</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f4f6fb;
            margin: 0;
            padding: 0;
        }
        .split-container {
            display: flex;
            flex-direction: row;
            height: 100vh;
            max-width: 1200px;
            margin: 0 auto;
        }
        .left-panel, .right-panel {
            flex: 1;
            padding: 40px 32px;
            box-sizing: border-box;
            background: #fff;
            min-width: 0;
        }
        .left-panel {
            border-right: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .right-panel {
            background: #f7fafc;
            overflow-y: auto;
        }
        h1 {
            text-align: center;
            color: #2d3748;
            margin-bottom: 24px;
        }
        label {
            font-weight: 500;
            color: #4a5568;
            margin-top: 12px;
            align-self: flex-start;
        }
        input, select, textarea, button {
            width: 100%;
            margin: 8px 0 16px 0;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            font-size: 1rem;
            box-sizing: border-box;
        }
        textarea {
            resize: vertical;
            min-height: 60px;
        }
        button {
            background: #3182ce;
            color: #fff;
            border: none;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.2s;
        }
        button:hover {
            background: #2563eb;
        }
        #preview {
            margin-top: 10px;
            width: 220px;
            height: auto;
            display: none;
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 8px;
            background: #f7fafc;
        }
        .chat-area {
            margin-top: 0;
            background: #f7fafc;
            border-radius: 12px;
            padding: 24px;
            min-height: 120px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }
        .bubble {
            display: flex;
            flex-direction: column;
            margin-bottom: 18px;
        }
        .question {
            align-self: flex-end;
            background: #3182ce;
            color: #fff;
            padding: 12px 18px;
            border-radius: 18px 18px 4px 18px;
            max-width: 80%;
            margin-bottom: 6px;
            font-size: 1rem;
            word-break: break-word;
        }
        .response {
            align-self: flex-start;
            background: #e2e8f0;
            color: #2d3748;
            padding: 12px 18px;
            border-radius: 18px 18px 18px 4px;
            max-width: 80%;
            font-size: 1rem;
            word-break: break-word;
        }
        .meta {
            font-size: 0.95rem;
            color: #718096;
            margin-top: 16px;
        }
        @media (max-width: 900px) {
            .split-container {
                flex-direction: column;
                height: auto;
            }
            .left-panel, .right-panel {
                border: none;
                padding: 24px 12px;
            }
        }
    </style>
</head>
<body>
    <div class="split-container">
        <div class="left-panel">
            <h1>ChartQA Chatbot</h1>
            <form id="question-form" style="width:100%;max-width:400px;">
                <label for="image">Select Image:</label>
                <input type="file" id="image" name="image">
                <img id="preview" src="#" alt="Image Preview">
                
                <label for="text">Enter Questions (one per line):</label>
                <textarea id="text" name="text" rows="5" cols="40" placeholder="Type each question on a new line..."></textarea>
                
                <label for="model_path">Choose Model:</label>
                <select name="model_path" id="model_path">
                    <option value="model1">SigLIP-tome_Resambler128_Phi4</option>
                    <option value="model2">SigLIP-tome_Resambler96_Phi4</option>
                    <option value="model3">SigLIP-tome_Resambler256_Phi4</option>
                    <option value="model4">SigLIP-tome_MLP4xGELU_Phi4</option>
                    <option value="model5">SigLIP-tome_MLP2xGELU_Phi4</option>
                </select>

                <label for="temperature">Temperature (0.00 ~ 1.00):</label>
                <input type="number" id="temperature" name="temperature" min="0.00" max="1.00" step="0.1" placeholder="default: 0.00">

                <label for="top_p">Top_p (0.00 ~ 1.00):</label>
                <input type="number" id="top_p" name="top_p" min="0.00" max="1.00" step="0.1" placeholder="default: 0.00">

                <label for="max_new_tokens">Max new token (higher = longer answer):</label>
                <input type="number" id="max_new_tokens" name="max_new_tokens" min="0" step="1" placeholder="default: 1000">

                <button type="submit" id="submit" name="submit">Ask</button>
            </form>
        </div>
        <div class="right-panel">
            <div class="chat-area" id="result"></div>
            <button type="button" id="clear" name="clear" style="background:#e53e3e;color:#fff;width:120px;margin:24px auto 0 auto;display:block;">Clear</button>
        </div>
    </div>

    <script>
        const imageInput = document.getElementById('image');
        const preview = document.getElementById('preview');

        imageInput.addEventListener('change', function () {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();

                reader.addEventListener('load', function () {
                    preview.setAttribute('src', this.result);
                    preview.style.display = 'block';
                });

                reader.readAsDataURL(file);
            }
        });

        const form = document.getElementById("question-form");
        const resultDisplay = document.getElementById("result");
        
        document.getElementById("submit").addEventListener("click", async function(e) {
            e.preventDefault();

            resultDisplay.innerHTML = '<span class="meta">Thinking...</span>';

            const formData = new FormData(form);
            const response = await fetch("/submit", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            if (response.ok) {
                let html = "";
                if (data.question && data.response) {
                    for (let i = 0; i < data.question.length; i++) {
                        html += `
                            <div class="bubble">
                                <div class="question">Q${i + 1}: ${data.question[i]}</div>
                                <div class="response">A${i + 1}: ${data.response[i]}</div>
                            </div>
                        `;
                    }
                }
                html += `
                    <div class="meta">
                        <strong>Run time:</strong> ${data.runtime} sec<br>
                        <strong>Model:</strong> ${data.model}<br>
                        <strong>Temperature:</strong> ${data.temperature}<br>
                        <strong>Top_p:</strong> ${data.top_p}<br>
                        <strong>Max Token:</strong> ${data.max_new_token}<br>
                        <strong>Image name:</strong> ${data.filename}
                    </div>
                `;
                resultDisplay.innerHTML = html;
            } else {
                resultDisplay.innerHTML = `<span class="meta">Error: ${data.error}</span>`;
            }
        });
        
        document.getElementById("clear").addEventListener("click", function() {
            // Clear all input fields
            document.getElementById("image").value = "";
            document.getElementById("preview").style.display = "none";
            document.getElementById("text").value = "";
            document.getElementById("model_path").selectedIndex = 0;
            document.getElementById("temperature").value = "";
            document.getElementById("top_p").value = "";
            document.getElementById("max_new_tokens").value = "";
            // Clear output
            document.getElementById("result").innerHTML = "";
        });
    </script>
</body>
</html>
"""
# LEGACY FEATURE

# <li><strong>Format:</strong> ${data.format}</li>
# <li><strong>Size:</strong> ${data.size[0]} x ${data.size[1]}</li>
# <li><strong>Mode:</strong> ${data.mode}</li>

# <button type="submit" id="choose" name="choose">Choose Model</button>
#         <p id="model_result"></p>
                            
# const ModelDisplay = document.getElementById("model_result");
    
#         document.getElementById("choose").addEventListener("click", async function(e) {
#             e.preventDefault();

#             const formData = new FormData(form);
            
#             const response = await fetch("/choose", {
#                 method: "POST",
#                 body: formData
#             });
#             const data = await response.json();
#             if (response.ok) {
#                 ModelDisplay.innerHTML  = `
#                     <h3>Info</h3>
#                         <ul>           
#                             <li><strong>Model:</strong> ${data.model}</li>
#                         </ul>
#                 `;
#             } else {
#                 ModelDisplay.textContent = `Error: ${data.error}`;
#             }
#         });

# siglip_-1-resampler2_768_128_3-phi4
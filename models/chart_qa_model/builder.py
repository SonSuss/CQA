import os

from transformers import AutoTokenizer, BitsAndBytesConfig
from models.chart_qa_model.model.phi_4_llava import PhiLlavaForCausalLM, PhiLlava_config
from models.components.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


import torch

def load_pretrained_llava_model(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    elif device == "cpu":
        kwargs['torch_dtype'] = torch.float32
    else:
        kwargs['torch_dtype'] = torch.float16
        
    #load Llava model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right")
    cfg_pretrained = PhiLlava_config.from_pretrained(model_path)
    # model = PhiLlavaForCausalLM.from_pretrained(cfg_pretrained._name_or_path, low_cpu_mem_usage=True, **kwargs)
    model = PhiLlavaForCausalLM.from_pretrained(
            model_path,
            config=cfg_pretrained,
            cache_dir="/root/data/cache",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **kwargs
        )
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
        
    # manually load vision tower state_dict if needed
    if cfg_pretrained.tune_vision_tower:
        print("manually loading vision tower state_dict.")
        vision_path = os.path.join(model_path, "vision_tower", "pytorch_model.bin")
        if os.path.exists(vision_path):
            finetuned_weights = torch.load(vision_path, map_location="cpu")
            _, unexpected_keys = vision_tower.vision_tower.load_state_dict(finetuned_weights, strict=False, assign=True)
            if unexpected_keys:
                print("Unexpected keys:", unexpected_keys)
                print("I AM COOKED!")
                return

    if cfg_pretrained.tune_mm_mlp_adapter:
        print("manually loading projecor state_dict.")
        projector_dir = os.path.join(model_path, "mm_projector")
        if os.path.isdir(projector_dir):
            projector_path = os.path.join(projector_dir, "mm_projector.bin")
            if os.path.exists(projector_path):
                finetuned_weights = torch.load(projector_path, map_location="cpu")
                _, unexpected_keys = model.load_state_dict(finetuned_weights, strict=False, assign=True)
                if unexpected_keys:
                    print("Unexpected keys:", unexpected_keys)
                    print("I AM COOKED!")
                    return
        else:
            print("mm_projector folder not found. Skipping projector weight loading.")
            
    # print("Model architecture:", model)
    # for name, module in model.named_children():
    #     print(f"{name}: {module}")

    # To see all submodules recursively:
    # for name, module in model.named_modules():
    #     print(name, ":", type(module))
        
    tokens_to_add = []
    if mm_use_im_patch_token and DEFAULT_IMAGE_PATCH_TOKEN not in tokenizer.vocab:
        tokens_to_add.append(DEFAULT_IMAGE_PATCH_TOKEN)
    if mm_use_im_start_end:
        if DEFAULT_IM_START_TOKEN not in tokenizer.vocab:
            tokens_to_add.append(DEFAULT_IM_START_TOKEN)
        if DEFAULT_IM_END_TOKEN not in tokenizer.vocab:
            tokens_to_add.append(DEFAULT_IM_END_TOKEN)
    
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        
    target_device = device if device != "auto" else "cuda"
    target_dtype = torch.float32 if target_device == 'cpu' else torch.float16
    print(f"Moving all model components to {target_device} with dtype {target_dtype}")
    model.to(device=target_device, dtype=target_dtype)
    
    dummy_image = torch.zeros(1, 3, 768, 768).to(target_dtype)
    with torch.no_grad():
        vision_out = vision_tower(dummy_image)
        model.get_model().mm_projector.to(target_dtype)
        projected_out = model.get_model().mm_projector(vision_out)
    print("Vision tower output shape: %s", vision_out.shape)
    print("Projected output shape: %s", projected_out.shape)
    
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
        
    model.eval()
    return tokenizer, model, image_processor, context_len
    
    
    



from transformers import AutoTokenizer, BitsAndBytesConfig
from models.data_generator.model.sigllama import SigLlamaForCausalLM, SiglipLlamaConfig
from models.data_generator.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


import torch

def load_pretrained_llava_model(model_path, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
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
    model = SigLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    if device != "auto":
        if device == 'cpu':
            vision_tower.to(device=device, dtype=torch.float32)
        else:
            vision_tower.to(device=device, dtype=torch.float16)
    
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
    
    
    



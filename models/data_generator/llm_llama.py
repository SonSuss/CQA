import torch

from transformers import BitsAndBytesConfig

def load_pretrained_llm_model(model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
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

    print(kwargs)

load_pretrained_llm_model("Aaa")

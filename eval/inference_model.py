from models.components.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from models.components.conversation import conv_templates, SeparatorStyle

from models.components.constants import IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN,DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN,IMAGE_PLACEHOLDER

import torch
import re
from PIL import Image

def inference_model(image_path ,input, model, tokenizer, image_processor, conv_mode, temperature=0,top_p=1.0, max_new_tokens=1024):
    qs = input
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(image_path).convert("RGB")
    image_tensor= image_processor.preprocess(image,return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16 if "cuda" in str(model.device) else torch.float32)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # outputs = tokenizer.batch_decode(
    #     output_ids, skip_special_tokens=True
    # )[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[: -len(stop_str)]
    # outputs = outputs.strip()
    # return outputs
    
    input_token_len = input_ids.shape[1]
    generated_tokens = output_ids[:, input_token_len:]
    
    # Filter out any invalid token IDs
    valid_tokens = []
    for token_id in generated_tokens[0]:
        if token_id is not None and token_id >= 0 and token_id < len(tokenizer):
            valid_tokens.append(token_id.item())
    
    # Decode only valid tokens
    if valid_tokens:
        outputs = tokenizer.decode(valid_tokens, skip_special_tokens=True)
    else:
        outputs = ""
    
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


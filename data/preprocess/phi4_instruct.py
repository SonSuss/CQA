import time
from typing import Dict, Optional, Sequence, List
import copy

import transformers
import tokenizers
import torch

from models.components import conversation as conversation_lib
from models.components.mm_utils import tokenizer_image_token
from data.process import register_preprocess


IGNORE_INDEX = -100

@register_preprocess('phi4_instruct')
def preprocess_phi_4_instruct(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv_template = conversation_lib.conv_phi4_instruct.copy()
    roles = {"human": conv_template.roles[0], "gpt": conv_template.roles[1]}

    conversations = []

    for i, source in enumerate(sources):
        # Skip first if not from human
        if roles[source[0]["from"]] != conv_template.roles[0]:
            source = source[1:]

        conv_template.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv_template.roles[j % 2], f"Role mismatch at index {i}"
            conv_template.append_message(role, sentence["value"])
        conversations.append(conv_template.get_prompt())

    # Tokenize full prompts
    if has_image:
        input_ids = torch.stack([
            tokenizer_image_token(prompt, tokenizer, return_tensors='pt') 
            for prompt in conversations
        ], dim=0)
    else:
        tokenized = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        input_ids = tokenized.input_ids

    targets = input_ids.clone()
    assert conv_template.sep_style == conversation_lib.SeparatorStyle.PHI4

    # For PHI style, we need to mask the user input and system tokens
    for conversation, target in zip(conversations, targets):
        # Split by the separator to get individual rounds
        rounds = conversation.split(conv_template.sep)
        cur_len = 0

        for i, rou in enumerate(rounds):
            if rou.strip() == "":
                break

            # For PHI format, check if this is a user input section
            if rou.startswith(conv_template.roles[0]):  # "<|user|>\n"
                # This is user input - mask it
                if has_image:
                    round_len = len(tokenizer_image_token(rou + conv_template.sep, tokenizer))
                else:
                    round_len = len(tokenizer(rou + conv_template.sep).input_ids)
                
                # Mask the entire user input section
                target[cur_len : cur_len + round_len] = IGNORE_INDEX
                cur_len += round_len
            
            elif rou.startswith(conv_template.roles[1]):  # "<|assistant|>\n"
                # This is assistant response - don't mask the actual response content
                role_part = conv_template.roles[1]  # "<|assistant|>\n"
                
                if has_image:
                    role_len = len(tokenizer_image_token(role_part, tokenizer))
                    round_len = len(tokenizer_image_token(rou + conv_template.sep, tokenizer))
                else:
                    role_len = len(tokenizer(role_part).input_ids)
                    round_len = len(tokenizer(rou + conv_template.sep).input_ids)
                
                # Mask only the role part, not the response content
                target[cur_len : cur_len + role_len] = IGNORE_INDEX
                cur_len += round_len
            
            else:
                # Handle any other content
                if has_image:
                    round_len = len(tokenizer_image_token(rou + conv_template.sep, tokenizer))
                else:
                    round_len = len(tokenizer(rou + conv_template.sep).input_ids)
                cur_len += round_len

        # Mask any remaining tokens beyond the conversation
        target[cur_len:] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets
    )
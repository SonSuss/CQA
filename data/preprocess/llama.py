import time
from typing import Dict, Optional, Sequence, List
import copy

import transformers
import tokenizers
import torch

from models.data_generator import conversation as conversation_lib
from models.data_generator.mm_utils import tokenizer_image_token
from data.process import register_preprocess


IGNORE_INDEX = -100
@register_preprocess('llama_3')
def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv_template = conversation_lib.conv_llama_3.copy()  
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
    assert conv_template.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv_template.sep + conv_template.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        rounds = conversation.split(conv_template.sep2)
        cur_len = 0

        for i, rou in enumerate(rounds):
            if rou.strip() == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Reattach separator

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX


    return dict(
        input_ids=input_ids,
        labels=targets
    )

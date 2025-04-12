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
@register_preprocess('llama_v3')
def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv_template = conversation_lib.default_conversation.copy()
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

    sep = conv_template.sep + conv_template.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        rounds = conversation.split(conv_template.sep2)
        cur_len = 0

        for round_str in rounds:
            if round_str.strip() == "":
                break

            parts = round_str.split(sep)
            if len(parts) != 2:
                break  # unexpected format
            prompt_part = parts[0] + sep

            # Compute lengths
            if has_image:
                round_len = len(tokenizer_image_token(round_str, tokenizer))
                instruction_len = len(tokenizer_image_token(prompt_part, tokenizer))
            else:
                round_len = len(tokenizer(round_str).input_ids)
                instruction_len = len(tokenizer(prompt_part).input_ids)

            # Mask the instruction (user message) part
            target[cur_len:cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        # Mask any leftover padding/extra
        target[cur_len:] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets
    )

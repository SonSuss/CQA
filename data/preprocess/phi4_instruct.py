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
    # sources = list(comversations)
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
    sep = "\n"+conv_template.roles[1]+ "\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv_template.sep)
        rounds = conversation.split(conv_template.sep)
        cur_len = 0
        for i, rou in enumerate(rounds):
            if rou.strip() == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if  has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print("number of rounds: ", len(rounds) - 1)
                print("rounds: ", rounds[:-1])
                print("conversation: ", conversations)
                print(target)
                print(input_ids)
                time.sleep(5)
                target[:] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets
    )
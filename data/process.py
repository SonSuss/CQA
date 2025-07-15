import os
os.environ["RANK"] = "0"
import importlib
from typing import Dict, Optional, Sequence, List

import transformers


from models.components import conversation as conversation_lib
from models.components.config import DataArguments
from models.components.utils import rank0_print
from models.components.constants import DEFAULT_IM_END_TOKEN,DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN

PREPROCESS_REGISTRY = {}


def register_preprocess(name):
    def register_preprocess_cls(cls):
        if name in PREPROCESS_REGISTRY:
            return PREPROCESS_REGISTRY[name]

        PREPROCESS_REGISTRY[name] = cls
        return cls

    return register_preprocess_cls


def import_modules(modules_dir, namespace):
    for file in os.listdir(modules_dir):
        path = os.path.join(modules_dir, file)

        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            module_name = file[: file.find(".py")] if file.endswith(".py") else file
            
            rank0_print(f"Importing module: {namespace + '.' + module_name}")
            importlib.import_module(namespace + "." + module_name)

models_dir = os.path.join(os.path.dirname(__file__), 'preprocess')
import_modules(models_dir, "data.preprocess")


def PreprocessSelect(version, logger=None):
    result = PREPROCESS_REGISTRY.get(version, None)
    if result is None:
        for name in PREPROCESS_REGISTRY.keys():
            if version in name:
                result = PREPROCESS_REGISTRY[name]
                break
    if result is None:
        result = PREPROCESS_REGISTRY['default']
    
    if logger:
        logger.info(f"Selected preprocess class: {result.__name__ if hasattr(result, '__name__') else result}")
    
    rank0_print(f"Selected preprocess class: {result.__name__ if hasattr(result, '__name__') else result}")
    return result



def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    logger=None
) -> Dict:
    preprocess_cls = PreprocessSelect(conversation_lib.default_conversation.version, logger)
    if logger:
        logger.info(f"Using preprocess class: {preprocess_cls.__name__ if hasattr(preprocess_cls, '__name__') else preprocess_cls}")
    else:
        rank0_print(f"Using preprocess class: {preprocess_cls.__name__ if hasattr(preprocess_cls, '__name__') else preprocess_cls}")
    return preprocess_cls(sources, tokenizer, has_image)
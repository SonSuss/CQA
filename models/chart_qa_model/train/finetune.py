import pathlib

from models.chart_qa_model.model.phi_4_llava import PhiLlava_config, PhiLlavaForCausalLM
from models.components.train.llava_trainer import LLaVATrainer
from models.components.config import ModelArguments, DataArguments, TrainingArguments
from models.components.utils import get_bnb_model_args, lora_setting, smart_tokenizer_and_embedding_resize, lora_kbit_setting, get_log_writer
from models.components.train.llava_trainer import LLaVATrainer
from models.components.train.train_utils import *
from models.components import conversation as conversation_lib
from data.dataset import *

import transformers
import torch

from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer

def finetune(model_path: str, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments, log_rewrite: bool):
    global local_rank
    logger = get_log_writer(log_dir=training_args.output_dir,
                            log_name="train.log",
                            level=logging.INFO,
                            rewrite=log_rewrite)
    logger.info("Model arguments: %s", model_args)
    logger.info("Data arguments: %s", data_args)
    logger.info("Training arguments: %s", training_args)
    local_rank = training_args.local_rank
    
    if not pathlib.Path(model_path).exists():
        logger.error("Model path does not exist: %s", model_path)
        return
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = get_bnb_model_args(training_args)
    cfg_pretrained = PhiLlava_config.from_pretrained(model_path)
    
    model = PhiLlavaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=cfg_pretrained,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args,
            attn_implementation="flash_attention_2",
            torch_dtype=compute_dtype,
            trust_remote_code=True
        )

    print(model)

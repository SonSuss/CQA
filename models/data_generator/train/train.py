import pathlib
import os
import copy

from models.data_generator.model.sigllama import SigLlamaForCausalLM, SiglipLlamaConfig
from models.data_generator.train.config import ModelArguments, DataArguments, TrainingArguments
from models.data_generator.train.utils import get_bnb_model_args, lora_setting, smart_tokenizer_and_embedding_resize, unlock_vit, lora_kbit_setting
from models.data_generator.train.llava_trainer import LLaVATrainer
from models.data_generator import conversation as conversation_lib
from models.data_generator.train.train_utils import *
from data.dataset import *

import transformers
import torch

from peft import prepare_model_for_kbit_training
from transformers import LlamaTokenizerFast


#train config

#model config
model_args = ModelArguments(
    model_name_or_path="meta-llama/Llama-3.1-8B",
    version="v1",
    freeze_backbone=True,
    tune_mm_mlp_adapter=False,
    vision_tower="google/siglip-so400m-patch14-384",
    mm_vision_select_layer=-1,
    pretrain_mm_mlp_adapter=None,
    mm_projector_type="linear",
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,
    mm_patch_merge_type="flat",
    mm_vision_select_feature="patch",
    resampler_hidden_size=768,
    num_queries=128,
    num_resampler_layers=3,
    tune_vision_tower=True,
    tune_entire_model=False,
    tune_vit_from_layer=-1,
    tune_embed_tokens=False,
)

data_args = DataArguments(
    data_path="./data/train.json",
    eval_data_path="./data/eval.json",
    lazy_preprocess=True,
    is_multimodal=True,
    image_folder="./data/images",
    image_aspect_ratio="square",
)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=10,
    learning_rate=1e-4,
    weight_decay=0.0,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    fp16=True,
    bf16=False,
    model_max_length=1024,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    report_to="tensorboard",
    cache_dir="./cache",
    optim="adamw_torch",
    bits=16,
    group_by_modality_length=True
)



def train():
    global local_rank
    local_rank = training_args.local_rank

    # prepare model and tokenizer
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = get_bnb_model_args(training_args)
        #load model llm
    if model_args.vision_tower is not None:
        model_name = model_args.model_name_or_path
        cfg_pretrained = SiglipLlamaConfig.from_pretrained(model_name)
        model_class = SigLlamaForCausalLM
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=cfg_pretrained,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args,
            # attn_implementation="flash_attention_2",
            attn_implementation=None,
            torch_dtype=compute_dtype
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
        
    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    # if training_args.gradient_checkpointing:
    #     if hasattr(model, "enable_input_require_grads"):
    #         model.enable_input_require_grads()
    #     else:
    #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if training_args.lora_enable:
        model = lora_setting(model, training_args)
        
    # using llama tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    # Handle Special Tokens
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token = tokenizer.pad_token

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    model.config.tokenizer_padding_side = tokenizer.padding_side
    
    #load vision tower
    if model_args.vision_tower is not None:
        # model.config.tune_embed_tokens = training_args.tune_embed_tokens = model_args.tune_embed_tokens
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        if training_args.gradient_checkpointing:
            vision_tower.vision_tower.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(vision_tower.vision_tower, "enable_input_require_grads"):
                vision_tower.vision_tower.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                vision_tower.vision_tower.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.tune_vision_tower = training_args.tune_vision_tower = model_args.tune_vision_tower
        model.config.tune_entire_model = training_args.tune_entire_model = model_args.tune_entire_model
        if model_args.tune_entire_model:
            lr_of_mlp = training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate
            # Tune the MLP, The LR of MLP is {lr_of_mlp}
            if training_args.lora_enable:
                unlock_vit(training_args, model_args, vision_tower)
            else:
                model.requires_grad_(True)
                unlock_vit(training_args, model_args, vision_tower)

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        
    print("mm_projector: ",model.get_model().mm_projector)
    # load data
    if training_args.bits in [4, 8]:
        lora_kbit_setting(model, training_args)

    data_module = make_supervised_data_module_with_eval(tokenizer=tokenizer,
                                              data_args=data_args)

    print("trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("total parameters: ", sum(p.numel() for p in model.parameters()))

    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    
    if training_args.lora_enable:
        lora_save_model(model, training_args)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
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
    
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if training_args.lora_enable:
        model = lora_setting(model, training_args)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=training_args.cache_dir,
        use_fast=True,
        trust_remote_code=True
        )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = training_args.model_max_length
    
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
    # print("Model architecture:", model)
    # for name, module in model.named_children():
    #     print(f"{name}: {module}")
    # print(cfg_pretrained.tune_mm_mlp_adapter, cfg_pretrained.tune_vision_tower)
    
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    
    if cfg_pretrained.tune_vision_tower:
        vision_path = os.path.join(model_path, "vision_tower", "pytorch_model.bin")
        if os.path.exists(vision_path):
            finetuned_weights = torch.load(vision_path, map_location="cpu")
            vision_tower.vision_tower.load_state_dict(finetuned_weights, strict=False, assign=True)
        else:
            logger.warning("Vision tower weights not found at: %s", vision_path)
            return
        
    if cfg_pretrained.tune_mm_mlp_adapter:
        projector_path = os.path.join(model_path, "mm_projector", "mm_projector.bin")
        if os.path.exists(projector_path):
            finetuned_weights = torch.load(projector_path, map_location="cpu")
            _, unexpected_keys = model.load_state_dict(finetuned_weights, strict=False, assign=True)
            if unexpected_keys:
                print("Unexpected keys:", unexpected_keys)
                print("I AM COOKED!")
                return
        else:
            logger.warning("MM projector weights not found at: %s", projector_path)
            return
    else:
        logger.warning("MM MLP adapter should be train at anystep!")
        return
    
    if model_args.tune_vision_tower:
        unlock_vit(training_args, model_args, vision_tower)
        logger.info(f"Vision tower frozen - trainable params: {sum(p.numel() for p in model.get_vision_tower().parameters() if p.requires_grad)}")
    
    #to make finetuning and loading easier, as long as the vision is trained once, always set to True
    model.config.tune_vision_tower = training_args.tune_vision_tower = model_args.tune_vision_tower or cfg_pretrained.tune_vision_tower
    print("Vision tower tuning:", model.config.tune_vision_tower)
    
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
    #this should always be set to True
    if model_args.tune_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        print("This should not be called")
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
    
    if model_args.tune_entire_model:
        # lr_of_mlp = training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate
        if training_args.lora_enable:
            unlock_vit(training_args, model_args, vision_tower)
        else:
            model.requires_grad_(True)
            unlock_vit(training_args, model_args, vision_tower)
    model.config.tune_entire_model = training_args.tune_entire_model = model_args.tune_entire_model
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.config.tune_embed_tokens = training_args.tune_embed_tokens = model_args.tune_embed_tokens
    
    if training_args.bits in [4, 8]:
        lora_kbit_setting(model, training_args)
    
    data_module = make_supervised_data_module_with_eval(tokenizer=tokenizer, data_args=data_args, logger=logger)
    
    vision_tower_params = sum(p.numel() for p in model.get_model().vision_tower.parameters())
    vision_tower_trainable = sum(p.numel() for p in model.get_model().vision_tower.parameters() if p.requires_grad)
    mm_projector_params = sum(p.numel() for p in model.get_model().mm_projector.parameters() if p.requires_grad)
    logger.info(f"Vision Tower total params: {vision_tower_params:,}")
    logger.info(f"Vision Tower trainable params: {vision_tower_trainable:,}")
    logger.info(f"MM Projector trainable params: {mm_projector_params:,}")
    logger.info(f"mm_projector: {model.get_model().mm_projector}")

    # Check if there are embedding parameters being trained
    embed_trainable = sum(p.numel() for p in model.get_input_embeddings().parameters() if p.requires_grad)
    output_embed_trainable = sum(p.numel() for p in model.get_output_embeddings().parameters() if p.requires_grad)
    logger.info(f"Input embeddings trainable: {embed_trainable:,}")
    logger.info(f"Output embeddings trainable: {output_embed_trainable:,}")
    
    logger.info("trainable parameters: %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info("total parameters: %s", sum(p.numel() for p in model.parameters()))
    
    logger.info("Model device: %s", training_args.device)
        
    model.to(device=training_args.device)
    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           custom_logger=logger,
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
        
    tokenizer.save_pretrained(training_args.output_dir)
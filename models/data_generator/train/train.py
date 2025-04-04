from CQA.models.data_generator.model.sigllama import SigLlamaForCausalLM
from CQA.models.data_generator.train.config import ModelArguments, DataArguments, TrainingArguments

import transformers
import torch



#train config

#model config
model_args = ModelArguments(
    model_name_or_path="meta-llama/Llama-3.1-8B",
    version="linear_projector",
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
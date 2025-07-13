import torch
import torch.nn as nn

from typing import Optional, List, Tuple, Union

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Phi3Config, Phi3ForCausalLM, Phi3Model

from models.components.llava import LlavaMetaForCausalLM, LlavaMetaModel


class PhiLlava_config(Phi3Config):
    model_type = "Phi_llava"
    
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration using the correct formula with partial_rotary_factor.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 3:
            raise ValueError(
                "`rope_scaling` must be a dictionary with three fields, `type`, `short_factor` and `long_factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_short_factor = self.rope_scaling.get("short_factor", None)
        rope_scaling_long_factor = self.rope_scaling.get("long_factor", None)
        
        if rope_scaling_type is None or rope_scaling_type not in ["longrope"]:
            raise ValueError(f"`rope_scaling`'s type field must be one of ['longrope'], got {rope_scaling_type}")
        
        if not (
            isinstance(rope_scaling_short_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_short_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s short_factor field must be a list of numbers, got {rope_scaling_short_factor}"
            )
        
        if not (
            isinstance(rope_scaling_long_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_long_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s long_factor field must be a list of numbers, got {rope_scaling_long_factor}"
            )
        
        # Use the correct formula with partial_rotary_factor
        rotary_ndims = int(self.hidden_size // self.num_attention_heads * self.partial_rotary_factor)
        expected_length = rotary_ndims // 2
        
        print(f"🔧 RoPE Scaling Validation:")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num attention heads: {self.num_attention_heads}")
        print(f"  Partial rotary factor: {getattr(self, 'partial_rotary_factor', 1.0)}")
        print(f"  Rotary ndims: {rotary_ndims}")
        print(f"  Expected length: {expected_length}")
        print(f"  Actual short_factor length: {len(rope_scaling_short_factor)}")
        print(f"  Actual long_factor length: {len(rope_scaling_long_factor)}")
        
        # Validate short_factor length
        if not len(rope_scaling_short_factor) == expected_length:
            raise ValueError(
                f"`rope_scaling`'s short_factor field must have length {expected_length}, got {len(rope_scaling_short_factor)}"
            )
            
        # Validate long_factor length  
        if not len(rope_scaling_long_factor) == expected_length:
            raise ValueError(
                f"`rope_scaling`'s long_factor field must have length {expected_length}, got {len(rope_scaling_long_factor)}"
            )

class Phi_LlavaModel(LlavaMetaModel, Phi3Model):
    config_class = PhiLlava_config

    def __init__(self, config: Phi3Config):
        super(Phi_LlavaModel, self).__init__(config)
        self.gradient_checkpointing = False

class PhiLlavaForCausalLM(Phi3ForCausalLM, LlavaMetaForCausalLM):
    config_class = PhiLlava_config

    def __init__(self, config):
        super().__init__(config)
        self.model = Phi_LlavaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.LongTensor:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
            )
        else:
            inputs_embeds = self.get_model().model.embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

# Register the tokenizer
def get_tokenizer():
    from transformers import AutoTokenizer
    def post_init(tokenizer):
        return tokenizer
    return AutoTokenizer, post_init

# Register configuration and model with Hugging Face
AutoConfig.register("Phi_llava", PhiLlava_config)
AutoModelForCausalLM.register(PhiLlava_config, PhiLlavaForCausalLM)


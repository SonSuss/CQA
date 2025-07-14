import torch
import torch.nn as nn

from typing import Optional, List, Tuple, Union

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


from models.chart_qa_model.model.configuration_phi3 import Phi3Config
from models.chart_qa_model.model.modeling_phi3 import Phi3ForCausalLM, Phi3Model
from models.components.llava import LlavaMetaForCausalLM, LlavaMetaModel


class PhiLlava_config(Phi3Config):
    model_type = "Phi_llava"

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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None and images is not None:
            # Only prepare multimodal inputs if we actually have images
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

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs
        )
        
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.LongTensor:
        if images is not None:
            # Only use custom logic when we have images
            position_ids = kwargs.pop("position_ids", None)
            attention_mask = kwargs.pop("attention_mask", None)
            
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                images,
            )
            
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        else:
            # For text-only generation, bypass any custom logic entirely
            # Just use the basic Phi3ForCausalLM generation without any overrides
            from models.chart_qa_model.model.modeling_phi3 import Phi3ForCausalLM
            return Phi3ForCausalLM.generate(self, input_ids=input_ids, **kwargs)

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


import torch
import torch.nn as nn

from typing import Optional, List, Tuple, Union

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaModel

# from transformers import BitsAndBytesConfig

from CQA.models.components.llava import LlavaMetaForCausalLM, LlavaMetaModel


# Define the custom configuration class for LLaMA 3
class SiglipLlamaConfig(LlamaConfig):
    model_type = "siglip_llama"

# Define the LLaMA-based multimodal model
class SiglipLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = SiglipLlamaConfig


    def __init__(self, config: LlamaConfig):
        super(SiglipLlamaModel, self).__init__(config)
        self.gradient_checkpointing = False

# Define the causal LM model wrapper
class SigLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = SiglipLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = SiglipLlamaModel(config)
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
AutoConfig.register("siglip_llama", SiglipLlamaConfig)
AutoModelForCausalLM.register(SiglipLlamaConfig, SigLlamaForCausalLM)




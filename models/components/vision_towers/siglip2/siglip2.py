import os
from typing import Union
from typing import Optional, Tuple, Union, Dict

import torch

from torch import nn
from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig, SiglipPreTrainedModel, SiglipVisionTransformer
from transformers.modeling_outputs import BaseModelOutputWithPooling

class SiglipVisionModel(SiglipPreTrainedModel):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)

        self.vision_model = SiglipVisionTransformer(config)
        del self.vision_model.encoder.layers[-1:]
        self.vision_model.head = nn.Identity()
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
            self,
            pixel_values,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class SigLip2VisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()
        self.is_loaded = False

        if vision_tower is not None:
            self.config = SiglipVisionConfig.from_pretrained(vision_tower)
        else:
            self.config = SiglipVisionConfig()

        self.vision_tower_name = vision_tower
        self.config.image_mean = [0.5, 0.5, 0.5]
        
        self.image_processor = SiglipImageProcessor(size=(self.config.image_size, self.config.image_size), image_mean=self.config.image_mean)
        
        self.layer_idx = getattr(vision_tower_cfg, "mm_vision_select_layer", -1)
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self):
        if self.is_loaded:
            return

        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)

        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        self.is_loaded = True
    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[self.layer_idx].to(image.dtype)

                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = image_forward_outs.hidden_states[self.layer_idx].to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
    
def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    return SigLip2VisionTower(vision_tower, vision_tower_cfg, **kwargs)
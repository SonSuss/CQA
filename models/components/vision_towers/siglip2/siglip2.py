import os
from typing import Union

import torch

from torch import nn
from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig


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
        self.layer_idx = getattr(self.vision_tower_cfg, "mm_vision_select_layer", -1)
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
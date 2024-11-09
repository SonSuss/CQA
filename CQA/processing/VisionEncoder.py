from transformers import AutoModel, AutoProcessor
import torch
import torch.nn as nn

class VisionEncoder(nn.Module):
    def __init__(self, model_name):
        super(VisionEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def forward(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        inputs = self.processor(images=image_tensor, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Add dummy text input
        inputs["input_ids"] = torch.tensor([[0]]).to(self.device)  # Dummy input_ids
        outputs = self.model(**inputs)
        # Assuming the output is a dictionary or a custom object with an attribute that contains the tensor
        if isinstance(outputs, dict):
            output_tensor = outputs[next(iter(outputs))]
        else:
            output_tensor = outputs.last_hidden_state  # Adjust based on the actual attribute
        # Reshape the output tensor to obtain patch features
        batch_size, num_patches, feature_dim = output_tensor.shape
        patch_features = output_tensor.view(batch_size, int(num_patches**0.5), int(num_patches**0.5), feature_dim)
        return patch_features

    def get_num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

vision_encoder_model = "google/siglip-base-patch16-512"
ViTEn = VisionEncoder(vision_encoder_model)
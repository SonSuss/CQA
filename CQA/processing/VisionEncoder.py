from transformers import AutoModel, AutoProcessor
import torch

class VisionEncoder:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def encode(self, image_tensor):
        inputs = self.processor(images=image_tensor, return_tensors="pt")
        # Add dummy text input
        inputs["input_ids"] = torch.tensor([[0]])  # Dummy input_ids
        outputs = self.model(**inputs)
        return outputs

# Create a dummy image tensor with random values
dummy_image_tensor = torch.rand(1, 3, 512, 512)  # Batch size of 1, 3 color channels, 384x384 image size

# Example usage
vision_encoder_model = "google/siglip-base-patch16-512"
ViTEn = VisionEncoder(vision_encoder_model)
outputs = ViTEn.encode(dummy_image_tensor)
print(outputs)

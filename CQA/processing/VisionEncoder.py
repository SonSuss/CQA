from transformers import AutoModel, AutoProcessor
import torch

class VisionEncoder:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def encode(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        inputs = self.processor(images=image_tensor, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Add dummy text input
        inputs["input_ids"] = torch.tensor([[0]]).to(self.device)  # Dummy input_ids
        outputs = self.model(**inputs)
        return outputs

    def get_num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

# Create a dummy image tensor with random values
dummy_image_tensor = torch.rand(1, 3, 512, 512)  # Batch size of 1, 3 color channels, 512x512 image size

# Example usage
vision_encoder_model = "google/siglip-base-patch16-512"
ViTEn = VisionEncoder(vision_encoder_model)
outputs = ViTEn.encode(dummy_image_tensor)
print(outputs)

num_parameters = ViTEn.get_num_parameters()
print(f"Number of parameters in the model: {num_parameters}")
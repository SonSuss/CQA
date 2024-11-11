import torch
from VisionEncoder import VisionTower

def test_vision_tower():
    # Create a sample input tensor representing an image of size 768x768 with 3 channels (RGB)
    image_size = 768
    in_channels = 3
    batch_size = 1
    sample_input = torch.randn(batch_size, in_channels, image_size, image_size)

    # Instantiate the VisionTower with the appropriate parameters
    hidden_size = 1152
    patch_size = 14
    num_patches = (image_size // patch_size) ** 2
    num_layers = 27
    mlp_dim = 4304
    r = 20

    vit = VisionTower(in_channels=in_channels, hidden_size=hidden_size, patch_size=patch_size, num_patches=num_patches, num_layers=num_layers, mlp_dim=mlp_dim, r=r)

    # Pass the input tensor through the VisionTower
    output = vit(sample_input)

    # Print the output shape
    print("Output shape:", output.shape)

    # Optionally, you can add assertions to check the output shape
    assert output.shape == (batch_size, hidden_size), "Output shape is incorrect"

if __name__ == "__main__":
    test_vision_tower()
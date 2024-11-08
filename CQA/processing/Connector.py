import torch
import torch.nn as nn
import torch.nn.functional as F

class Connector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Connector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

# Example usage
input_dim = 512  # Example input dimension from vision encoder
hidden_dim = 256  # Example hidden dimension
output_dim = 768  # Example output dimension for LLM

connector = Connector(input_dim, hidden_dim, output_dim)

# Create a dummy input tensor
dummy_input = torch.rand(1, input_dim)  # Batch size of 1

# Forward pass through the connector
output = connector(dummy_input)
print(output.shape)
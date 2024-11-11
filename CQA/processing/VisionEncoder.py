import torch
import torch.nn as nn
import torch.nn.functional as F
from ViTmerging import VisionTokenMerger

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size, num_patches):
        super(SiglipVisionEmbeddings, self).__init__()
        self.patch_embedding = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, padding=0)
        self.position_embedding = nn.Embedding(num_patches, hidden_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size, num_channels, height, width = x.size()
        x = x.flatten(2).transpose(1, 2)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)
        return x

class SiglipSdpaAttention(nn.Module):
    def __init__(self, hidden_size, r):
        super(SiglipSdpaAttention, self).__init__()
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.token_merger = VisionTokenMerger(hidden_size, r)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Merge tokens based on similarity
        K_merged, num_patches_per_token = self.token_merger(K)

        # Adjust attention scores based on the number of patches each token represents
        attn_scores = torch.matmul(Q, K_merged.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        attn_scores = attn_scores + torch.log(torch.tensor(num_patches_per_token, dtype=torch.float32, device=attn_scores.device))
        attn_weights = F.softmax(attn_scores, dim=-1)
        # Merge the value tensor V to match the merged keys
        V_merged, _ = self.token_merger(V)
        output = torch.matmul(attn_weights, V_merged)
        return self.out_proj(output)

class SiglipMLP(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super(SiglipMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.activation_fn = nn.GELU()

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x

class SiglipEncoderLayer(nn.Module):
    def __init__(self, hidden_size, mlp_dim, r):
        super(SiglipEncoderLayer, self).__init__()
        self.self_attn = SiglipSdpaAttention(hidden_size, r)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = SiglipMLP(hidden_size, mlp_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        attn_output = self.self_attn(self.layer_norm1(x))
        x = x + attn_output
        mlp_output = self.mlp(self.layer_norm2(x))
        x = x + mlp_output
        return x

class SiglipEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, mlp_dim, r):
        super(SiglipEncoder, self).__init__()
        self.layers = nn.ModuleList([SiglipEncoderLayer(hidden_size, mlp_dim, r) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SiglipVisionTransformer(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size, num_patches, num_layers, mlp_dim, r):
        super(SiglipVisionTransformer, self).__init__()
        self.embeddings = SiglipVisionEmbeddings(in_channels, hidden_size, patch_size, num_patches)
        self.encoder = SiglipEncoder(num_layers, hidden_size, mlp_dim, r)
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.head = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        x = self.post_layernorm(x)
        x = self.head(x.mean(dim=1))
        return x

class SiglipVisionModel(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size, num_patches, num_layers, mlp_dim, r):
        super(SiglipVisionModel, self).__init__()
        self.vision_model = SiglipVisionTransformer(in_channels, hidden_size, patch_size, num_patches, num_layers, mlp_dim, r)

    def forward(self, x):
        return self.vision_model(x)

class VisionTower(nn.Module):
    def __init__(self, in_channels=3, hidden_size=1152, patch_size=14, num_patches=2916, num_layers=27, mlp_dim=4304, r=10):
        super(VisionTower, self).__init__()
        self._vision_tower = SiglipVisionModel(in_channels, hidden_size, patch_size, num_patches, num_layers, mlp_dim, r)

    def forward(self, x):
        return self._vision_tower(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTokenMerger(nn.Module):
    def __init__(self, hidden_size, r):
        super(VisionTokenMerger, self).__init__()
        self.hidden_size = hidden_size
        self.r = r

    def forward(self, K):
        batch_size, num_tokens, hidden_size = K.size()
        assert hidden_size == self.hidden_size, "Hidden size mismatch"

        # Compute cosine similarity between tokens
        K_norm = F.normalize(K, p=2, dim=-1)
        similarity_matrix = torch.matmul(K_norm, K_norm.transpose(-2, -1))

        # Divide tokens into two disjoint sets
        set1_indices = torch.arange(0, num_tokens, 2, device=K.device)
        set2_indices = torch.arange(1, num_tokens, 2, device=K.device)

        set1 = K[:, set1_indices, :]
        set2 = K[:, set2_indices, :]

        # Compute similarity between tokens in set1 and set2
        set1_norm = F.normalize(set1, p=2, dim=-1)
        set2_norm = F.normalize(set2, p=2, dim=-1)
        similarity = torch.matmul(set1_norm, set2_norm.transpose(-2, -1))

        # Find top-r most similar pairs
        top_r_values, top_r_indices = torch.topk(similarity.view(batch_size, -1), self.r, dim=-1)

        # Merge the features of the two endpoints through average pooling
        merged_tokens = []
        for b in range(batch_size):
            merged_set = []
            for idx in top_r_indices[b]:
                i = idx // set2.size(1)
                j = idx % set2.size(1)
                merged_token = (set1[b, i, :] + set2[b, j, :]) / 2
                merged_set.append(merged_token)
            merged_tokens.append(torch.stack(merged_set))

        merged_tokens = torch.stack(merged_tokens)

        return merged_tokens, num_tokens // 2
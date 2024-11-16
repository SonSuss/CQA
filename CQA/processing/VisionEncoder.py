import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.models.siglip.modeling_siglip import SiglipMLP, SiglipConfig, SiglipSdpaAttention
from ViTmerging import VisionTokenMerger
class SiglipSdpaAttentionWithVTM(SiglipSdpaAttention):
    def __init__(self, config: SiglipConfig, r=20):
        super().__init__(config)
        self.config = config
        self.r = r  # Number of token pairs to merge per layer
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        s: torch.Tensor,  # New input for token size vector
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = hidden_states.size()

        # Project hidden states to obtain queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        # Apply Visual Token Merging
         # Apply Visual Token Merging (VTM) on keys and values
        merge_fn, _ = self.token_merger.bipartite_soft_matching(
            metric=key_states.mean(dim=-1),  # Metric for matching, e.g., mean over heads
            r=self.r,
        )
        key_states, key_sizes = merge_fn(key_states)
        value_states, value_sizes = merge_fn(value_states)

        # Calculate attention scores with the addition of log(s)
        # Reshape `s` to align with attention scores dimensions
        log_s = torch.log(key_sizes).view(batch_size, 1, 1, -1)  # Shape: (batch_size, 1, 1, seq_len)
        
        # Scaled dot-product attention with `log s`
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_scores += log_s  # Add log(s) to attention scores

        # Apply attention mask, if provided
        if attention_mask is not None:
            attention_scores += attention_mask

        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Multiply with values to get the output
        attn_output = torch.matmul(attention_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.embed_dim)

        # Final output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, None

class CustomSiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipConfig, r: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipSdpaAttentionWithVTM(config, r=r)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
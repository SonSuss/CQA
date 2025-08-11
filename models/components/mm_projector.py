import math
import torch
import re
from typing import Union, Optional

from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}
# class ResamplerBlock(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int = 768,
#         image_hidden_size: int = 1024,
#         num_heads: int = 12,
#         intermediate_size: int = None,
#         attn_dropout: float = 0.1,
#         ffn_dropout: float = 0.1
#     ):
#         super().__init__()
#         assert hidden_size % num_heads == 0, "For MHSA, you must have number of heads divisible by initial hidden size"
#         intermediate_size = hidden_size * 4 if intermediate_size is None else intermediate_size
#         # intermediate_size = hidden_size * 4
#         self.scale = 1 / math.sqrt(hidden_size // num_heads)
#         self.num_heads = num_heads
#         self.to_q = nn.Linear(hidden_size, hidden_size, bias=True)
#         self.to_k = nn.Linear(image_hidden_size, hidden_size, bias=True)
#         self.to_v = nn.Linear(image_hidden_size, hidden_size, bias=True)

#         self.to_out = nn.Linear(hidden_size, hidden_size, bias=True)

#         self.attn_dropout = nn.Dropout(attn_dropout)
#         self.ffn_norm = nn.LayerNorm(hidden_size)
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_size, intermediate_size, bias=True),
#             nn.GELU(),
#             nn.Dropout(ffn_dropout),
#             nn.Linear(intermediate_size, hidden_size, bias=True),
#             nn.Dropout(ffn_dropout)
#         )
#         # prenorm for image features
#         self.norm_image = nn.LayerNorm(image_hidden_size)
#         self.norm_hidden = nn.LayerNorm(hidden_size)

#     def forward(self, hidden_states: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#         x = self.norm_image(x)
#         residual_hidden_states = hidden_states
#         hidden_states = self.norm_hidden(hidden_states)

#         # Attention
#         q = self.to_q(hidden_states)
#         k = self.to_k(x)
#         v = self.to_v(x)

#         q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
#         k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
#         v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

#         scores = torch.einsum("... i d, ... j d -> ... i j", q * self.scale, k)
#         scores = scores - scores.amax(dim=-1, keepdim=True).detach()
#         attn = self.attn_dropout(scores.softmax(dim=-1))

#         out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
#         out = rearrange(out, "b h n d -> b n (h d)", h=self.num_heads)
#         out = self.to_out(out) + residual_hidden_states

#         # FFN
#         residual = out
#         out = self.ffn_norm(out)
#         out = self.ffn(out)
#         out = residual + out
#         return out


# class Resampler(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int = 768,
#         image_hidden_size: int = 1024,
#         final_hidden_size: int = 4096,
#         num_heads: int = 12,
#         intermediate_size: int = None,
#         num_queries: int = 128,
#         num_layers: int = 3,
#         # initializer_range: float = 0.02,
#         delay_load: bool = False,
#         attn_dropout: float = 0.05,
#         ffn_dropout: float = 0.05
#     ):
#         super().__init__()
#         self.resampler_blocks = nn.ModuleList(
#             [
#                 ResamplerBlock(
#                     hidden_size, image_hidden_size, num_heads, intermediate_size,
#                     attn_dropout=attn_dropout, ffn_dropout=ffn_dropout
#                 ) for _ in range(num_layers)
#             ]
#         )
#         self.queries = nn.Parameter(torch.zeros(num_queries, hidden_size))
#         self.pos_embed = nn.Parameter(torch.zeros(num_queries, hidden_size))
#         self.query_scale = nn.Parameter(torch.ones(1) * 0.1)
#         self.post_norm = nn.LayerNorm(hidden_size)
#         self.final_proj = nn.Linear(hidden_size, final_hidden_size, bias=False)

#         if not delay_load:
#             # self.initializer_range = initializer_range
#             for module in self.modules():
#                 if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
#                     self._init_weights(module)
    
#     def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
#         """Improved weight initialization"""
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             if hasattr(module, 'weight') and module.weight is not None:
#                 nn.init.xavier_uniform_(module.weight.data)
#                 if hasattr(module, 'bias') and module.bias is not None:
#                     module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             if hasattr(module, 'bias') and module.bias is not None:
#                 module.bias.data.zero_()
#             if hasattr(module, 'weight') and module.weight is not None:
#                 module.weight.data.fill_(1.0)

#     def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
#         b = image_hidden_states.size(0)
#         queries = repeat(self.queries * self.query_scale, 'n d -> b n d', b=b)
#         pos_embed = repeat(self.pos_embed, 'n d -> b n d', b=b)
#         queries = queries + pos_embed
        
#         for resampler_block in self.resampler_blocks:
#             queries = resampler_block(queries, image_hidden_states)

#         queries = self.post_norm(queries)
#         return self.final_proj(queries)

class ResamplerBlock(nn.Module):
    """
    Multi-Query Cross-Attention Resampler Block with gated residuals and stable pre-norms.
    - num_kv_heads: how many KV heads to compute (1 == classic multi-query attention).
    - Uses a learnable gate to blend attention output with residual.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        image_hidden_size: int = 1024,
        num_heads: int = 12,
        num_kv_heads: int = 1,   # multi-query default
        intermediate_size: Optional[int] = None,
        attn_dropout: float = 0.05,
        ffn_dropout: float = 0.05,
        use_bottleneck: bool = True,  # optional bottleneck projection for image features
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        intermediate_size = hidden_size * 4 if intermediate_size is None else intermediate_size

        # optional bottleneck to reduce KV projection size from image_hidden_size -> hidden_size
        if use_bottleneck and image_hidden_size != hidden_size:
            self.image_bottleneck = nn.Linear(image_hidden_size, hidden_size, bias=False)
            self.norm_image_pre = nn.LayerNorm(image_hidden_size)  # norm then bottleneck
        else:
            self.image_bottleneck = None
            self.norm_image_pre = nn.LayerNorm(image_hidden_size)

        # Query, KV projections
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=True)

        # For KV we compute num_kv_heads * head_dim outputs then expand to heads
        self.to_kv = nn.Linear(hidden_size if self.image_bottleneck is not None else image_hidden_size,
                               (self.num_kv_heads * self.head_dim) * 2,
                               bias=True)

        # output projection
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=True)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=True),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(intermediate_size, hidden_size, bias=True),
            nn.Dropout(ffn_dropout)
        )

        # prenorm for hidden queries
        self.norm_hidden = nn.LayerNorm(hidden_size)

        # gated residual parameter (scalar per block)
        self.gate_param = nn.Parameter(torch.tensor(0.0))

        # small projection to stabilize attention outputs if needed
        self.attn_out_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        # hidden_states: (b, n_q, hidden_size)
        # image_features: (b, n_img, image_hidden_size)

        # normalize image then optional bottleneck
        image_features = self.norm_image_pre(image_features)
        if self.image_bottleneck is not None:
            # project into hidden_size
            image_proj = self.image_bottleneck(image_features)  # (b, n_img, hidden_size)
        else:
            image_proj = image_features  # (b, n_img, image_hidden_size) expected to equal hidden_size

        # prenorm hidden/query
        residual = hidden_states
        q_in = self.norm_hidden(hidden_states)
        q = self.to_q(q_in)  # (b, n_q, hidden_size)
        b, n_q, _ = q.shape
        b, n_img, _ = image_proj.shape

        # to_kv => (b, n_img, 2 * num_kv_heads * head_dim)
        kv = self.to_kv(image_proj)
        kv = rearrange(kv,
                "b n (two h d) -> two b h n d",
                two=2,
                h=self.num_kv_heads,
                d=self.head_dim
            )
        k, v = kv[0], kv[1]  # each: (b, num_kv_heads, n_img, head_dim)

        # make queries per head: (b, num_heads, n_q, head_dim)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)

        # Expand K/V to full heads if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            # broadcast along head dimension
            k = k.unsqueeze(1) if k.dim() == 3 else k  # ensure dims
            # currently k: (b, num_kv_heads, n_img, head_dim)
            k = repeat(k, "b h n d -> b (h repeat) n d", repeat=(self.num_heads // self.num_kv_heads))
            v = repeat(v, "b h n d -> b (h repeat) n d", repeat=(self.num_heads // self.num_kv_heads))
        # now k,v shapes: (b, num_heads, n_img, head_dim)

        # scaled dot product
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("b h i d, b h j d -> b h i j", q * scale, k)
        scores = scores - scores.amax(dim=-1, keepdim=True).detach()  # numeric stability
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=self.num_heads)

        out = self.to_out(out)  # (b, n_q, hidden_size)
        out = self.attn_out_norm(out)

        # gated residual: sigmoid(g) * out + (1 - sigmoid(g)) * residual
        gate = torch.sigmoid(self.gate_param)
        hidden_after_attn = residual + gate * out

        # FFN (post-norm)
        residual_ffn = hidden_after_attn
        ffn_in = self.ffn_norm(hidden_after_attn)
        ffn_out = self.ffn(ffn_in)
        out_final = residual_ffn + ffn_out
        return out_final


class Resampler(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        image_hidden_size: int = 1024,
        final_hidden_size: int = 4096,
        num_heads: int = 12,
        num_kv_heads: int = 1,         # multi-query default
        intermediate_size: Optional[int] = None,
        num_queries: int = 128,
        num_layers: int = 3,
        attn_dropout: float = 0.05,
        ffn_dropout: float = 0.05,
        use_bottleneck: bool = True,
        delay_load: bool = False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_scale = nn.Parameter(torch.ones(1) * 0.1)

        # learnable queries and pos embed
        self.queries = nn.Parameter(torch.empty(num_queries, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(num_queries, hidden_size))

        # init queries with xavier (helps convergence)
        nn.init.xavier_uniform_(self.queries)

        # build stack of blocks
        self.blocks = nn.ModuleList([
            ResamplerBlock(
                hidden_size=hidden_size,
                image_hidden_size=image_hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
                use_bottleneck=use_bottleneck,
            ) for _ in range(num_layers)
        ])

        self.post_norm = nn.LayerNorm(hidden_size)
        self.final_proj = nn.Linear(hidden_size, final_hidden_size, bias=False)

        if not delay_load:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
                    self._init_weights(module)

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.xavier_uniform_(module.weight.data)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        b = image_hidden_states.size(0)
        queries = repeat(self.queries * self.query_scale, "n d -> b n d", b=b)
        pos_embed = repeat(self.pos_embed, "n d -> b n d", b=b)
        queries = queries + pos_embed

        for block in self.blocks:
            queries = block(queries, image_hidden_states)

        queries = self.post_norm(queries)
        out = self.final_proj(queries)  # (b, num_queries, final_hidden_size)
        return out

def build_vision_projector(config, delay_load=False):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'resampler':
        hidden_size = getattr(config, 'resampler_hidden_size', 768)
        image_hidden_size = config.mm_hidden_size
        num_queries = getattr(config, 'num_queries', 128)
        final_hidden_size = config.hidden_size
        num_heads = 12
        if hidden_size == 512:
            num_heads = 8
        num_layers = getattr(config, 'num_resampler_layers', 3)

        
        print(
            f"resampler config: resampler hidden size: {hidden_size}, num_queries: {num_queries}, "
            f"num_resampler_layers: {num_layers}"
        )
        return Resampler(
            hidden_size=hidden_size,
            image_hidden_size=image_hidden_size,
            num_queries=num_queries,
            final_hidden_size=final_hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=1,
            use_bottleneck=True,
            delay_load=delay_load,
        )

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        mlp = nn.Sequential(*modules)
        if getattr(config, 'load_moe_mm_projector', False):
            from deepspeed.moe.layer import MoE
            mlp = MoE(
                config.mm_hidden_size,
                expert=mlp,
                num_experts=4,
                ep_size=1,
                k=2,
                capacity_factor=1.,
                eval_capacity_factor=1.,
                min_capacity=4,
                use_residual=False,
            )

            def moe_forward_wrapper(forward_func):
                return lambda *args, **kwargs: forward_func(*args, **kwargs)[0]
            mlp.forward = moe_forward_wrapper(mlp.forward)
        return mlp

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

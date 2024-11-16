import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Callable

class VisionTokenMerger(nn.Module):
    def __init__(self, r=20):
        super(VisionTokenMerger, self).__init__()
        self.r = r


    def bipartite_soft_matching(self,
        metric: torch.Tensor,
        r: int,
        class_token: bool = False,
        distill_token: bool = False,
    ) -> Tuple[Callable, Callable]:
        """
        Applies ToMe with a balanced matching set (50%, 50%).

        Input size is [batch, tokens, channels].
        r indicates the number of tokens to remove (max 50% of tokens).

        Extra args:
        - class_token: Whether or not there's a class token.
        - distill_token: Whether or not there's also a distillation token.

        When enabled, the class token and distillation tokens won't get merged.
        """
        protected = 0
        if class_token:
            protected += 1
        if distill_token:
            protected += 1

        # We can only reduce by a maximum of 50% tokens
        t = metric.shape[1]
        r = min(r, (t - protected) // 2)

        if r <= 0:
            return metric, metric

        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)

            if class_token:
                scores[..., 0, :] = -math.inf
            if distill_token:
                scores[..., :, 0] = -math.inf

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

            if class_token:
                # Sort to ensure the class token is at the start
                unm_idx = unm_idx.sort(dim=1)[0]

        def merge(x: torch.Tensor, mode="mean") -> Tuple[torch.Tensor, torch.Tensor]:
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape

            # Initialize sizes to track the number of patches
            sizes = torch.ones(n, t1, device=x.device)

            # Gather sizes for unmerged and merged tokens
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            # Update sizes for merged tokens
            unm_sizes = sizes.gather(dim=-1, index=unm_idx.squeeze(-1))
            src_sizes = sizes.gather(dim=-1, index=src_idx.squeeze(-1))
            dst_sizes = sizes.scatter_reduce(-1, dst_idx.squeeze(-1), src_sizes, reduce="sum")

            # Combine the tokens and sizes
            merged_tokens = torch.cat([unm, dst], dim=1)
            merged_sizes = torch.cat([unm_sizes, dst_sizes], dim=-1)

            return merged_tokens, merged_sizes

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            unm_len = unm_idx.shape[1]
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
            n, _, c = unm.shape

            src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

            out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

            out[..., 1::2, :] = dst
            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
            out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

            return out

        return merge, unmerge
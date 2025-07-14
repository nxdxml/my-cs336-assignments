from cs336_basics.RMSnorm import RMSnorm
from cs336_basics.Attention import MultiheadSelfAttention
from cs336_basics.SwiGLU import SwiGLU
import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self,
                    d_model: int,
                    num_heads: int,
                    d_ff: int,
                    max_seq_len: int,
                    theta: float,
                 ):
        """

        Args:
            d_model (int): Dimensionality of the feed-forward inner layer.
            num_heads (int): _description_
            d_ff (int): Dimensionality of the feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): theta (float): RoPE parameter.
            weights (dict):             
                State dict of our reference implementation.
                The keys of this dictionary are:
                - `attn.q_proj.weight`
                    The query projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
                - `attn.k_proj.weight`
                    The key projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
                - `attn.v_proj.weight`
                    The value projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_v),
                    so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
                - `attn.output_proj.weight`
                    Weight of the multi-head self-attention output projection
                    Shape is (d_model, d_model).
                - `ln1.weight`
                    Weights of affine transform for the first RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `ffn.w1.weight`
                    Weight of the first linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `ffn.w2.weight`
                    Weight of the second linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `ffn.w3.weight`
                    Weight of the third linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `ln2.weight`
                    Weights of affine transform for the second RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
        """        
        super().__init__()

        self.rms_norm1 = RMSnorm(d_model=d_model)

        self.multi_head_attn = MultiheadSelfAttention(
            d_model=d_model,
            num_head=num_heads,
            apply_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        self.rms_norm2 = RMSnorm(d_model=d_model)

        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        y = x + self.multi_head_attn(self.rms_norm1(x))
        output = y + self.swiglu(self.rms_norm2(y))

        return output
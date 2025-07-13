import torch
from torch import nn
from einops import rearrange
"""
Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input
tensor.
The following interface is recommended:
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) Construct the
RoPE module and create buffers if needed.
theta: float Θ value for the RoPE
d_k: int dimension of query and key vectors
max_seq_len: int Maximum sequence length that will be inputted
device: torch.device | None = None Device to store the buffer on
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor Process
an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
that you should tolerate x with an arbitrary number of batch dimensions. You should assume
that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
x along the sequence dimension.
You should use the token positions to slice your (possibly precomputed) cos and sin tensors along
the sequence dimension.
To test your implementation, complete [adapters.run_rope] and make sure it passes uv run
pytest -k test_rope.

"""

class Rope(nn.Module):
    def __init__(self,
                 theta : float,
                 d_k : int,
                 max_seq_len : int,
                 device: torch.device | None = None,
                 ):
        """

        Args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device | None, optional): 设备
        """        
        super().__init__()
        # 使用enisum简化计算
        inv_freqs = 1.0 / theta ** (torch.arange(0, d_k, 2).float() / d_k) # [d_k // 2]
        pos = torch.arange(max_seq_len).float() # [maxseq]
        freqs = torch.einsum("i,j -> ij", pos, inv_freqs) # [maxseq, d_k // 2]
        # 得到对应cos,sin表
        cos = freqs.cos() # [maxseq, d_k // 2]
        sin = freqs.sin() # [maxseq, d_k // 2]
        # 避免成为参数，仍能保存到模型
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x : torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): (..., seq_len, d_k)
            token_positions (torch.Tensor): (..., seq_len)

        Returns:
            torch.Tensor: (..., seq_len, d_k)
        """

        x0 = x[..., 0::2] # 0 2 4 6 ...
        x1 = x[..., 1::2] # 1 3 5 6 ...

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x0_rote = cos * x0 - sin * x1
        x1_rote = sin * x0 + cos * x1

        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x0_rote
        x_out[..., 1::2] = x1_rote

        return x_out
                
import torch
from einops import einsum, rearrange
import math
from torch import nn
from cs336_basics.Linear import Linear
from cs336_basics.Rope import Rope


def softmax(x : torch.Tensor, dim : int) -> torch.Tensor:
    """softmax

    Args:
        x (torch.Tensor): Float[Tensor, "..."]  
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        torch.Tensor: Tensor of with the same shape as `in_features` with the output of
    """    
    # keepdim 保留原有维度
    # (values, indices)
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max
    x_exp = torch.exp(x_stable)
    output = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return output


def scaled_dot_product_attention(
        Q : torch.Tensor,
        K : torch.Tensor,
        V : torch.Tensor,
        mask : torch.Tensor | None = None,
) -> torch.Tensor :
    """scaled_dot_product_attention

    Args:
        Q (torch.Tensor): Float[Tensor, " ... queries d_k"]
        K (torch.Tensor): Float[Tensor, " ... keys d_k"]
        V (torch.Tensor): Float[Tensor, " ... values d_v"]
        mask (torch.Tensor | None, optional): Float[Tensor, " ... queries keys"] | None

    Returns:
        torch.Tensor: Float[Tensor, " ... queries d_v"]
    """    
    d_k = Q.shape[-1]
    att_weight = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    
    if mask is not None:
        att_weight = att_weight.masked_fill(mask==0, float("-inf"))

    output = softmax(att_weight, -1) @ V
    return output


class MultiheadSelfAttention(nn.Module):
    def __init__(self,
                 d_model : int, 
                 num_head : int, 
                 apply_rope : bool = False,
                 max_seq_len: int | None = None,
                 theta: float | None = None,
                 device: torch.device | str = "cpu", # 解决设备问题
                 ):
        """多头注意力,可以实现rope

        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_head (int): Number of heads to use in multi-headed attention.
        """        
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head
        self.d_v = self.d_k

        self.Q_proj = Linear(d_model, num_head * self.d_k, device=device)
        self.K_proj = Linear(d_model, num_head * self.d_k, device=device)
        self.V_proj = Linear(d_model, num_head * self.d_v, device=device)
        self.O_proj = Linear(num_head * self.d_v, d_model, device=device)


        self.apply_rope = apply_rope
        if apply_rope is True:
            self.rope = Rope(d_k=self.d_k, max_seq_len=max_seq_len, theta=theta)



    def forward(self, x : torch.Tensor, token_positions : torch.Tensor | None = None) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

        Returns:
            torch.Tensor: Float[Tensor, " ... sequence_length d_out"]
        """ 
        seq_len = x.shape[-2]
        # [..., seq, (num_head * d_k)]           

        # print(f"multi_atten{x.device}")

        Q = self.Q_proj(x);
        K = self.K_proj(x);
        V = self.V_proj(x);

        Q = rearrange(Q, "... seq (num_head d_k) -> ... num_head seq d_k", num_head = self.num_head)
        K = rearrange(K, "... seq (num_head d_k) -> ... num_head seq d_k", num_head = self.num_head)
        V = rearrange(V, "... seq (num_head d_v) -> ... num_head seq d_v", num_head = self.num_head)
        # 加入rope，注意要在变换维度后对每个头分别做rope
        if self.apply_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        # tril取下三角
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

        output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        output = rearrange(output, "... num_head seq d_v  -> ... seq (num_head d_v)")

        output = self.O_proj(output)
        
        return output
            
import torch
from einops import einsum
import math

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



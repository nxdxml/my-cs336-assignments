import torch

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
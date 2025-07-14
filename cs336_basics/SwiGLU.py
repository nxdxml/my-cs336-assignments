import torch
from torch import nn
from cs336_basics.Linear import Linear

def silu(x : torch.Tensor):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self,
                    d_model: int,
                    d_ff: int,
                    device : torch.device | None = None, 
                    dtype : torch.dtype | None = None,
                 ):
        """

        Args:
            d_model (int): _description_
            d_ff (int): _description_
            device (torch.device | None, optional): 设备
            dtype (torch.dtype | None, optional): 数据类型
        """
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        w3x = self.w3(x)
        silu = w1x * torch.sigmoid(w1x)
        return self.w2(silu * w3x)
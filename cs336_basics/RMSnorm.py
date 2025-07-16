"""
Deliverable: Implement RMSNorm as a torch.nn.Module. We recommend the following interface:
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
Construct the RMSNorm module. This function should accept the following parameters:
d_model: int Hidden dimension of the model
eps: float = 1e-5 Epsilon value for numerical stability
device: torch.device | None = None Device to store the parameters on
dtype: torch.dtype | None = None Data type of the parameters
def forward(self, x: torch.Tensor) -> torch.Tensor Process an input tensor of shape
(batch_size, sequence_length, d_model) and return a tensor of the same shape.
Note: Remember to upcast your input to torch.float32 before performing the normalization (and
later downcast to the original dtype), as described above.
To test your implementation, implement the test adapter at [adapters.run_rmsnorm]. Then, run uv
run pytest -k test_rmsnorm.
"""

import torch
from torch import nn


class RMSnorm(nn.Module):
    def __init__(self, 
                 d_model : int, 
                 eps : float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 ):
        """初始化
        Args:
            d_model (int): Hidden dimension of the model
            epse (float, optional): Epsilon value for numerical stability
            device (torch.device | None, optional): 设备
            dtype (torch.dtype | None, optional): 数据类型
        """        
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # 计算每个位置的均方根
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # shape (..., 1)

        # 归一化
        x_normed = x / rms  # shape (..., d_model)

        # 乘以可训练参数 g
        output = x_normed * self.g  # 广播乘法，g形状(d_model,)自动广播

        return output
    
    # def forward(self, x : torch.Tensor) -> torch.Tensor:
    #     """RMSnorm前向过程
    #     Args:
    #         x (torch.Tensor): (batch_size, sequence_length, d_model)

    #     Returns:
    #         torch.Tensor: (batch_size, sequence_length, d_model)
    #     """        

    #     # You should upcast your input to torch.float32 to prevent overflow when you square the input.
    #     in_dtype = x.dtype
    #     x = x.to(torch.float32)

    #     # RMSnorm here
    #     batch_size, sequence_length, d_model = x.shape
    #     result = torch.empty(batch_size, sequence_length, d_model)
    #     for i, batch in enumerate(x):
    #         for j, seq in enumerate(batch):
    #             # 分母
    #             d = 0
    #             for k, a in enumerate(seq):
    #                 d += a * a
    #             d = (d / d_model + self.eps) ** 0.5
    #             for k, a in enumerate(seq):
    #                 result[i][j][k] = a / d * self.g[k]



    #     return result.to(in_dtype)

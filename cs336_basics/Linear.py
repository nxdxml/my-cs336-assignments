import torch
from torch import nn
class Linear(nn.Module):
    """y=Wx
    """    
    # subclass nn.Module
    def __init__(self, 
                 in_features : int,
                 out_features : int,
                 device : torch.device | None = None, 
                 dtype : torch.dtype | None = None):
        """

        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device | None, optional): 设备
            dtype (torch.dtype | None, optional): 数据类型
        """        # call the superclass constructor
        super().__init__()
        # construct and store your parameter as W (not WT) for memory ordering reasons
        # putting it in an nn.Parameter
        # 等价于 self.register_parameter('W', nn.Parameter(...))
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # torch.nn.init.trunc_normal_ to initialize the weights.
        mean = 0
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=mean, std=std, a=-3*std, b=3*std)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x @ self.W.T




        
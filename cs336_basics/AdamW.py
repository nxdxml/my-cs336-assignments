from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps = 1e-8, weight_decay = 0.1):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1":betas[0], "beta2":betas[1], "eps":eps, "weight_decay" : weight_decay}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # 获取该参数对应的状态字典，用于保存m, v, t
                
                t = state.get("t", 1) # 获取当前迭代次数t，如果是第一次则初始化为1，远小于m,v算内存不考虑开销
                m = state.get("m", torch.zeros_like(p.data)) # 一阶动量，和数据一样大
                v = state.get("v", torch.zeros_like(p.data)) # 二阶动量，和数据一样大
                
                grad = p.grad.data # 获取当前参数的梯度
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad

                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                
                p.data -= alpha_t * m / (v.sqrt() + eps)
                p.data -= lr * weight_decay * p.data

                # 更新状态：迭代步数 +1，保存新的动量m和v
                state["t"] = t + 1 
                state["m"] = m
                state["v"] = v
        return loss



class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p] # Get state associated with p.
            t = state.get("t", 0) # Get iteration number from the state, or initial value.
            grad = p.grad.data # Get the gradient of loss with respect to p.
            p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
            state["t"] = t + 1 # Increment iteration number.
        return loss
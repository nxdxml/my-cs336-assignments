
import torch
from torch import Tensor
from jaxtyping import Float, Int
from typing import Iterable
from typing import IO, Any, BinaryIO
import math
import numpy.typing as npt
import numpy as np
import os

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    batch_size, _ = inputs.shape

    # 数值稳定性
    input_stable = inputs - inputs.max(dim=-1, keepdim=True).values

    # 对每个样本求 log(∑_j exp(logit_j)) [batch]
    log_sum_exp = torch.logsumexp(input_stable, dim=-1)

    # 返回正确标签logit [batch]
    target_logits = input_stable[torch.arange(batch_size), targets]

    # loss [batch]
    loss = -(target_logits - log_sum_exp)
    return loss.mean()


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    学习率动态变化
    训练初期使用 较大的学习率，可以快速收敛；
    随着训练进行，逐渐减小学习率，从而更加精细地调整参数。
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    if it < warmup_iters:
        alpha = it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        alpha = min_learning_rate + 0.5 * (1.0 + \
            math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * \
            (max_learning_rate - min_learning_rate)
    else :
        alpha = min_learning_rate
    return alpha


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    在训练过程中，有时某些样本会导致模型产生非常大的梯度，这会破坏训练的稳定性（如导致梯度爆炸）。为了解决这个问题，进行梯度剪裁
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6
    sum_norm_sq = 0
    # norm(x) x范数
    # .item() Tensor -> float
    for params in parameters:
        if params.grad is not None:
            sum_norm_sq += (params.grad.data ** 2).sum().item()
    
    sum_norm = sum_norm_sq ** 0.5

    if sum_norm > max_l2_norm:
        scale_factor = max_l2_norm / (sum_norm + eps)
        for params in parameters:
            if params.grad is not None:
                # mul_ 原地操作
                params.grad.data.mul_(scale_factor)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从长序列x中采样一个batch的训练数据。
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = len(dataset) # 数据总长度
    high = n - context_length # 最后可用的起点
    # 生成batch_size个起始位置
    start_idx = np.random.randint(low=0, high=high, size=batch_size)

    input_batch = np.stack([dataset[i : i + context_length] for i in start_idx])
    target_batch = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_idx])

    # 转到tensor放到对应的设备上
    input_batch = torch.tensor(input_batch, dtype=torch.long, device=device)
    target_batch = torch.tensor(target_batch, dtype=torch.long, device=device)
    
    return input_batch, target_batch





def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    保存 模型、优化器、当前训练迭代次数 到指定路径下
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model. 
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    # 需要保存的东西
    checkpoint = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "iteration" : iteration
    }

    torch.save(checkpoint, out)




def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    加载 模型、优化器、当前训练迭代次数 从指定路径
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
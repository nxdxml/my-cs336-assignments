
import torch
from torch import Tensor
from jaxtyping import Float, Int
import math

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
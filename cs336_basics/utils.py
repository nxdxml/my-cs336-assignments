
import torch
from torch import Tensor
from jaxtyping import Float, Int
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
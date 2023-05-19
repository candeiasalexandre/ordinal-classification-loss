from typing import Optional
import torch


def non_diff_ordinal_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1,
    reduction: Optional[str] = "mean",
) -> torch.Tensor:
    ce = torch.nn.functional.cross_entropy(input, target, reduction="none")

    reg = target - input.argmax(dim=1)
    reg = torch.abs(reg)

    loss = ce + alpha * reg

    if reduction is None or reduction == "none":
        return loss

    if reduction == "sum":
        return loss.sum()

    if reduction == "mean":
        return loss.mean()

    raise ValueError("reduction should be None, sum or mean!")


def _create_weight_matrix(num_classes: int) -> torch.Tensor:
    W_matrix = []
    for i in range(num_classes):
        W_matrix.append([abs(i - j) for j in range(num_classes)])

    return torch.tensor(W_matrix)


def diff_ordinal_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1,
    reduction: Optional[str] = "mean",
) -> torch.Tensor:

    num_classes = input.shape[1]
    weights_matrix = _create_weight_matrix(num_classes)

    ce = torch.nn.functional.cross_entropy(input, target, reduction="none")

    prob = torch.nn.functional.softmax(input, dim=1)
    reg = weights_matrix[target, :] * prob
    reg = reg.sum(dim=1)

    loss = ce + alpha * reg

    if reduction is None or reduction == "none":
        return loss

    if reduction == "sum":
        return loss.sum()

    if reduction == "mean":
        return loss.mean()

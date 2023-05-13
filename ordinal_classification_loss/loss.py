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

import torch
from dataclasses import dataclass


@dataclass
class ClassificationOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor


class SimpleLinearClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, num_class: int) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(input_dim, num_class)

    def forward(self, x: torch.Tensor) -> ClassificationOutput:
        logits = self._linear(x)
        prob = torch.nn.functional.softmax(logits, dim=1)
        return ClassificationOutput(logits, prob)

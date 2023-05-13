import pytest
from ordinal_classification_loss.model import SimpleLinearClassifier
import torch


@pytest.fixture()
def input_dim() -> int:
    return 32


@pytest.fixture()
def number_classes() -> int:
    return 5


@pytest.fixture()
def input_data() -> torch.Tensor:
    input_data = torch.tensor([[0.1] * 32, [0.2] * 32, [1.2] * 32])
    return input_data


@pytest.fixture()
def label_data() -> torch.Tensor:
    label_data = torch.tensor([0, 1, 2])
    return label_data


def test_model_forward(
    input_data: torch.Tensor, input_dim: torch.Tensor, number_classes: torch.Tensor
) -> None:
    input = input_data
    model = SimpleLinearClassifier(input_dim, number_classes)
    out = model.forward(input)

    assert out.probabilities.shape[1] == number_classes

import pytest
from ordinal_classification_loss.loss import (
    non_diff_ordinal_cross_entropy,
    _create_weight_matrix,
    diff_ordinal_cross_entropy,
)
from torch.nn.functional import cross_entropy
import torch


@pytest.fixture()
def logits() -> torch.Tensor:
    return torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.5, 0.4, 0.3, 0.2],
        ]
    )


@pytest.fixture()
def labels() -> torch.Tensor:
    return torch.tensor([0, 0, 0])



def assert_is_not_close(x: torch.Tensor, y: torch.Tensor) -> None:
    assert not torch.all(torch.isclose(x, y))

def test_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> None:
    loss = cross_entropy(logits, labels, reduce=False)
    print(f"cross_entropy loss: \n {loss}")
    assert loss is not None


class TestOrdinalCrossEntropy:
    @pytest.mark.parametrize(
        "loss_fn", [non_diff_ordinal_cross_entropy, diff_ordinal_cross_entropy]
    )
    def test_alpha_0(self, loss_fn, logits: torch.Tensor, labels: torch.Tensor) -> None:
        loss_ce = cross_entropy(logits, labels, reduce=False)
        loss_ordinal_ce = loss_fn(logits, labels, reduction="none", alpha=0)

        torch.testing.assert_close(loss_ce, loss_ordinal_ce)

    @pytest.mark.parametrize(
        "loss_fn, expected_loss",
        [
            (
                non_diff_ordinal_cross_entropy,
                torch.tensor([1.8194, 1.4194, 1.8194]) + torch.tensor([4, 0, 1]),
            ),
            (
                diff_ordinal_cross_entropy,
                # ce + reg calculated for W0 * softmax(logits)
                torch.tensor([1.8194, 1.4194, 1.8194]) + torch.tensor([2.1991, 1.8009, 1.9903]),
            ),
        ],
    )
    def test_values(
        self, loss_fn, expected_loss, logits: torch.Tensor, labels: torch.Tensor
    ) -> None:

        loss = loss_fn(logits, labels, reduction="none")

        print(f"{loss_fn.__name__} loss: \n {loss}")
        torch.testing.assert_close(loss, expected_loss, atol=1E-4, rtol=1E-4)

    @pytest.mark.parametrize(
        "loss_fn, assert_fn",
        [
            (non_diff_ordinal_cross_entropy, torch.testing.assert_close),
            (diff_ordinal_cross_entropy, assert_is_not_close),
        ],
    )
    def test_gradient(
        self, loss_fn, assert_fn, logits: torch.Tensor, labels: torch.Tensor
    ) -> None:
        logits_ce = torch.tensor(logits.numpy(), requires_grad=True)
        loss_ce = cross_entropy(logits_ce, labels, reduction="mean")
        loss_ce.backward()
        grad_ce = logits_ce.grad

        logits_oce = torch.tensor(logits.numpy(), requires_grad=True)
        loss_ordinal_ce = loss_fn(logits_oce, labels, reduction="mean", alpha=1)
        loss_ordinal_ce.backward()
        grad_oce = logits_oce.grad

        print(f"Gradient cross entropy: \n {grad_ce}")
        print(f"Gradient {loss_fn.__name__}: \n {grad_oce}")

        assert_fn(grad_ce, grad_oce)


def test_create_weight_matrix() -> None:
    num_classes = 3
    expected_matrix = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    matrix = _create_weight_matrix(num_classes)
    torch.testing.assert_close(matrix, expected_matrix)

import pytest
from ordinal_classification_loss.loss import non_diff_ordinal_cross_entropy
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


def test_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> None:
    loss = cross_entropy(logits, labels, reduce=False)
    print(f"cross_entropy loss: \n {loss}")
    assert loss is not None


class TestNonDiffOrdinalCrossEntropy:
    def test_alpha_0(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        loss_ce = cross_entropy(logits, labels, reduce=False)
        loss_ordinal_ce = non_diff_ordinal_cross_entropy(
            logits, labels, reduction="none", alpha=0
        )

        torch.testing.assert_close(loss_ce, loss_ordinal_ce)

    def test_values(self, logits: torch.Tensor, labels: torch.Tensor) -> None:

        loss = non_diff_ordinal_cross_entropy(logits, labels, reduction="none")

        print(f"non_diff_ordinal_cross_entropy loss: \n {loss}")

        expected_loss = cross_entropy(logits, labels, reduce=False) + torch.tensor(
            [4, 0, 1]
        )
        torch.testing.assert_close(loss, expected_loss)

    def test_gradient(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits_ce = torch.tensor(logits.numpy(), requires_grad=True)
        loss_ce = cross_entropy(logits_ce, labels, reduction="mean")
        loss_ce.backward()
        grad_ce = logits_ce.grad

        logits_oce = torch.tensor(logits.numpy(), requires_grad=True)
        loss_ordinal_ce = non_diff_ordinal_cross_entropy(
            logits_oce, labels, reduction="mean", alpha=1
        )
        loss_ordinal_ce.backward()
        grad_oce = logits_oce.grad

        print(f"Gradient cross entropy: \n {grad_ce}")
        print(f"Gradient non differential ordinal cross entropy: \n {grad_oce}")

        torch.testing.assert_close(grad_ce, grad_oce)

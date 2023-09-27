import pytest
import torch
from torch import nn

from l1_dim_pruner.pruner import L1DimUnstructured, l1_dim_unstructured


class TestL1DimUnstructured:
    @pytest.mark.parametrize(
        "proportion, dim, input_tensor, expected_output",
        [
            (0.5, 1, torch.tensor([[2.0]]), torch.tensor([[1.0]])),
            (0.8, 1, torch.tensor([[2.0]]), torch.tensor([[0.0]])),
            (1.0, 1, torch.tensor([[2.0]]), torch.tensor([[0.0]])),
            (0.5, 1, torch.tensor([[2.0, 3.0]]), torch.tensor([[0.0, 1.0]])),
            (0.5, 0, torch.tensor([[2.0], [3.0]]), torch.tensor([[0.0], [1.0]])),
            (
                0.5,
                1,
                torch.tensor([[2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 5.0, 3.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]]),
            ),
            (
                0.5,
                0,
                torch.tensor([[2.0, 2.0], [3.0, 4.0], [4.0, 5.0], [5.0, 3.0]]),
                torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
            ),
            (
                0.5,
                1,
                torch.tensor([[-0.1, -0.9, 1.0, 0.7, 0.8], [0.0, 0.2, 0.2, -0.1, -0.3]]),
                torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0, 1.0]]),
            ),
        ],
    )
    def test_compute_mask(self, proportion, dim, input_tensor, expected_output):
        default_mask = torch.ones_like(input_tensor)

        pruner = L1DimUnstructured(proportion=proportion, pruning_dim=dim)
        result = pruner.compute_mask(t=input_tensor, default_mask=default_mask)

        assert result.shape == input_tensor.shape
        assert result.dtype == default_mask.dtype
        assert torch.all(result == expected_output)


@pytest.mark.parametrize(
    "proportion, dim, input_tensor, expected_output",
    [
        (0.5, 1, torch.tensor([[2.0, 3.0]]), torch.tensor([[0.0, 3.0]])),
        (0.5, 0, torch.tensor([[2.0], [3.0]]), torch.tensor([[0.0], [3.0]])),
    ],
)
def test_l1_dim_unstructured(proportion, dim, input_tensor, expected_output):
    layer = nn.Linear(input_tensor.shape[1], input_tensor.shape[0])
    layer.weight.data = input_tensor
    l1_dim_unstructured(layer, "weight", proportion=proportion, pruning_dim=dim)
    assert torch.all(layer.weight.data == expected_output)

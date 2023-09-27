import torch
from torch import Tensor
from torch.nn.utils import prune


class L1DimUnstructured(prune.BasePruningMethod):
    """
    Prunes the specified proportion of parameters with the lowest L1-norm in a specific
    dimension of a tensor.

    This is similar to the `torch.nn.utils.prune.L1Unstructured` pruner, except that it
    operates on a specific dimension of a tensor, rather than the entire tensor.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, proportion: float, pruning_dim: int) -> None:
        """
        :param proportion: This should be a float between 0.0 and 1.0, and represents the
            proportion of parameters to prune.

        :param pruning_dim: The dimension in which to prune.
        """
        super().__init__()
        assert 0.0 <= proportion <= 1.0, "Proportion must be between 0.0 and 1.0"
        self._amount = proportion
        self._pruning_dim = pruning_dim

    def compute_mask(self, t: Tensor, default_mask: Tensor):
        assert t.shape == default_mask.shape, "Tensor and mask must have the same shape"

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        num_to_prune_per_channel = round(self._amount * t.shape[self._pruning_dim])

        if num_to_prune_per_channel != 0:
            top_k = torch.topk(
                torch.abs(t), k=num_to_prune_per_channel, dim=self._pruning_dim, largest=False
            )
            zeros = torch.zeros_like(top_k.indices).to(mask.dtype)
            mask = torch.scatter(mask, dim=self._pruning_dim, index=top_k.indices, src=zeros)

        return mask


def l1_dim_unstructured(
    module: torch.nn.Module,
    name: str,
    proportion: float,
    pruning_dim: int,
):
    """
    Applies pruning to a given module parameter using the `L1ChannelUnstructured` pruner.
    See that class for more details.
    """
    L1DimUnstructured.apply(
        module,
        name,
        proportion=proportion,
        pruning_dim=pruning_dim,
    )
    return module

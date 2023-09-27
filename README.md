Prunes the specified proportion of parameters with the lowest L1-norm in a specific
dimension of a tensor.

This is similar to [torch.nn.utils.prune.l1_unstructured](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html#torch-nn-utils-prune-l1-unstructured),
except that it operates on a specific dimension of a tensor, rather than the entire tensor.

Usage:
```python
from l1_dim_pruner.pruner import l1_dim_unstructured
l1_dim_unstructured(layer, "weight", proportion=0.5, pruning_dim=1)
```

This prunes 50% of the values of "weight" parameter in the `layer` module, using
a pruning dimension of 1 (i.e. pruning the second dimension of the weight tensor).

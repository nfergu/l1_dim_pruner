
# L1 Dim Pruner

Prunes the specified proportion of parameters with the lowest L1-norm in a specific
dimension of a tensor.

This is similar to [torch.nn.utils.prune.l1_unstructured](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html#torch-nn-utils-prune-l1-unstructured),
except that it operates on a specific dimension of a tensor, rather than the entire tensor.

This outperforms standard L1-norm pruning when pruning the fully-connected layers of a GPT-2 model. See [Benchmarks](#benchmarks) for more details.

## Installation

```
pip install git+https://github.com/nfergu/l1_dim_pruner.git
```

## Usage

```python
from l1_dim_pruner.pruner import l1_dim_unstructured
l1_dim_unstructured(module, name="weight", proportion=0.5, pruning_dim=1)
```

This prunes 50% of the values of "weight" parameter in the `layer` module, using
a pruning dimension of 1 (i.e. pruning along the second dimension of the weight tensor).

## Benchmarks

|Benchmark|Unpruned Validation Loss|[l1_unstructured (PyTorch)](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html#torch-nn-utils-prune-l1-unstructured) Validation Loss|L1 Dim Pruner (this repo) Validation Loss|
|:--------:|:--------:|:--------:|:--------:|
|[GPT-2 (nanoGPT)](https://github.com/nfergu/nanogpt_l1_dim_pruner)<sup>1</sup>|3.0959|4.3384|**3.5679**|

<sup>1</sup> Pruning 50% of the weights in the fully-connected layers of a GPT-2 model. See [this repo](https://github.com/nfergu/nanogpt_l1_dim_pruner) for the benchmark code.

TODO: More benchmarks...

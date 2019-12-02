# filter-response-normalization-layer-pytorch (FRN)

Unofficial PyTorch implementation of Filter Response Normalization Layer.

## Make a preact ResNet50 model with FRN Layer

```
from preact_resnet import preact_resnet50_frn
model = preact_resnet50_frn(num_classes=1000)
```

## Reference

[Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks](https://arxiv.org/abs/1911.09737)

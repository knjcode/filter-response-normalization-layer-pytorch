import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preact_resnet import *

model_dict = {
    'preact_resnet18': preact_resnet18,
    'preact_resnet34': preact_resnet34,
    'preact_resnet50': preact_resnet50,
    'preact_resnet101': preact_resnet101,
    'preact_resnet152': preact_resnet152,
    'preact_resnet200': preact_resnet200,
    'preact_resnet18_frn': preact_resnet18_frn,
    'preact_resnet34_frn': preact_resnet34_frn,
    'preact_resnet50_frn': preact_resnet50_frn,
    'preact_resnet101_frn': preact_resnet101_frn,
    'preact_resnet152_frn': preact_resnet152_frn,
    'preact_resnet200_frn': preact_resnet200_frn,
}

input = torch.randn(2,3,224,224)

for model_name, model in model_dict.items():
    print("check:", str(model_name))
    current_model = model(num_classes=10)
    output = current_model(input)

print('ok')

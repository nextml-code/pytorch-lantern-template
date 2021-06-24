from typing import List
import torch
import torch.nn as nn
from lantern import module_device

from {{cookiecutter.package_name}} import model


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),
            nn.Flatten(start_dim=1),
            nn.Linear(12544, 128),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(128, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, prepared):
        return model.PredictionBatch(logits=self.logits(prepared))

    def predictions(self, features: List[model.StandardizedImage]):
        return self.forward(
            torch.stack([feature.data for feature in features]).to(module_device(self))
        )


Model.StandardizedImage = model.StandardizedImage
Model.PredictionBatch = model.PredictionBatch

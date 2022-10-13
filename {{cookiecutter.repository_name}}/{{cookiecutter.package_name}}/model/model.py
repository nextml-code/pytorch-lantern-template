from typing import List
import torch
import torch.nn as nn
from lantern import module_device, Tensor, Numpy

from .standardize import standardize
from .resize import resize
from .prediction_batch import PredictionBatch


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
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

    @property
    def device(self):
        return module_device(self)

    def forward(self, prepared: Tensor.dims("NCHW").float()) -> PredictionBatch:
        return PredictionBatch(logits=self.network(prepared))

    def predictions_(self, images: List[Numpy.dims("HWC").uint8()]) -> PredictionBatch:
        """
        This method has side effects as it modifies for example batch norm layers
        if training is enabled.
        """
        return self.forward(
            torch.stack([standardize(resize(image)) for image in images]).to(
                module_device(self)
            )
        )

    def predictions(self, images: List[Numpy.dims("HWC").uint8()]) -> PredictionBatch:
        if self.training:
            raise RuntimeError(
                "Model is in training mode when calling `predictions`. Use `predictions_` instead."
            )
        return self.predictions_(images)

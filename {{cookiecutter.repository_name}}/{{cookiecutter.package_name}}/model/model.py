from typing import List
import numpy as np
import torch
import torch.nn as nn
from pydantic import validate_arguments
from lantern import module_device, Tensor, Numpy

from .standardized_image import StandardizedImage
from .prediction import PredictionBatch


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

    @property
    def device(self):
        return module_device(self)

    @validate_arguments
    def forward(self, prepared: Tensor.dims("NCHW")):
        return PredictionBatch(logits=self.logits(prepared))

    @validate_arguments
    def predictions(self, images: List[Numpy.dims("HWC").dtype(np.uint8)]):
        return self.forward(
            torch.stack(
                [StandardizedImage.from_image(image).data for image in images]
            ).to(self.device)
        )


Model.StandardizedImage = StandardizedImage
Model.PredictionBatch = PredictionBatch

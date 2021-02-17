from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from lantern import ModuleCompose, module_device

from {{cookiecutter.package_name}} import architecture


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = ModuleCompose(
            nn.Conv2d(3, 32, 3, 1),
            F.relu,
            nn.Conv2d(32, 64, 3, 1),
            F.relu,
            partial(F.max_pool2d, kernel_size=2),
            nn.Dropout2d(0.25),
            partial(torch.flatten, start_dim=1),
            nn.Linear(12544, 128),
            F.relu,
            nn.Dropout2d(0.5),
            nn.Linear(128, 10),
            partial(F.log_softmax, dim=1),
        )

    def forward(self, prepared):
        return architecture.PredictionBatch(logits=self.logits(prepared))

    def predictions(self, feature_batch: architecture.FeatureBatch):
        return self.forward(feature_batch.stack().to(module_device(self)))

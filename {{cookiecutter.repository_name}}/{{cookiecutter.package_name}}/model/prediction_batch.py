import numpy as np
import torch
import torch.nn.functional as F
from lantern import FunctionalBase, Tensor

from .prediction import Prediction


class PredictionBatch(FunctionalBase):
    logits: Tensor.dims("NC").float()

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, index):
        return Prediction(
            logits=self.logits[index],
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @property
    def probabilities(self):
        return self.logits.detach().cpu().sigmoid()

    def stack_class_indices(self, examples):
        return torch.as_tensor(
            np.stack([example.class_index for example in examples]),
            device=self.logits.device,
        )

    def loss(self, examples):
        return self.cross_entropy(examples)

    def cross_entropy(self, examples):
        return F.cross_entropy(
            self.logits,
            self.stack_class_indices(examples),
        )

    def accuracy(self, examples):
        return self.logits.argmax(dim=1) == self.stack_class_indices(examples)

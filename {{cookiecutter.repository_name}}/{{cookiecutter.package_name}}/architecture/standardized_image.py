from typing import List
import numpy as np
import torch
from PIL import Image
from lantern import FunctionalBase, Tensor
from lantern.functional import starcompose

from {{cookiecutter.package_name}} import settings, problem, architecture


def standardize(image: np.ndarray) -> Tensor:
    return torch.as_tensor(image).permute(2, 0, 1).float() / 255 * 2 - 1


class StandardizedImage(FunctionalBase):
    data: Tensor

    @staticmethod
    def from_image(image: Image.Image):
        return StandardizedImage(data=standardize(architecture.resize(image=image)))

    @staticmethod
    def from_example(example: problem.Example):
        resized_example = example.augment(architecture.resize)
        return resized_example, StandardizedImage.from_image(resized_example.image)

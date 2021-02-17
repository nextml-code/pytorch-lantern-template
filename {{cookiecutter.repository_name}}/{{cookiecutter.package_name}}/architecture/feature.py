from typing import List
import numpy as np
import torch
from PIL import Image
from lantern import FunctionalBase
from lantern.functional import starcompose

from {{cookiecutter.package_name}} import settings, problem


def resized(image: Image.Image):
    return image.resize(settings.input_size)


def standardized(image: Image.Image):
    return (
        torch.as_tensor(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255 * 2
        - 1
    )


prepared = starcompose(
    resized,
    standardized,
)


class Feature(FunctionalBase):
    data: torch.Tensor

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    @staticmethod
    def from_image(image: Image.Image):
        return Feature(data=prepared(image))


class FeatureBatch(FunctionalBase):
    features: List[Feature]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    @staticmethod
    def from_examples(examples: List[problem.Example]):
        return FeatureBatch.from_images([example.image for example in examples])

    @staticmethod
    def from_images(images: List[Image.Image]):
        return FeatureBatch(features=[Feature.from_image(image) for image in images])

    def stack(self):
        return torch.stack([feature.data for feature in self.features])

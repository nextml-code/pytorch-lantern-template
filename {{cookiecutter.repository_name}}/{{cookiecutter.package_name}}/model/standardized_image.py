from PIL import Image
import torch
from lantern import FunctionalBase, Tensor, Numpy

from .resize import resize


def standardize(image: Numpy) -> Tensor:
    return torch.as_tensor(image).permute(2, 0, 1).float() / 255 * 2 - 1


class StandardizedImage(FunctionalBase):
    data: Tensor

    @staticmethod
    def from_image(image: Image.Image):
        return StandardizedImage(data=standardize(resize(image=image)))

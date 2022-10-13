import torch
from lantern import Tensor, Numpy


def standardize(image: Numpy.dims("HWC").uint8()) -> Tensor.dims("NCHW").float():
    return torch.as_tensor(image).permute(2, 0, 1).float() / 255 * 2 - 1

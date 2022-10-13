from typing import Union
from lantern import Numpy
from PIL import Image
import numpy as np
import imgaug

from {{cookiecutter.package_name}} import settings


resizer = imgaug.augmenters.Resize(
    dict(
        height=settings.INPUT_HEIGHT,
        width=settings.INPUT_WIDTH,
    )
)

def resize(image: Union[Numpy.dims("HW"), Image.Image]):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    return resizer.augment_image(image)

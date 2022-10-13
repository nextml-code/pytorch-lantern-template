from itertools import product
from PIL import Image, ImageDraw, ImageFont
from lantern import FunctionalBase, Tensor

from {{cookiecutter.package_name}} import settings


class Prediction(FunctionalBase):
    logits: Tensor.dims("C").float()

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    @property
    def probabilities(self):
        return self.logits.detach().cpu().sigmoid()

    @property
    def class_name(self):
        return settings.CLASS_NAMES[self.logits.argmax()]

    def representation(self, example=None):
        if example:
            image = Image.fromarray(example.image).resize((256, 256))
        else:
            image = Image.new("RGB", (256, 256))

        probabilities = dict(
            zip(
                settings.CLASS_NAMES,
                self.probabilities,
            )
        )

        draw = ImageDraw.Draw(image)
        for index, (class_name, probability) in enumerate(probabilities.items()):
            text_(draw, f"{class_name}: {probability:.2f}", 10, 5 + 10 * index)
        return image

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_


def text_(draw, text, x, y, fill="black", outline="white", size=12):
    try:
        font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", size)
    except OSError:
        font = ImageFont.load_default()

    for x_shift, y_shift in product([-1, 0, 1], [-1, 0, 1]):
        draw.text((x + x_shift, y + y_shift), text, font=font, fill=outline)

    draw.text((x, y), text, font=font, fill=fill)

import imgaug

from {{cookiecutter.package_name}} import settings


resize = imgaug.augmenters.Resize(
    dict(
        height=settings.INPUT_HEIGHT,
        width=settings.INPUT_WIDTH,
    )
)

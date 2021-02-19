from {{cookiecutter.package_name}} import settings
import imgaug


resize = imgaug.augmenters.Resize(
    dict(
        height=settings.INPUT_HEIGHT,
        width=settings.INPUT_WIDTH,
    )
)

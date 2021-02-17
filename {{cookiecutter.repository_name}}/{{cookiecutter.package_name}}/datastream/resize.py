from {{cookiecutter.package_name}} import settings
import imgaug


resize = imgaug.augmenters.Resize(
    dict(
        height=settings.input_size[1],
        width=settings.input_size[0],
    )
)

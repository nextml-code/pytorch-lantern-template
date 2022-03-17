"""
Saving images and labels to disk to simulate a more realistic use case
"""
import argparse
from pathlib import Path
import pandas as pd
import torchvision

from {{cookiecutter.package_name}} import settings

# more realistic to have natural class names
CLASS_TO_NAME = dict(zip(range(10), settings.CLASS_NAMES))
CACHE_ROOT = "/tmp/cifar10-template-cache"


def image_path(index):
    return Path(f"images/{index}.png")


def dataframe(dataset):
    return (
        pd.DataFrame(
            dict(
                index=range(len(dataset)),
                number=dataset.targets,
            )
        )
        .assign(
            class_name=lambda df: (df["number"].map(CLASS_TO_NAME)),
            image_path=lambda df: df["index"].apply(
                lambda index: image_path(index)
            ),
        )[["index", "image_path", "class_name"]]
        .sample(n=128)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    dataset = torchvision.datasets.CIFAR10(CACHE_ROOT, train=True, download=True)
    Path("images").mkdir()
    for index, (image, _) in enumerate(dataset):
        image.save(image_path(index))
    dataframe(dataset).to_csv("data.csv")

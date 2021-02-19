from datastream import Datastream

from {{cookiecutter.package_name}} import datastream
from {{cookiecutter.package_name}}.problem import settings


def TrainDatastream():
    dataset = datastream.datasets()["train"]
    augmenter = datastream.augmenter()
    return Datastream.merge(
        [
            Datastream(dataset.subset(lambda df: df["class_name"] == class_name))
            for class_name in settings.CLASS_NAMES
        ]
    ).map(lambda example: example.augment(augmenter))

from datastream import Datastream

from .datasets import datasets
from .augmenter import augmenter
from {{cookiecutter.package_name}} import settings


def TrainDatastream():
    train_dataset = datasets()["train"]
    augmenter_ = augmenter()
    return Datastream.merge(
        [
            Datastream(train_dataset.subset(lambda df: df["class_name"] == class_name))
            for class_name in settings.CLASS_NAMES
        ]
    ).map(lambda example: example.augment(augmenter_))

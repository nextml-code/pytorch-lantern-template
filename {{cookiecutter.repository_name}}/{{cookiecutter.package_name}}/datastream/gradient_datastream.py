from datastream import Datastream

from mnist_template import datastream
from mnist_template.problem import settings


def GradientDatastream():
    dataset = datastream.datasets()["gradient"]
    augmenter = datastream.augmenter()
    return Datastream.merge(
        [
            Datastream(dataset.subset(lambda df: df["class_name"] == class_name))
            for class_name in settings.CLASS_NAMES
        ]
    ).map(lambda example: example.augment(augmenter))

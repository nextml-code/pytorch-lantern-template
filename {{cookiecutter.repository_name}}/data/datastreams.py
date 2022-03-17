from datastream import Datastream, samplers
from imgaug import augmenters as iaa

from .datasets import datasets
from {{cookiecutter.package_name}} import settings


def datastreams():
    datasets_ = datasets()
    augmenter_ = augmenter()

    return dict(
        train=(
            Datastream.merge(
                [
                    Datastream(datasets_["train"].subset(lambda df: df["class_name"] == class_name))
                    for class_name in settings.CLASS_NAMES
                ]
            ).map(lambda example: example.augment(augmenter_))
        ),
        **{
            f"evaluate_{split_name}": Datastream(dataset, samplers.SequentialSampler(len(dataset)))
            for split_name, dataset in datasets_.items()
        },
    )


def augmenter():
    return iaa.Sequential(
        [
            iaa.Sometimes(0.5,
                iaa.Affine(
                    scale=(0.9, 1.0),
                    translate_percent=dict(
                        x=(-0.02, 0.02),
                        y=(-0.02, 0.02),
                    ),
                    rotate=(-3, 3),
                ),
            ),
            iaa.HorizontalFlip(0.5),
        ],
        random_order=True,
    )

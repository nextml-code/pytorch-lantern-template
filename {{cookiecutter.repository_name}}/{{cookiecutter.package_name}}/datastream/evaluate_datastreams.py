from datastream import Datastream

from {{cookiecutter.package_name}} import datastream
from {{cookiecutter.package_name}}.datastream import resize


def evaluate_datastreams():
    return {
        split_name: Datastream(dataset).map(lambda example: example.augment(resize))
        for split_name, dataset in datastream.datasets().items()
    }

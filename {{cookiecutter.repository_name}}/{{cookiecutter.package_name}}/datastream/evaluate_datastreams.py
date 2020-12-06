from datastream import Datastream

from {{cookiecutter.package_name}} import datastream


def evaluate_datastreams():
    return {
        split_name: Datastream(dataset)
        for split_name, dataset in datastream.datasets().items()
    }
